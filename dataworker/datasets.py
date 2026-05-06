"""
Dataset for SFT of PairEsm3Qwen3ForCausalLM on PPI description generation.

Each sample carries two proteins and an interaction description. Reads either a parquet or a jsonl (auto-detected),
loads pre-encoded protein tensors for both proteins, and returns raw text components. The collator builds the final
chat template — the per-sample interaction-chunk token count depends on batch-level padding.

Supported input formats:
  (1) released parquet (ppi2text_dataset.parquet) with columns `pair_id` ("<acc_a>_<acc_b>") and `response`.
      `pair_id` is split on "_" to derive `uniprot_a`/`uniprot_b`.
  (2) jsonl, one record per line, required keys: `pair_id`, `acc_a` or `uniprot_a`, `acc_b` or `uniprot_b`,
      `answer` or `response`. Optional: `reasoning`.

Pass `read_list_path=<split>.list` to select pair_ids for a split. Isoform accessions ("P41778-2") map to their
canonical parent ("P41778") for tensor/embedding lookup.

L (including BOS/EOS) is always read from the first dim of the on-disk .pt — embedding mode: tensor.shape[0]; tokens
mode: sequence_tokens.shape[0]. Anything downstream (pad_target / compressed_len / sequence_mask) is derived from
this L so the mask and the tensor the model consumes can't drift apart.

Per-accession lengths are cached at <scan_dir>/_tensor_lengths.json on first load — delete to force a rebuild.
"""

import json
import os
from typing import Iterator

from esm.tokenization import TokenizerCollection
from esm.utils.constants import esm3 as C
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from model.modeling_pairesm3qwen3 import SequenceCompressor


def _iter_records(path: str) -> Iterator[dict]:
    """Yield one record dict per sample from a Parquet or JSONL file (auto-detected)."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        cols = {name: table.column(name).to_pylist() for name in table.column_names}
        n = table.num_rows
        if "pair_id" not in cols or "response" not in cols:
            raise ValueError(
                f"Parquet at {path} must have at least 'pair_id' and 'response' columns; "
                f"got {sorted(cols.keys())}"
            )
        for i in range(n):
            record = {name: values[i] for name, values in cols.items()}
            if "uniprot_a" not in record and "acc_a" not in record:
                acc_a, acc_b = record["pair_id"].split("_", 1)
                record["uniprot_a"] = acc_a
                record["uniprot_b"] = acc_b
            yield record
    else:
        with open(path, "r") as f:
            for line in f:
                yield json.loads(line)


def _count_records(path: str) -> int:
    """Count records for tqdm — cheap for parquet, requires line scan for JSONL."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    with open(path, "r") as f:
        return sum(1 for _ in f)


def _compute_pad_target_len(compressed_len: int) -> int:
    """Maximum input length L that compresses to exactly `compressed_len` under two stride-2 k=4 p=1 conv layers.

    For these conv params, compute_compressed_length(L) = C for L in [4C, 4C+3]. Padding to 4C+3 (the maximum) keeps
    the compressed output deterministic regardless of batch composition.
    """
    return 4 * compressed_len + 3


def _scan_tensor_lengths(
        scan_dir: str,
        prefix: str,
        extract_length,
        cache_basename: str = "_tensor_lengths.json",
) -> dict[str, int]:
    """Return {accession: L} for every `<prefix><acc>.pt` file in scan_dir, cached on disk.

    Loads <scan_dir>/<cache_basename> if present; otherwise scans every .pt, calls extract_length(loaded_obj) -> int,
    and writes the cache via atomic rename so concurrent ranks can race safely.
    """
    cache_path = os.path.join(scan_dir, cache_basename)
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    filenames = [fn for fn in os.listdir(scan_dir) if fn.startswith(prefix) and fn.endswith(".pt")]
    lengths: dict[str, int] = {}
    for fn in tqdm(filenames, desc=f"Scanning {prefix}*.pt lengths"):
        acc = fn[len(prefix):-3]
        obj = torch.load(os.path.join(scan_dir, fn), map_location="cpu", weights_only=True)
        lengths[acc] = int(extract_length(obj))

    tmp_path = cache_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(lengths, f)
    os.replace(tmp_path, cache_path)
    return lengths


class PairSFTDataset(Dataset):
    """
    SFT dataset for PPI description generation.

    Each sample returns raw text components and pre-encoded protein tensors for both proteins; the collator computes
    batch-level interaction-chunk token counts and builds the final chat-template prompt.

    Per sample: pair_id, question_text, answer_text ("<think>\\n...\\n</think>\\n\\n....<|im_end|>"),
    seq_len_a/b (incl. BOS/EOS), compressed_len_a/b, pad_target_a/b, protein_a_*/protein_b_*.
    """

    PROTEIN_FIELDS = [
        "sequence_tokens", "structure_tokens", "ss8_tokens", "sasa_tokens",
        "function_tokens", "residue_annotation_tokens",
        "average_plddt", "per_res_plddt", "structure_coords", "chain_id",
    ]

    def __init__(
            self,
            read_data_path: str,
            read_pt_dir: str,
            text_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
            protein_tokenizers: TokenizerCollection | None = None,
            max_seq_len: int = 2040,
            question_statement: str = "Describe the interaction between Protein A and Protein B.",
            read_emb_dir: str | None = None,
            read_list_path: str | None = None,
            proteins_jsonl_path: str | None = None,
    ):
        super().__init__()
        self.read_pt_dir = read_pt_dir
        self.read_emb_dir = read_emb_dir
        self.text_tokenizer = text_tokenizer
        self.protein_tokenizers = protein_tokenizers
        self.max_seq_len = max_seq_len
        self.question_statement = question_statement

        if read_emb_dir is None and protein_tokenizers is None:
            raise ValueError("protein_tokenizers is required when not using pre-computed embeddings")

        if proteins_jsonl_path is not None:
            print(f"[PairSFTDataset] note: proteins_jsonl_path={proteins_jsonl_path!r} is ignored; "
                  f"sequence length is derived from the on-disk tensor files.")

        # min sequence length must be long enough for two conv layers:
        # conv1(k=4,s=2,p=1) needs L>=4, conv2 needs l1>=4 → L>=10
        self.min_seq_len = 10

        # optional pair_id allow-list (one pair_id per line)
        allowed_pair_ids: set[str] | None = None
        if read_list_path is not None:
            with open(read_list_path, "r") as f:
                allowed_pair_ids = {ln.strip() for ln in f if ln.strip()}
            print(f"Loaded {len(allowed_pair_ids)} pair_ids from {read_list_path}")

        # scan on-disk tensors once and build {accession: L}; L is the exact first-dim of the tensor the model will
        # consume, so compressed_len/pad_target/sequence_mask downstream stay consistent with it
        if read_emb_dir is not None:
            scan_dir = read_emb_dir
            prefix = "esm3emb_"
            extract = lambda obj: obj.shape[0]
            cache_name = "_emb_lengths.json"
        else:
            scan_dir = read_pt_dir
            prefix = "esm3enc_"
            extract = lambda obj: obj["sequence_tokens"].shape[0]
            cache_name = "_enc_lengths.json"
        acc_to_L = _scan_tensor_lengths(scan_dir, prefix, extract, cache_name)
        print(f"[PairSFTDataset] resolved L for {len(acc_to_L)} accessions from {scan_dir}")

        n_skipped_not_in_list = 0
        n_skipped_missing_tensor = 0
        n_skipped_bad_len = 0
        self.samples: list[dict] = []
        total_records = _count_records(read_data_path)
        for raw in tqdm(_iter_records(read_data_path), total=total_records,
                        desc="Loading samples", postfix=read_data_path):
            if allowed_pair_ids is not None and raw.get("pair_id") not in allowed_pair_ids:
                n_skipped_not_in_list += 1
                continue

            # normalize accession fields (legacy: acc_a/b; new: uniprot_a/b);
            # isoforms ("P41778-2") share a canonical parent file, strip the suffix
            acc_a_raw = raw.get("acc_a") or raw.get("uniprot_a")
            acc_b_raw = raw.get("acc_b") or raw.get("uniprot_b")
            acc_a = acc_a_raw.split("-", 1)[0]
            acc_b = acc_b_raw.split("-", 1)[0]

            seq_len_a = acc_to_L.get(acc_a)
            seq_len_b = acc_to_L.get(acc_b)
            if seq_len_a is None or seq_len_b is None:
                n_skipped_missing_tensor += 1
                continue

            if not (self.min_seq_len <= seq_len_a <= max_seq_len
                    and self.min_seq_len <= seq_len_b <= max_seq_len):
                n_skipped_bad_len += 1
                continue

            # resolve text fields (legacy: reasoning + answer; new: response)
            answer = raw.get("answer")
            if answer is None:
                answer = raw.get("response")
                if answer is None:
                    raise ValueError(
                        f"Sample {raw.get('pair_id')} has neither 'answer' nor 'response'"
                    )
            reasoning = raw.get("reasoning", "")

            self.samples.append({
                "pair_id": raw["pair_id"],
                "acc_a": acc_a,
                "acc_b": acc_b,
                "seq_len_a": seq_len_a,
                "seq_len_b": seq_len_b,
                "reasoning": reasoning,
                "answer": answer,
            })

        print(f"PairSFTDataset loaded {len(self.samples)} samples"
              f"  (skipped: not_in_list={n_skipped_not_in_list}, "
              f"bad_len={n_skipped_bad_len}, "
              f"missing_tensor={n_skipped_missing_tensor})")

    def __len__(self):
        return len(self.samples)

    def _load_protein_tensors(self, accession: str, seq_len: int) -> dict[str, torch.Tensor]:
        """Load pre-encoded protein tensors from disk and fill default values for missing fields."""
        encoded_tensor_path = os.path.join(self.read_pt_dir, f"esm3enc_{accession}.pt")
        encoded = torch.load(encoded_tensor_path)

        defaults = lambda x, tok: (
            torch.full((seq_len,), tok, dtype=torch.long) if x is None else x
        )

        sequence_tokens = defaults(encoded["sequence_tokens"], self.protein_tokenizers.sequence.mask_token_id)
        structure_tokens = defaults(encoded["structure_tokens"], self.protein_tokenizers.structure.mask_token_id)
        ss8_tokens = defaults(encoded["ss8_tokens"], C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(encoded["sasa_tokens"], C.SASA_PAD_TOKEN)

        function_tokens = torch.full((seq_len, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long)
        residue_annotation_tokens = torch.full((seq_len, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long)
        average_plddt = torch.full(
            (seq_len,),
            encoded["average_plddt"].item() if encoded["average_plddt"] is not None else 1.0,
            dtype=torch.float32,
        )
        per_res_plddt = defaults(encoded["per_res_plddt"], 0.0).float()

        if encoded["structure_coords"] is None:
            structure_coords = torch.full((seq_len, 3, 3), float("nan"), dtype=torch.float)
        else:
            structure_coords = encoded["structure_coords"].float()

        chain_id = defaults(encoded["chain_id"], 0)

        return {
            "sequence_tokens": sequence_tokens,
            "structure_tokens": structure_tokens,
            "ss8_tokens": ss8_tokens,
            "sasa_tokens": sasa_tokens,
            "function_tokens": function_tokens,
            "residue_annotation_tokens": residue_annotation_tokens,
            "average_plddt": average_plddt,
            "per_res_plddt": per_res_plddt,
            "structure_coords": structure_coords,
            "chain_id": chain_id,
        }

    def _load_protein_embedding(self, accession: str, expected_L: int) -> torch.Tensor:
        """Load pre-computed ESM3 embedding from disk. Shape: (L, 1536) in bfloat16."""
        emb_path = os.path.join(self.read_emb_dir, f"esm3emb_{accession}.pt")
        emb = torch.load(emb_path, weights_only=True)
        if emb.shape[0] != expected_L:
            raise RuntimeError(
                f"embedding length drift for {accession}: "
                f"manifest says {expected_L}, file has {emb.shape[0]}. "
                f"Delete {os.path.join(self.read_emb_dir, '_emb_lengths.json')} to rebuild."
            )
        return emb

    def __getitem__(self, idx):
        sample = self.samples[idx]

        seq_len_a = sample["seq_len_a"]
        seq_len_b = sample["seq_len_b"]

        compressed_len_a = SequenceCompressor.compute_compressed_length(seq_len_a)
        compressed_len_b = SequenceCompressor.compute_compressed_length(seq_len_b)
        pad_target_a = _compute_pad_target_len(compressed_len_a)
        pad_target_b = _compute_pad_target_len(compressed_len_b)

        if self.read_emb_dir is not None:
            protein_a_emb = self._load_protein_embedding(sample["acc_a"], seq_len_a)
            protein_b_emb = self._load_protein_embedding(sample["acc_b"], seq_len_b)
        else:
            protein_a = self._load_protein_tensors(sample["acc_a"], seq_len_a)
            protein_b = self._load_protein_tensors(sample["acc_b"], seq_len_b)

        answer_text = "<think>\n" + sample["reasoning"] + "\n</think>\n\n" + sample["answer"] + "<|im_end|>"

        result = {
            "pair_id": sample["pair_id"],
            "question_text": self.question_statement,
            "answer_text": answer_text,
            "seq_len_a": seq_len_a,
            "seq_len_b": seq_len_b,
            "compressed_len_a": compressed_len_a,
            "compressed_len_b": compressed_len_b,
            "pad_target_a": pad_target_a,
            "pad_target_b": pad_target_b,
        }

        if self.read_emb_dir is not None:
            result["protein_a_embedding"] = protein_a_emb
            result["protein_b_embedding"] = protein_b_emb
        else:
            for key, val in protein_a.items():
                result[f"protein_a_{key}"] = val
            for key, val in protein_b.items():
                result[f"protein_b_{key}"] = val

        return result
