"""
Collator for PairEsm3Qwen3ForCausalLM training and inference.

Pads protein_a/b to deterministic targets so compressed output lengths are batch-independent, builds the chat-template
prompt with per-sample <|roi|> token groups (protein A inline, protein B inline, pair map inside <|boi|>/<|eoi|>),
tokenizes the text (left-pad prompts, right-pad answers), and packages protein inputs into per-protein dicts.
"""

import math
from typing import Literal

from esm.tokenization import TokenizerCollection
from esm.utils.constants import esm3 as C
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# pad token value per ESM3 field
_PROTEIN_PAD = {
    "sequence_tokens": None,  # filled dynamically from the tokenizer
    "structure_tokens": None,
    "ss8_tokens": C.SS8_PAD_TOKEN,
    "sasa_tokens": C.SASA_PAD_TOKEN,
    "function_tokens": C.INTERPRO_PAD_TOKEN,
    "residue_annotation_tokens": C.RESIDUE_PAD_TOKEN,
    "average_plddt": 1.0,
    "per_res_plddt": 0.0,
    "structure_coords": float("nan"),
    "chain_id": 0,
}


class PairSFTCollator:
    """
    Dynamic-padding collator for PPI samples.

    Uses pad_target_a/b from the dataset so the Conv1d-compressed output length stays the same regardless of batch
    composition. Each sample gets its own number of <|roi|> tokens based on its own compressed lengths.

    train_val mode returns: input_ids, attention_mask, labels (teacher-forcing), protein_a/b dicts (padded ESM3
    tensors + sequence_id + sequence_mask), compressed_len_a/b [B] long.
    inference mode returns: pair_ids list, input_ids/attention_mask (prompt only), labels (ground-truth answer for
    eval, not used in generation), same protein_a/b and compressed_len_a/b.
    """

    PROTEIN_FIELDS = [
        "sequence_tokens", "structure_tokens", "ss8_tokens", "sasa_tokens",
        "function_tokens", "residue_annotation_tokens",
        "average_plddt", "per_res_plddt", "structure_coords", "chain_id",
    ]

    def __init__(
            self,
            protein_tokenizers: TokenizerCollection | None = None,
            text_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = None,
            pair_map_n_tokens: int = 256,
            mode: Literal["train_val", "inference"] = "train_val",
            boi_token: str = "<|boi|>",
            eoi_token: str = "<|eoi|>",
            roi_token: str = "<|roi|>",
            max_answer_length: int = 2048,
    ):
        self.protein_tokenizers = protein_tokenizers
        self.text_tokenizer = text_tokenizer
        self.pair_map_n_tokens = pair_map_n_tokens
        self.mode = mode
        self.boi_token = boi_token
        self.eoi_token = eoi_token
        self.roi_token = roi_token
        self.max_answer_length = max_answer_length

        # fill dynamic pad values (only needed for token mode)
        self.pad_values = dict(_PROTEIN_PAD)
        if protein_tokenizers is not None:
            self.pad_values["sequence_tokens"] = protein_tokenizers.sequence.pad_token_id
            self.pad_values["structure_tokens"] = protein_tokenizers.structure.pad_token_id

    @staticmethod
    def _pad_batch(
            data_list: list[dict],
            key: str,
            pad_token: int | float,
            max_length: int | None = None,
    ) -> torch.Tensor:
        """Pad a list of tensors to the longest in the batch, optionally extending to max_length."""
        padded = torch.nn.utils.rnn.pad_sequence(
            [data[key] for data in data_list],
            batch_first=True,
            padding_value=pad_token,
            padding_side="right",
        )
        if max_length is not None and padded.size(1) < max_length:
            pad_len = max_length - padded.size(1)
            pad_shape = (padded.size(0), pad_len) + padded.size()[2:]
            padding = torch.full(pad_shape, pad_token, dtype=padded.dtype, device=padded.device)
            padded = torch.cat([padded, padding], dim=1)
        return padded

    def _pad_protein(self, data_list: list[dict], prefix: str, pad_target_key: str) -> dict[str, torch.Tensor]:
        """Pad all protein fields for a prefix ('protein_a' or 'protein_b') and build masks."""
        # batch-level padding target derived from per-sample pad_targets
        padding_target_len = max(data[pad_target_key] for data in data_list)

        # rename keys: "protein_a_sequence_tokens" -> "sequence_tokens"
        renamed_list = []
        for data in data_list:
            renamed = {}
            for field in self.PROTEIN_FIELDS:
                renamed[field] = data[f"{prefix}_{field}"]
            renamed_list.append(renamed)

        result = {}
        for field in self.PROTEIN_FIELDS:
            result[field] = self._pad_batch(renamed_list, field, self.pad_values[field], padding_target_len)

        # build sequence_id and sequence_mask from padded sequence_tokens
        pad_mask = result["sequence_tokens"] != self.pad_values["sequence_tokens"]
        bsz = result["sequence_tokens"].size(0)
        row_indices = torch.arange(1, bsz + 1, dtype=torch.long).unsqueeze(1)
        result["sequence_id"] = row_indices * pad_mask.long()
        result["chain_id"] = result["sequence_id"].clone()
        result["sequence_mask"] = pad_mask.to(torch.long)

        return result

    def _pad_protein_embeddings(
            self, data_list: list[dict], prefix: str, pad_target_key: str, seq_len_key: str
    ) -> dict[str, torch.Tensor]:
        """Pad pre-computed ESM3 embeddings and build sequence_mask from known lengths."""
        padding_target_len = max(data[pad_target_key] for data in data_list)

        renamed_list = [{"embedding": data[f"{prefix}_embedding"]} for data in data_list]
        embedding = self._pad_batch(renamed_list, "embedding", 0.0, padding_target_len)

        bsz = len(data_list)
        seq_mask = torch.zeros(bsz, padding_target_len, dtype=torch.long)
        for i, data in enumerate(data_list):
            seq_mask[i, :data[seq_len_key]] = 1

        return {"embedding": embedding, "sequence_mask": seq_mask}

    def __call__(self, data_list: list[dict]):
        batch_size = len(data_list)

        # 1. pad protein A and protein B to deterministic targets
        use_embeddings = "protein_a_embedding" in data_list[0]
        if use_embeddings:
            protein_a = self._pad_protein_embeddings(data_list, "protein_a", "pad_target_a", "seq_len_a")
            protein_b = self._pad_protein_embeddings(data_list, "protein_b", "pad_target_b", "seq_len_b")
        else:
            protein_a = self._pad_protein(data_list, "protein_a", "pad_target_a")
            protein_b = self._pad_protein(data_list, "protein_b", "pad_target_b")

        # 2. construct prompts with per-sample roi token counts
        prompts = []
        for i, data in enumerate(data_list):
            n_a = data["compressed_len_a"]
            n_b = data["compressed_len_b"]
            user_message = (
                "Describe the interaction between protein A "
                + self.roi_token * n_a
                + " and protein B "
                + self.roi_token * n_b
                + " with pair map "
                + self.boi_token
                + self.roi_token * self.pair_map_n_tokens
                + self.eoi_token
            )
            prompt = self.text_tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=False,
                add_special_tokens=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompts.append(prompt)

        # 3. tokenize and pad text
        self.text_tokenizer.padding_side = "left"
        tokenized_prompts = self.text_tokenizer(
            prompts,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        self.text_tokenizer.padding_side = "right"
        tokenized_answers = self.text_tokenizer(
            [data["answer_text"] for data in data_list],
            add_special_tokens=False,
            padding="longest",
            max_length=self.max_answer_length if self.mode == "train_val" else None,
            truncation=True,
            return_tensors="pt",
        )

        answer_labels = tokenized_answers["input_ids"].clone()
        answer_labels[tokenized_answers["attention_mask"] == 0] = -100

        # 4. package outputs
        seq_len_a = torch.tensor([data["seq_len_a"] for data in data_list], dtype=torch.long)
        seq_len_b = torch.tensor([data["seq_len_b"] for data in data_list], dtype=torch.long)
        compressed_len_a = torch.tensor(
            [data["compressed_len_a"] for data in data_list], dtype=torch.long
        )
        compressed_len_b = torch.tensor(
            [data["compressed_len_b"] for data in data_list], dtype=torch.long
        )

        if self.mode == "train_val":
            return {
                "input_ids": torch.cat([tokenized_prompts["input_ids"], tokenized_answers["input_ids"]], dim=1),
                "attention_mask": torch.cat(
                    [tokenized_prompts["attention_mask"], tokenized_answers["attention_mask"]], dim=1
                ),
                "labels": torch.cat([torch.full_like(tokenized_prompts["input_ids"], -100), answer_labels], dim=1),
                "protein_a": protein_a,
                "protein_b": protein_b,
                "seq_len_a": seq_len_a,
                "seq_len_b": seq_len_b,
                "compressed_len_a": compressed_len_a,
                "compressed_len_b": compressed_len_b,
            }
        elif self.mode == "inference":
            return {
                "pair_ids": [data["pair_id"] for data in data_list],
                "input_ids": tokenized_prompts["input_ids"],
                "attention_mask": tokenized_prompts["attention_mask"],
                "labels": tokenized_answers["input_ids"],
                "protein_a": protein_a,
                "protein_b": protein_b,
                "seq_len_a": seq_len_a,
                "seq_len_b": seq_len_b,
                "compressed_len_a": compressed_len_a,
                "compressed_len_b": compressed_len_b,
            }
        else:
            raise ValueError(f"Invalid mode {self.mode}")
