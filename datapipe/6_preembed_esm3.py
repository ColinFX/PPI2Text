"""
Pre-compute ESM3 transformer outputs (x_postnorm) for proteins with len <= max_seq_len.

Loads the pre-encoded token files from step 23, runs the 48-layer ESM3, and writes one .pt per protein at
output_dir/esm3emb_{accession}.pt with shape (L, 1536) in bfloat16. The saved tensor matches what _encode_protein
in PairEsm3Qwen3ForCausalLM produces at training time, so training can skip loading ESM3 entirely.

Processed one protein at a time (batch_size=1) to avoid O(B*N^2) memory from geometric attention; one model per GPU.
"""

import argparse
import math
import multiprocessing as mp
import os
import random
import traceback

from esm.utils.constants import esm3 as C
import torch
from tqdm import tqdm

from model.modeling_openesm3 import OpenESM3
from model.tokenization_openesm3 import get_openesm3_model_tokenizers


def process_single(
        acc: str,
        model: OpenESM3,
        protein_tokenizers,
        device: torch.device,
        input_pt_dir: str,
        output_dir: str,
        max_seq_len: int,
    ) -> None:
    """Load pre-encoded tokens for one protein, run ESM3 forward pass, and save the embedding."""
    output_path = os.path.join(output_dir, f"esm3emb_{acc}.pt")
    if os.path.isfile(output_path):
        return

    input_path = os.path.join(input_pt_dir, f"esm3enc_{acc}.pt")
    if not os.path.isfile(input_path):
        return

    encoded = torch.load(input_path, map_location="cpu")

    if encoded["sequence_tokens"] is None:
        return

    L = encoded["sequence_tokens"].shape[0]
    if L - 2 > max_seq_len:  # L includes BOS and EOS tokens
        return

    # fill defaults for missing fields, matching PairSFTDataset._load_protein_tensors
    sequence_tokens = encoded["sequence_tokens"].unsqueeze(0).to(device)

    if encoded["structure_tokens"] is not None:
        structure_tokens = encoded["structure_tokens"].unsqueeze(0).to(device)
    else:
        structure_tokens = torch.full(
            (1, L), protein_tokenizers.structure.mask_token_id, dtype=torch.long, device=device
        )

    if encoded["ss8_tokens"] is not None:
        ss8_tokens = encoded["ss8_tokens"].unsqueeze(0).to(device)
    else:
        ss8_tokens = torch.full((1, L), C.SS8_PAD_TOKEN, dtype=torch.long, device=device)

    if encoded["sasa_tokens"] is not None:
        sasa_tokens = encoded["sasa_tokens"].unsqueeze(0).to(device)
    else:
        sasa_tokens = torch.full((1, L), C.SASA_PAD_TOKEN, dtype=torch.long, device=device)

    # function_tokens and residue_annotation_tokens are always pad — they're never in pre-encoded files
    function_tokens = torch.full((1, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device)
    residue_annotation_tokens = torch.full((1, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device)

    if encoded["average_plddt"] is not None:
        average_plddt = torch.full(
            (1, L), encoded["average_plddt"].item(), dtype=torch.float32, device=device
        )
    else:
        average_plddt = torch.full((1, L), 1.0, dtype=torch.float32, device=device)

    if encoded["per_res_plddt"] is not None:
        per_res_plddt = encoded["per_res_plddt"].unsqueeze(0).to(device).float()
    else:
        per_res_plddt = torch.zeros((1, L), dtype=torch.float32, device=device)

    if encoded["structure_coords"] is not None:
        structure_coords = encoded["structure_coords"].unsqueeze(0).to(device).float()
    else:
        structure_coords = torch.full((1, L, 3, 3), float("nan"), dtype=torch.float32, device=device)

    # single chain, no padding → all 1s for sequence_id and chain_id
    # (matches collator: sequence_id = row_index * pad_mask, with row_index=1 and no padding)
    sequence_id = torch.ones((1, L), dtype=torch.long, device=device)
    chain_id = torch.ones((1, L), dtype=torch.long, device=device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        x_postnorm, _ = model.batch_forward(
            sequence_tokens=sequence_tokens,
            structure_tokens=structure_tokens,
            ss8_tokens=ss8_tokens,
            sasa_tokens=sasa_tokens,
            function_tokens=function_tokens,
            residue_annotation_tokens=residue_annotation_tokens,
            average_plddt=average_plddt,
            per_res_plddt=per_res_plddt,
            structure_coords=structure_coords,
            chain_id=chain_id,
            sequence_id=sequence_id,
        )

    # save on CPU in bfloat16 to halve disk usage (~158GB vs ~315GB for all proteins)
    torch.save(x_postnorm.squeeze(0).to(torch.bfloat16).cpu(), output_path)


def process_chunk(
        worker_id: int,
        gpu_id: int,
        accessions_chunk: list,
        esm3_dir: str,
        input_pt_dir: str,
        output_dir: str,
        max_seq_len: int,
    ):
    """Load ESM3 on one GPU and process a chunk of accessions."""
    device = torch.device(f"cuda:{gpu_id}")

    print(f"Loading ESM3 model on worker {worker_id}, GPU {gpu_id}...")
    model = OpenESM3.from_pretrained(local_dir=esm3_dir, device=device)
    model.eval()

    print(f"Loading tokenizers on worker {worker_id}, GPU {gpu_id}...")
    protein_tokenizers = get_openesm3_model_tokenizers(local_dir=esm3_dir)

    for acc in tqdm(accessions_chunk, desc=f"Worker {worker_id}, GPU {gpu_id}", position=worker_id):
        try:
            process_single(acc, model, protein_tokenizers, device, input_pt_dir, output_dir, max_seq_len)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM for {acc} (worker {worker_id}, GPU {gpu_id}), skipping...", flush=True)
            torch.cuda.empty_cache()
        except Exception:
            print(f"Error processing {acc} (worker {worker_id}, GPU {gpu_id}):", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm3_dir", type=str, required=True, help="Local dir of esm3-sm-open-v1 weights.")
    parser.add_argument("--input_pt_dir", type=str, required=True, help="Per-protein .pt from 23_preencode_esm3.py.")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir for per-protein .pt embedding tensors.")
    parser.add_argument("--acc_list_path", type=str, required=True, help="Text file, one UniProt accession per line.")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length, excluding BOS/EOS.")
    args = parser.parse_args()

    mp.set_start_method("spawn")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading accessions from {args.acc_list_path}")
    with open(args.acc_list_path, 'r') as f:
        accessions = [line.strip().split("-")[0] for line in tqdm(f, desc="Loading accessions")]
    random.shuffle(accessions)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected, exiting...")

    # one worker per GPU — ESM3 is large (~3GB bf16) and geometric attention is memory-intensive
    total_workers = num_gpus
    print(f"Detected {num_gpus} GPUs, spawning {total_workers} workers (1 per GPU)...")
    chunk_size = math.ceil(len(accessions) / total_workers)
    chunks = [accessions[i : i + chunk_size] for i in range(0, len(accessions), chunk_size)]

    processes = []
    for worker_id, chunk in enumerate(chunks):
        gpu_id = worker_id
        p = mp.Process(target=process_chunk, args=(
            worker_id, gpu_id, chunk,
            args.esm3_dir, args.input_pt_dir, args.output_dir, args.max_seq_len,
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
