"""
Pre-encode proteins with OpenESM3 tokenizers and the structure encoder, one .pt file per accession.

Designed for multi-GPU setups with multiple worker processes per GPU to maximize throughput.
"""

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import random
from typing import List

from esm.utils import encoding
import numpy as np
import torch
from tqdm import tqdm

from model.api_openesm3 import OpenESMProtein
from model.pretrained_openesm3 import OpenESM3_structure_encoder_v0
from model.tokenization_openesm3 import get_openesm3_model_tokenizers


def process_single(
        acc: str,
        structure_encoder,
        tokenizers,
        sequence: str | None,
        ss8: str | None,
        sasa: List[float] | None,
        pdb_dir: str,
        output_dir: str,
    ) -> None:
    """Encode one protein and save its tensor dict; no-op if the output already exists."""
    output_path = os.path.join(output_dir, f"esm3enc_{acc}.pt")
    if os.path.isfile(output_path):
        return  # skip already-processed accessions for cheap re-runs

    # prepare OpenESMProtein object
    pdb_path_gz = f"{pdb_dir}/AF-{acc}-F1-model_v6.pdb.gz"
    pdb_path = f"{pdb_dir}/AF-{acc}-F1-model_v6.pdb"
    if os.path.isfile(pdb_path_gz):
        open_fn, resolved_path = gzip.open, pdb_path_gz
    elif os.path.isfile(pdb_path):
        open_fn, resolved_path = open, pdb_path
    else:
        open_fn, resolved_path = None, None

    if resolved_path is not None:
        with open_fn(resolved_path, 'rt') as f:
            protein = OpenESMProtein.from_pdb(f, is_predicted=True)
        protein.secondary_structure = ss8
        protein.sasa = sasa
    else:
        protein = OpenESMProtein(sequence=sequence)

    # mimic ESM3.encode() for tokenization
    sequence_tokens = None
    secondary_structure_tokens = None
    sasa_tokens = None
    structure_tokens = None
    coordinates = None
    per_res_plddt = None
    average_plddt = None

    if protein.sequence is not None:
        sequence_tokens = encoding.tokenize_sequence(
            protein.sequence, tokenizers.sequence, add_special_tokens=True
        )

    if protein.secondary_structure is not None:
        secondary_structure_tokens = encoding.tokenize_secondary_structure(
            protein.secondary_structure,
            tokenizers.secondary_structure,
            add_special_tokens=True,
        )

    if protein.sasa is not None:
        sasa_tokens = encoding.tokenize_sasa(
            protein.sasa, tokenizers.sasa, add_special_tokens=True
        )

    if protein.coordinates is not None:
        with torch.no_grad():
            coords, _, s_tokens = encoding.tokenize_structure(
                protein.coordinates,
                structure_encoder,
                structure_tokenizer=tokenizers.structure,
                reference_sequence=protein.sequence or "",
                add_special_tokens=True,
            )
        coordinates = coords[..., :3, :].to("cpu")  # keep only the first three atoms
        structure_tokens = s_tokens.to("cpu")

    if protein.plddt is not None:
        plddt = np.pad(protein.plddt, (1, 1), constant_values=0.0)
        per_res_plddt = torch.tensor(plddt, dtype=torch.float32, device="cpu") / 100.0
        average_plddt = per_res_plddt.mean()

    data_dict = {
        "sequence_tokens": sequence_tokens,
        "structure_tokens": structure_tokens,
        "ss8_tokens": secondary_structure_tokens,
        "sasa_tokens": sasa_tokens,
        "function_tokens": None,
        "residue_annotation_tokens": None,
        "average_plddt": average_plddt,
        "per_res_plddt": per_res_plddt,
        "structure_coords": coordinates,
        "chain_id": None,
        "sequence_id": None
    }
    torch.save(data_dict, output_path)


def process_chunk(
        worker_id: int,
        gpu_id: int,
        accessions_chunk: list,
        esm3_dir: str,
        pdb_dir: str,
        sequences_jsonl_path: str,
        sasa_npz_path: str,
        ss8_npz_path: str,
        output_dir: str,
    ):
    """Process a chunk of accessions on a specific GPU."""
    # initialize structure encoder and tokenizers on the designated GPU
    structure_encoder = OpenESM3_structure_encoder_v0(local_dir=esm3_dir, device=torch.device(f"cuda:{gpu_id}"))
    tokenizers = get_openesm3_model_tokenizers(local_dir=esm3_dir)

    print(f"Loading sequences on worker {worker_id}, GPU {gpu_id}...")
    seq_dict = {}
    with open(sequences_jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            seq_dict[entry["accession"]] = entry["sequence"]

    print(f"Loading ss8 data on worker {worker_id}, GPU {gpu_id}...")
    ss8_dict = np.load(ss8_npz_path, allow_pickle=True)

    print(f"Loading sasa data on worker {worker_id}, GPU {gpu_id}...")
    sasa_dict = np.load(sasa_npz_path, allow_pickle=True)

    for acc in tqdm(accessions_chunk, desc=f"Worker {worker_id}, GPU {gpu_id}", position=worker_id):
        process_single(
            acc=acc,
            structure_encoder=structure_encoder,
            tokenizers=tokenizers,
            sequence=seq_dict[acc] if acc in seq_dict else None,
            ss8=str(ss8_dict[acc]) if acc in ss8_dict else None,
            sasa=sasa_dict[acc].tolist() if acc in sasa_dict else None,
            pdb_dir=pdb_dir,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm3_dir", type=str, required=True, help="Local dir of esm3-sm-open-v1 weights.")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Dir with AlphaFold PDB files.")
    parser.add_argument("--sequences_jsonl_path", type=str, required=True, help="JSONL of {accession, sequence}.")
    parser.add_argument("--sasa_npz_path", type=str, required=True, help="SASA NPZ from 21_calculate_sasa.py.")
    parser.add_argument("--ss8_npz_path", type=str, required=True, help="SS8 NPZ from 22_calculate_ss8.py.")
    parser.add_argument("--acc_list_path", type=str, required=True, help="Text file, one UniProt accession per line.")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir for per-protein .pt encoded tensors.")
    parser.add_argument("--workers_per_gpu", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: all visible).")
    args = parser.parse_args()

    mp.set_start_method("spawn")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.acc_list_path, 'r') as f:
        accessions = [line.strip().split("-")[0] for line in tqdm(f, desc="Loading accessions")]
    random.shuffle(accessions)  # shuffle for balanced workload across workers

    num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected, exiting...")

    total_workers = num_gpus * args.workers_per_gpu
    print(f"Using {num_gpus} GPUs, spawning {total_workers} workers...")
    chunk_size = math.ceil(len(accessions) / total_workers)
    chunks = [accessions[i : i + chunk_size] for i in range(0, len(accessions), chunk_size)]

    # oversubscribe each GPU with multiple processes to maximize throughput
    processes = []
    for worker_id, chunk in enumerate(chunks):
        gpu_id = worker_id % num_gpus
        p = mp.Process(target=process_chunk, args=(
            worker_id, gpu_id, chunk,
            args.esm3_dir, args.pdb_dir, args.sequences_jsonl_path,
            args.sasa_npz_path, args.ss8_npz_path, args.output_dir,
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
