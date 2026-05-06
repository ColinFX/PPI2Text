"""
Compute per-residue SASA for AlphaFold structures.

Reads a list of UniProt accessions, fetches their AlphaFold PDB files (cached locally, with bounded download
concurrency), runs ShrakeRupley, and writes one SASA array per accession into a single NPZ file.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import io
import multiprocessing
import os
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
import numpy as np
from tqdm import tqdm


ALPHAFOLD_URL_TEMPLATE = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.pdb"

# populated from argparse in __main__; subprocesses inherit via fork
PDB_CACHE_DIR: str = ""
_download_semaphore: Optional[multiprocessing.Semaphore] = None


def _init_worker(semaphore: multiprocessing.Semaphore, pdb_cache_dir: str):
    global _download_semaphore, PDB_CACHE_DIR
    _download_semaphore = semaphore
    PDB_CACHE_DIR = pdb_cache_dir


def _find_cache_path(accession: str) -> Optional[str]:
    """Return the local cache file path for a given accession if it exists (.pdb.gz or .pdb)."""
    canonical = accession.split("-")[0] if "-" in accession else accession
    for ext in (".pdb.gz", ".pdb"):
        path = os.path.join(PDB_CACHE_DIR, f"AF-{canonical}-F1-model_v6{ext}")
        if os.path.isfile(path):
            return path
    return None


def _read_cache_file(path: str) -> str:
    """Read PDB content from a cache file, handling gzip or plain text."""
    if path.endswith(".gz"):
        with gzip.open(path, 'rt') as f:
            return f.read()
    with open(path, 'r') as f:
        return f.read()


def _download_pdb(accession: str) -> Optional[str]:
    """Download PDB content from AlphaFold DB and save to cache. Returns PDB text or None."""
    canonical = accession.split("-")[0] if "-" in accession else accession
    url = ALPHAFOLD_URL_TEMPLATE.format(accession=canonical)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            pdb_content = resp.read().decode("utf-8")
        save_path = os.path.join(PDB_CACHE_DIR, f"AF-{canonical}-F1-model_v6.pdb")
        with open(save_path, 'w') as f:
            f.write(pdb_content)
        return pdb_content
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def _load_or_download_pdb(accession: str) -> Optional[str]:
    """Load PDB from local cache if available, otherwise download (respecting concurrency limit)."""
    cache_path = _find_cache_path(accession)
    if cache_path is not None:
        return _read_cache_file(cache_path)

    _download_semaphore.acquire()
    try:
        pdb_content = _download_pdb(accession)
    finally:
        _download_semaphore.release()
    return pdb_content


def _calculate_single_sasa(accession: str) -> Tuple[str, Optional[np.ndarray]]:
    """Load (or download) one PDB and return (accession, sasa_array) or (accession, None) on failure."""
    pdb_content = _load_or_download_pdb(accession)

    if pdb_content is None:
        return accession, None

    # compute phase runs at full worker concurrency, the semaphore only gates downloads
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("AF_Model", io.StringIO(pdb_content))

        model = structure[0]
        sr = ShrakeRupley()
        sr.compute(model, level="R")

        sasa_values = [
            residue.sasa for chain in model
            for residue in chain if residue.id[0] == ' '
        ]
        return accession, np.array(sasa_values, dtype=np.float16)
    except Exception:
        return accession, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acc_list_path", type=str, required=True, help="Text file, one UniProt accession per line.")
    parser.add_argument("--output_path", type=str, required=True, help="Output .npz, one key per accession.")
    parser.add_argument("--pdb_cache_dir", type=str, required=True, help="Local cache for AlphaFold PDB files.")
    parser.add_argument("--num_workers", type=int, default=28)
    parser.add_argument("--max_download_concurrent", type=int, default=5)
    args = parser.parse_args()

    PDB_CACHE_DIR = args.pdb_cache_dir
    os.makedirs(PDB_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.acc_list_path, 'r') as f:
        accessions = [line.strip().split()[0] for line in f]

    sasa_dict: Dict[str, np.ndarray] = {}
    found_count = 0
    missing_count = 0

    semaphore = multiprocessing.Semaphore(args.max_download_concurrent)

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=_init_worker,
        initargs=(semaphore, PDB_CACHE_DIR),
    ) as executor:
        futures = {executor.submit(_calculate_single_sasa, acc): acc for acc in accessions}

        with tqdm(total=len(accessions), desc="Calculating SASA", unit="entry") as pbar:
            for future in as_completed(futures):
                acc, result = future.result()

                if result is not None:
                    sasa_dict[acc] = result
                    found_count += 1
                else:
                    missing_count += 1

                pbar.update(1)
                pbar.set_postfix({"Found": found_count, "Missing": missing_count})

    np.savez(args.output_path, **sasa_dict)
    print(f"Saved: {args.output_path}")
