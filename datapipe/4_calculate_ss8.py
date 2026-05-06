"""
Compute per-residue 8-state secondary structure (SS8) for AlphaFold structures via DSSP.

Reads a list of accessions, loads matching PDB files from a local directory, and writes one SS8 string per accession
into an NPZ file. Runs in parallel across CPU cores.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import os
import tempfile
from typing import Dict, Optional, Tuple

from Bio.PDB import PDBParser, DSSP
import numpy as np
from tqdm import tqdm


# populated from argparse in __main__; subprocesses inherit via fork
PDB_DIR: str = ""


def _init_worker(pdb_dir: str):
    global PDB_DIR
    PDB_DIR = pdb_dir


def _find_pdb_file(accession: str) -> Optional[str]:
    """Return the PDB file path for a given accession (.pdb.gz or .pdb), or None if not found."""
    canonical = accession.split("-")[0] if "-" in accession else accession
    for ext in (".pdb.gz", ".pdb"):
        path = os.path.join(PDB_DIR, f"AF-{canonical}-F1-model_v6{ext}")
        if os.path.isfile(path):
            return path
    return None


def _calculate_single_ss8(accession: str) -> Tuple[str, Optional[str]]:
    """Run DSSP on one PDB and return (accession, ss8_string) or (accession, None) on failure."""
    pdb_file = _find_pdb_file(accession)

    if pdb_file is None:
        return accession, None

    try:
        parser = PDBParser(QUIET=True)
        open_fn = gzip.open if pdb_file.endswith('.gz') else open
        with open_fn(pdb_file, 'rt') as f:
            structure = parser.get_structure("AF_Model", f)

        model = structure[0]

        # write a stripped uncompressed copy for DSSP, keeping only CRYST1/ATOM/HETATM/TER —
        # DBREF lines with accessions longer than 6 chars make mkdssp choke
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb') as temp_pdb:
            with open_fn(pdb_file, 'rt') as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM", "TER", "CRYST1")):
                        temp_pdb.write(line)
            temp_pdb.flush()

            try:
                dssp_data = DSSP(model, temp_pdb.name, dssp='mkdssp')
            except Exception as e:
                raise RuntimeError(f"DSSP failed. Ensure it is installed. Error: {e}")

        # map DSSP's '-' (coil) and 'P' (PPII helix) to 'C' to match standard ESM conventions
        ss8_chars = []
        for key in dssp_data.keys():
            char = dssp_data[key][2]
            if char == '-' or char == 'P':
                char = 'C'
            ss8_chars.append(char)
        ss8_string = "".join(ss8_chars)

        return accession, ss8_string
    except Exception:
        return accession, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb_dir", type=str, required=True, help="Dir with AF-<acc>-F1-model_v6.pdb[.gz] files.")
    parser.add_argument("--acc_list_path", type=str, required=True, help="Text file, one UniProt accession per line.")
    parser.add_argument("--output_path", type=str, required=True, help="Output .npz, one key per accession.")
    parser.add_argument("--num_workers", type=int, default=28)
    args = parser.parse_args()

    PDB_DIR = args.pdb_dir
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.acc_list_path, 'r') as f:
        accessions = [line.strip() for line in f]

    ss8_dict: Dict[str, str] = {}
    found_count = 0
    missing_count = 0

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=_init_worker,
        initargs=(PDB_DIR,),
    ) as executor:
        futures = {executor.submit(_calculate_single_ss8, acc): acc for acc in accessions}

        with tqdm(total=len(accessions), desc="Calculating SS8", unit="entry") as pbar:
            for future in as_completed(futures):
                acc, result = future.result()

                if result is not None:
                    ss8_dict[acc] = result
                    found_count += 1
                else:
                    ss8_dict[acc] = None
                    missing_count += 1

                pbar.update(1)
                pbar.set_postfix({"Found": found_count, "Missing": missing_count})

    np.savez(args.output_path, **{k: v for k, v in ss8_dict.items() if v is not None})
    print(f"Saved: {args.output_path}")
