"""
Fetch UniProt sequences for a list of accessions and write JSONL of {"accession", "sequence"} records.

Output matches the `sequences_jsonl_path` argument of 23_preencode_esm3.py. Isoform suffixes (e.g. 'P41778-2') are
queried as their canonical parent ('P41778') since AlphaFold and ESM3 outputs live under the canonical accession.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from tqdm import tqdm


UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


def _fetch_one(acc: str, retries: int = 3, backoff: float = 2.0) -> tuple[str, str | None]:
    """Fetch a single accession's sequence; return (accession, sequence) or (accession, None)."""
    canonical = acc.split("-", 1)[0]
    url = UNIPROT_FASTA_URL.format(acc=canonical)
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=30) as resp:
                text = resp.read().decode("utf-8")
            lines = text.splitlines()
            if not lines or not lines[0].startswith(">"):
                return acc, None
            sequence = "".join(lines[1:])
            return acc, sequence
        except (HTTPError, URLError, TimeoutError):
            if attempt + 1 < retries:
                time.sleep(backoff * (2 ** attempt))
            else:
                return acc, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acc_list_path", type=str, required=True, help="Text file, one accession per line.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL of {accession, sequence}.")
    parser.add_argument("--num_workers", type=int, default=16, help="Concurrent HTTP requests; be polite to UniProt.")
    args = parser.parse_args()

    with open(args.acc_list_path, "r") as f:
        accessions = [ln.strip() for ln in f if ln.strip()]

    # skip accessions already present in the output (resume-safe)
    existing: set[str] = set()
    if os.path.isfile(args.output_jsonl):
        with open(args.output_jsonl, "r") as f:
            for ln in f:
                try:
                    existing.add(json.loads(ln)["accession"])
                except (json.JSONDecodeError, KeyError):
                    continue

    todo = [acc for acc in accessions if acc not in existing]
    print(f"{len(existing)} already present; fetching {len(todo)} from UniProt.")

    failures = 0
    with open(args.output_jsonl, "a") as out, \
         ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(_fetch_one, acc): acc for acc in todo}
        with tqdm(total=len(futures), desc="Fetching UniProt sequences", unit="acc") as pbar:
            for fut in as_completed(futures):
                acc, sequence = fut.result()
                if sequence is None:
                    failures += 1
                else:
                    out.write(json.dumps({"accession": acc, "sequence": sequence}) + "\n")
                pbar.update(1)
                pbar.set_postfix({"failures": failures})

    print(f"Done. Output: {args.output_jsonl}. Failures: {failures}/{len(todo)}.")


if __name__ == "__main__":
    main()
