"""
Extract the deduplicated UniProt accession list from ppi2text_dataset.parquet's pair_id column.

The pair_id has the form '<acc_a>_<acc_b>'; output is one accession per line, sorted.
"""

import argparse

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=str, required=True, help="Path to ppi2text_dataset.parquet.")
    parser.add_argument("--output", type=str, required=True, help="Output text file, one accession per line.")
    args = parser.parse_args()

    table = pq.read_table(args.parquet, columns=["pair_id"])
    pair_ids = table.column("pair_id").to_pylist()

    accessions = set()
    for pid in pair_ids:
        a, b = pid.split("_", 1)
        accessions.add(a)
        accessions.add(b)

    sorted_accs = sorted(accessions)
    with open(args.output, "w") as f:
        for acc in sorted_accs:
            f.write(acc + "\n")

    print(f"Wrote {len(sorted_accs)} unique accessions from {len(pair_ids)} pairs to {args.output}")


if __name__ == "__main__":
    main()
