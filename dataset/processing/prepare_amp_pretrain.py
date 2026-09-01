"""
Data optimization script for AMP pretraining data.

Steps:
  1. Load CSV, extract 'sequence' column
  2. Drop null / empty sequences
  3. Filter non-standard AA characters (keep only ACDEFGHIKLMNPQRSTVWYX)
  4. Filter: keep only sequences with 5 <= len <= 100 (matches dataset loader defaults)
  5. Deduplicate: keep first occurrence of each unique sequence
  6. Print stats summary
  7. Save as CSV with single 'sequence' column, no index
"""

import argparse
import pandas as pd

MIN_LEN = 5
MAX_LEN = 100
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")


def main():
    parser = argparse.ArgumentParser(description="Clean and deduplicate AMP pretraining CSV.")
    parser.add_argument("--input",   default="data/pretrain/AMP_51345.csv")
    parser.add_argument("--output",  default="data/pretrain/AMP_51345_clean.csv")
    parser.add_argument("--min_len", type=int, default=MIN_LEN)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    n_raw = len(df)

    # Keep only sequence column, drop nulls and whitespace-only
    seqs = df["sequence"].dropna().str.strip()
    seqs = seqs[seqs != ""]
    n_after_null = len(seqs)

    # Filter non-standard AA characters
    seqs = seqs[seqs.apply(lambda s: set(s).issubset(VALID_AA))]
    n_after_aa = len(seqs)

    # Length filter
    seqs = seqs[seqs.str.len().between(args.min_len, args.max_len)]
    n_after_len = len(seqs)

    # Deduplicate (keep first occurrence)
    seqs = seqs.drop_duplicates()
    n_final = len(seqs)

    # Save
    out_df = seqs.reset_index(drop=True).to_frame(name="sequence")
    out_df.to_csv(args.output, index=False)

    # Stats
    print(f"Raw rows:            {n_raw:>7}")
    print(f"After null drop:     {n_after_null:>7}  (removed {n_raw - n_after_null})")
    print(f"After AA filter:     {n_after_aa:>7}  (removed {n_after_null - n_after_aa} with non-standard chars)")
    print(f"After length filter: {n_after_len:>7}  (removed {n_after_aa - n_after_len} outside [{args.min_len}, {args.max_len}])")
    print(f"After dedup:         {n_final:>7}  (removed {n_after_len - n_final} duplicates)")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
