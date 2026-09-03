#!/usr/bin/env python3
"""Evaluate AMP activity and hemolysis predictors for generated sequences."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.Generalizability.amp_scorers import ACTIVITY_SCORERS, build_scorer_from_args  # noqa: E402


DEFAULT_SEQUENCE_COLUMNS = (
    "variant_sequence",
    "generated_sequence",
    "sequence",
    "peptide",
    "seq",
)
DEFAULT_ID_COLUMNS = (
    "variant_name",
    "sequence_id",
    "name",
    "id",
)


def clean_sequence(value: object) -> str:
    return "".join(ch for ch in str(value).strip().upper() if ch.isalpha())


def read_sequences(path: Path, seq_col: str | None, id_col: str | None) -> Tuple[pd.DataFrame, List[str]]:
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa"}:
        rows = [
            {"sequence_id": record.id or f"seq_{idx}", "sequence": clean_sequence(record.seq)}
            for idx, record in enumerate(SeqIO.parse(str(path), "fasta"), start=1)
        ]
        return pd.DataFrame(rows), []

    sep = "\t" if suffix == ".tsv" else ","
    source = pd.read_csv(path, sep=sep)
    selected_seq_col = seq_col or next((col for col in DEFAULT_SEQUENCE_COLUMNS if col in source.columns), None)
    if selected_seq_col is None:
        raise ValueError(
            f"No sequence column found in {path}. Pass --seq-col; tried {DEFAULT_SEQUENCE_COLUMNS}."
        )
    selected_id_col = id_col or next((col for col in DEFAULT_ID_COLUMNS if col in source.columns), None)
    rows = pd.DataFrame(
        {
            "sequence_id": (
                source[selected_id_col].astype(str)
                if selected_id_col
                else [f"seq_{idx + 1}" for idx in range(len(source))]
            ),
            "sequence": source[selected_seq_col].map(clean_sequence),
        }
    )
    return rows, list(source.columns)


def summarize_scores(df: pd.DataFrame, activity_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for col in list(activity_cols) + ["hemolysis_score", "activity_mean_score"]:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append(
            {
                "metric": col,
                "n": int(values.shape[0]),
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
            }
        )
    if "dual_hit" in df.columns:
        rows.append(
            {
                "metric": "dual_hit_rate",
                "n": int(df["dual_hit"].notna().sum()),
                "mean": df["dual_hit"].mean(),
                "std": float("nan"),
                "median": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AMP activity and hemolysis predictors.")
    parser.add_argument("--input", required=True, help="Input FASTA, CSV, or TSV.")
    parser.add_argument("--seq-col", default=None, help="Sequence column for CSV/TSV input.")
    parser.add_argument("--id-col", default=None, help="Identifier column for CSV/TSV input.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument(
        "--scorers",
        default="pepnet,amppred_mfa,iamp_attenpred,unidl4biopep,hemopi2",
        help="Comma-separated scorers, or none.",
    )
    parser.add_argument("--activity-threshold", type=float, default=0.80)
    parser.add_argument("--hemolysis-threshold", type=float, default=0.40)
    parser.add_argument("--pepnet-root", default="/mnt/wucy/WUCHUYA/PepNet")
    parser.add_argument("--hemopi2-root", default="/mnt/wucy/WUCHUYA/hemopi2")
    parser.add_argument("--amppred-root", default="/mnt/wucy/WUCHUYA/AMPpred-MFA")
    parser.add_argument("--iamp-root", default="/mnt/wucy/WUCHUYA/iAMP-Attenpred")
    parser.add_argument("--unidl-root", default="/mnt/wucy/WUCHUYA/UniDL4BioPep")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else input_path.resolve().parent / "activity_hemolysis_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs, _ = read_sequences(input_path, args.seq_col, args.id_col)
    seqs = seqs[seqs["sequence"].astype(bool)].reset_index(drop=True)
    if seqs.empty:
        raise ValueError("No non-empty sequences were loaded.")

    scorer = build_scorer_from_args(
        args.scorers,
        infer_single_guidance=False,
        pepnet_root=args.pepnet_root,
        hemopi2_root=args.hemopi2_root,
        amppred_root=args.amppred_root,
        iamp_root=args.iamp_root,
        unidl_root=args.unidl_root,
        device=args.device,
    )
    if scorer is None:
        results = seqs.copy()
    else:
        scores = scorer.evaluate(seqs["sequence"].tolist())
        results = pd.concat([seqs, scores.drop(columns=["sequence"], errors="ignore")], axis=1)

    activity_cols = [f"{name}_score" for name in ACTIVITY_SCORERS if f"{name}_score" in results.columns]
    if activity_cols:
        results["activity_mean_score"] = results[activity_cols].mean(axis=1)
    if "activity_mean_score" in results.columns and "hemolysis_score" in results.columns:
        results["dual_hit"] = (
            (results["activity_mean_score"] >= args.activity_threshold)
            & (results["hemolysis_score"] < args.hemolysis_threshold)
        )

    results.to_csv(out_dir / "activity_hemolysis_scores.csv", index=False)
    summarize_scores(results, activity_cols).to_csv(out_dir / "activity_hemolysis_summary.csv", index=False)
    print(f"Scores: {out_dir / 'activity_hemolysis_scores.csv'}")
    print(f"Summary: {out_dir / 'activity_hemolysis_summary.csv'}")


if __name__ == "__main__":
    main()
