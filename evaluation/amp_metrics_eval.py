#!/usr/bin/env python3
"""Compute paper-facing AMP physicochemical and design metrics."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.SeqUtils.ProtParam import ProteinAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)
ALIGNER = PairwiseAligner()
ALIGNER.mode = "global"
ALIGNER.match_score = 1.0
ALIGNER.mismatch_score = 0.0
ALIGNER.open_gap_score = 0.0
ALIGNER.extend_gap_score = 0.0
KYTE_DOOLITTLE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
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
ACTIVITY_SCORE_COLUMNS = (
    "activity_mean_score",
    "pepnet_score",
    "amppred_mfa_score",
    "iamp_attenpred_score",
    "unidl4biopep_score",
)


def clean_sequence(value: object) -> str:
    return "".join(ch for ch in str(value).strip().upper() if ch.isalpha())


def standard_sequence(value: object) -> str:
    return "".join(ch for ch in clean_sequence(value) if ch in AA_SET)


def read_table_or_fasta(path: Path, seq_col: Optional[str], id_col: Optional[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa"}:
        return pd.DataFrame(
            [
                {"sequence_id": record.id or f"seq_{idx}", "sequence": clean_sequence(record.seq)}
                for idx, record in enumerate(SeqIO.parse(str(path), "fasta"), start=1)
            ]
        )

    sep = "\t" if suffix == ".tsv" else ","
    source = pd.read_csv(path, sep=sep)
    selected_seq_col = seq_col or next((col for col in DEFAULT_SEQUENCE_COLUMNS if col in source.columns), None)
    if selected_seq_col is None:
        raise ValueError(
            f"No sequence column found in {path}. Pass --seq-col; tried {DEFAULT_SEQUENCE_COLUMNS}."
        )
    selected_id_col = id_col or next((col for col in DEFAULT_ID_COLUMNS if col in source.columns), None)
    out = source.copy()
    out["sequence_id"] = (
        source[selected_id_col].astype(str)
        if selected_id_col
        else [f"seq_{idx + 1}" for idx in range(len(source))]
    )
    out["sequence"] = source[selected_seq_col].map(clean_sequence)
    leading = ["sequence_id", "sequence"]
    out = out[leading + [col for col in out.columns if col not in leading]]
    return out


def hydrophobic_moment(seq: str, angle_degrees: float = 100.0) -> float:
    std = standard_sequence(seq)
    if not std:
        return float("nan")
    theta = math.radians(angle_degrees)
    x_sum = 0.0
    y_sum = 0.0
    for idx, aa in enumerate(std):
        h = KYTE_DOOLITTLE[aa]
        x_sum += h * math.cos(idx * theta)
        y_sum += h * math.sin(idx * theta)
    return math.sqrt(x_sum * x_sum + y_sum * y_sum) / len(std)


def physicochemical_metrics(sequence_id: str, sequence: str) -> Dict[str, object]:
    clean = clean_sequence(sequence)
    std = standard_sequence(clean)
    invalid = "".join(sorted(set(clean) - AA_SET))
    if not std:
        return {
            "sequence_id": sequence_id,
            "sequence": clean,
            "length": len(clean),
            "valid_standard_aa": False,
            "invalid_residues": invalid,
            "net_charge_pH7_4": np.nan,
            "gravy_kyte_doolittle": np.nan,
            "hydrophobic_moment_100deg": np.nan,
            "helicity": np.nan,
            "aggregation_na4vss": np.nan,
        }
    analysis = ProteinAnalysis(std)
    return {
        "sequence_id": sequence_id,
        "sequence": clean,
        "length": len(clean),
        "valid_standard_aa": len(std) == len(clean),
        "invalid_residues": invalid,
        "net_charge_pH7_4": analysis.charge_at_pH(7.4),
        "gravy_kyte_doolittle": analysis.gravy(),
        "hydrophobic_moment_100deg": hydrophobic_moment(std),
        "helicity": np.nan,
        "aggregation_na4vss": np.nan,
    }


def read_training_sequences(path: Optional[str], seq_col: Optional[str]) -> List[str]:
    if not path:
        return []
    df = read_table_or_fasta(Path(path), seq_col, None)
    return [standard_sequence(seq) for seq in df["sequence"] if standard_sequence(seq)]


def sequence_identity(seq_a: str, seq_b: str) -> float:
    if not seq_a or not seq_b:
        return 0.0
    return float(ALIGNER.score(seq_a, seq_b)) / max(len(seq_a), len(seq_b), 1)


def novelty_scores(sequences: Sequence[str], train_sequences: Sequence[str]) -> List[float]:
    if not train_sequences:
        return [np.nan] * len(sequences)
    scores = []
    for seq in sequences:
        std = standard_sequence(seq)
        max_identity = max(sequence_identity(std, train) for train in train_sequences) if std else 0.0
        scores.append(1.0 - max_identity)
    return scores


def add_external_columns(metrics: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    for source_col, dest_col in (
        ("helicity", "helicity"),
        ("aggregation_na4vss", "aggregation_na4vss"),
        ("Na4vSS", "aggregation_na4vss"),
        ("na4vss", "aggregation_na4vss"),
    ):
        if source_col in source.columns:
            metrics[dest_col] = pd.to_numeric(source[source_col], errors="coerce")
    for col in list(ACTIVITY_SCORE_COLUMNS) + ["hemolysis_score"]:
        if col in source.columns and col not in metrics.columns:
            metrics[col] = pd.to_numeric(source[col], errors="coerce")
    return metrics


def add_composite_metrics(
    metrics: pd.DataFrame,
    activity_threshold: float,
    hemolysis_threshold: float,
    ref: Optional[pd.DataFrame],
) -> pd.DataFrame:
    charge = metrics["net_charge_pH7_4"]
    gravy = metrics["gravy_kyte_doolittle"]
    charge_std = charge.std(ddof=0)
    gravy_std = gravy.std(ddof=0)
    if charge_std and gravy_std and not np.isnan(charge_std) and not np.isnan(gravy_std):
        z_charge = (charge - charge.mean()) / charge_std
        z_gravy = (gravy - gravy.mean()) / gravy_std
        metrics["chbi"] = 1.0 / (1.0 + np.exp(-(z_charge - z_gravy)))
    else:
        metrics["chbi"] = np.nan

    activity_col = next((col for col in ACTIVITY_SCORE_COLUMNS if col in metrics.columns), None)
    if activity_col and "hemolysis_score" in metrics.columns:
        metrics["dual_hit"] = (
            (metrics[activity_col] >= activity_threshold)
            & (metrics["hemolysis_score"] < hemolysis_threshold)
        )
    else:
        metrics["dual_hit"] = np.nan

    if ref is None or ref.empty:
        phys_cols = [
            "gravy_kyte_doolittle",
            "net_charge_pH7_4",
            "hydrophobic_moment_100deg",
            "helicity",
            "aggregation_na4vss",
        ]
        ref = metrics[phys_cols]
    desirability_cols = [
        "gravy_kyte_doolittle",
        "net_charge_pH7_4",
        "hydrophobic_moment_100deg",
        "helicity",
        "aggregation_na4vss",
    ]
    components = []
    for col in desirability_cols:
        if col not in metrics.columns or col not in ref.columns:
            continue
        mu = pd.to_numeric(ref[col], errors="coerce").mean()
        sigma = pd.to_numeric(ref[col], errors="coerce").std(ddof=0)
        if not sigma or np.isnan(mu) or np.isnan(sigma):
            continue
        components.append(np.exp(-0.5 * ((metrics[col] - mu) / sigma) ** 2))
    if components:
        phys = np.prod(components, axis=0) ** (1.0 / len(components))
        metrics["physicochemical_desirability"] = phys
    else:
        metrics["physicochemical_desirability"] = np.nan

    required = [activity_col, "hemolysis_score", "novelty", "physicochemical_desirability"]
    if activity_col and all(col in metrics.columns for col in required):
        product = (
            metrics[activity_col]
            * (1.0 - metrics["hemolysis_score"])
            * metrics["novelty"]
            * metrics["physicochemical_desirability"]
        )
        metrics["overall_desirability"] = product.clip(lower=0) ** 0.25
    else:
        metrics["overall_desirability"] = np.nan
    return metrics


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in metrics.columns:
        if col in {"sequence_id", "sequence", "invalid_residues"}:
            continue
        if metrics[col].dtype == bool:
            values = metrics[col].astype(float)
        else:
            values = pd.to_numeric(metrics[col], errors="coerce")
        values = values.dropna()
        if values.empty:
            rows.append({"metric": col, "n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "min": np.nan, "max": np.nan})
            continue
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
    return pd.DataFrame(rows)


def write_notes(path: Path, metrics: pd.DataFrame) -> None:
    missing = [
        col
        for col in ["helicity", "aggregation_na4vss", "novelty", "hemolysis_score"]
        if col in metrics.columns and metrics[col].isna().all()
    ]
    lines = [
        "# AmpDiff AMP Metrics Notes",
        "",
        "- Net charge is calculated at pH 7.4 with Biopython ProteinAnalysis.",
        "- Hydrophobicity is Kyte-Doolittle GRAVY.",
        "- Amphipathicity is the alpha-helical hydrophobic moment with 100-degree residue spacing.",
        "- CHBI uses the evaluated set as the default standardization reference.",
        "- Dual hit uses activity >= 0.80 and hemolysis < 0.40 unless overridden by CLI flags.",
    ]
    if missing:
        lines.append(f"- Not computed because required input/reference columns were unavailable: {', '.join(missing)}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute AmpDiff paper AMP metrics.")
    parser.add_argument("--input", required=True, help="Input FASTA, CSV, or TSV.")
    parser.add_argument("--seq-col", default=None)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--train", default=None, help="Training FASTA/CSV/TSV for novelty calculation.")
    parser.add_argument("--train-seq-col", default=None)
    parser.add_argument("--reference", default=None, help="Optional CSV/TSV reference distribution for desirability.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--activity-threshold", type=float, default=0.80)
    parser.add_argument("--hemolysis-threshold", type=float, default=0.40)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else input_path.resolve().parent / "amp_metrics_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    source = read_table_or_fasta(input_path, args.seq_col, args.id_col)
    rows = [
        physicochemical_metrics(row.sequence_id, row.sequence)
        for row in source.itertuples(index=False)
    ]
    metrics = pd.DataFrame(rows)
    metrics = add_external_columns(metrics, source)
    train_sequences = read_training_sequences(args.train, args.train_seq_col)
    metrics["novelty"] = novelty_scores(metrics["sequence"].tolist(), train_sequences)

    ref = None
    if args.reference:
        ref_path = Path(args.reference)
        ref_sep = "\t" if ref_path.suffix.lower() == ".tsv" else ","
        ref = pd.read_csv(ref_path, sep=ref_sep)
    metrics = add_composite_metrics(metrics, args.activity_threshold, args.hemolysis_threshold, ref)

    metrics.to_csv(out_dir / "amp_metrics.csv", index=False)
    summarize(metrics).to_csv(out_dir / "amp_metrics_summary.csv", index=False)
    write_notes(out_dir / "amp_metrics_notes.md", metrics)
    print(f"Metrics: {out_dir / 'amp_metrics.csv'}")
    print(f"Summary: {out_dir / 'amp_metrics_summary.csv'}")
    print(f"Notes: {out_dir / 'amp_metrics_notes.md'}")


if __name__ == "__main__":
    main()
