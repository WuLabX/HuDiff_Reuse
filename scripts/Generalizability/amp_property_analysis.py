#!/usr/bin/env python3
"""
Downstream AMP property, substitution and motif-logo analysis for AmpDiff.

This script is intentionally independent of AmpDiff training and sampling. It
uses generated FASTA/CSV files as input and writes reproducible analysis tables.
"""

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.Generalizability.amp_motifs import load_active_motifs  # noqa: E402


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)
HYDROPHOBIC_AA = set("AVILMFWYC")

KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

EISENBERG = {
    "A": 0.62,
    "C": 0.29,
    "D": -0.90,
    "E": -0.74,
    "F": 1.19,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "K": -1.50,
    "L": 1.06,
    "M": 0.64,
    "N": -0.78,
    "P": 0.12,
    "Q": -0.85,
    "R": -2.53,
    "S": -0.18,
    "T": -0.05,
    "V": 1.08,
    "W": 0.81,
    "Y": 0.26,
}

AA_COLORS = {
    "D": "#d73027",
    "E": "#d73027",
    "K": "#4575b4",
    "R": "#4575b4",
    "H": "#4575b4",
    "S": "#66bd63",
    "T": "#66bd63",
    "N": "#66bd63",
    "Q": "#66bd63",
    "A": "#fdae61",
    "V": "#fdae61",
    "I": "#fdae61",
    "L": "#fdae61",
    "M": "#fdae61",
    "F": "#f46d43",
    "W": "#f46d43",
    "Y": "#f46d43",
    "C": "#fee08b",
    "G": "#999999",
    "P": "#999999",
}


def clean_sequence(seq: str) -> str:
    return "".join(ch for ch in str(seq).upper() if ch.isalpha() or ch == "-").replace("-", "")


def standard_sequence(seq: str) -> str:
    return "".join(ch for ch in clean_sequence(seq) if ch in AA_SET)


def load_sequences(args) -> pd.DataFrame:
    rows = []
    if args.fasta:
        for idx, record in enumerate(SeqIO.parse(args.fasta, "fasta"), start=1):
            rows.append({"sequence_id": record.id or f"seq_{idx}", "sequence": clean_sequence(str(record.seq))})
    elif args.csv:
        df = pd.read_csv(args.csv)
        if args.seq_col not in df.columns:
            raise ValueError(f"Sequence column {args.seq_col!r} not found in {args.csv}")
        id_col = args.id_col if args.id_col in df.columns else None
        for idx, row in df.iterrows():
            rows.append(
                {
                    "sequence_id": str(row[id_col]) if id_col else f"seq_{idx + 1}",
                    "sequence": clean_sequence(row[args.seq_col]),
                }
            )
    else:
        raise ValueError("Provide either --fasta or --csv")
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No sequences were loaded")
    return out


def hydrophobic_moment(seq: str, scale: Dict[str, float], angle_degrees: float = 100.0) -> float:
    clean = standard_sequence(seq)
    if not clean:
        return float("nan")
    theta = math.radians(angle_degrees)
    x_sum = 0.0
    y_sum = 0.0
    for idx, aa in enumerate(clean):
        h = scale[aa]
        x_sum += h * math.cos(idx * theta)
        y_sum += h * math.sin(idx * theta)
    return math.sqrt(x_sum * x_sum + y_sum * y_sum) / len(clean)


def property_row(sequence_id: str, seq: str, moment_scale_name: str) -> Dict[str, object]:
    std = standard_sequence(seq)
    invalid = "".join(sorted(set(clean_sequence(seq)) - AA_SET))
    if not std:
        return {
            "sequence_id": sequence_id,
            "sequence": seq,
            "length": len(clean_sequence(seq)),
            "standard_aa_length": 0,
            "valid_standard_aa": False,
            "invalid_residues": invalid,
            "net_charge_pH7": np.nan,
            "isoelectric_point": np.nan,
            "gravy_kyte_doolittle": np.nan,
            "hydrophobic_residue_fraction": np.nan,
            f"hydrophobic_moment_{moment_scale_name}_100deg": np.nan,
        }
    analysis = ProteinAnalysis(std)
    scale = EISENBERG if moment_scale_name == "eisenberg" else KYTE_DOOLITTLE
    return {
        "sequence_id": sequence_id,
        "sequence": seq,
        "length": len(clean_sequence(seq)),
        "standard_aa_length": len(std),
        "valid_standard_aa": len(std) == len(clean_sequence(seq)),
        "invalid_residues": invalid,
        "net_charge_pH7": analysis.charge_at_pH(7.0),
        "isoelectric_point": analysis.isoelectric_point(),
        "gravy_kyte_doolittle": analysis.gravy(),
        "hydrophobic_residue_fraction": sum(aa in HYDROPHOBIC_AA for aa in std) / len(std),
        f"hydrophobic_moment_{moment_scale_name}_100deg": hydrophobic_moment(std, scale),
    }


def compute_properties(seqs: pd.DataFrame, moment_scale_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        property_row(row.sequence_id, row.sequence, moment_scale_name)
        for row in seqs.itertuples(index=False)
    )


def summarize_properties(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "length",
        "standard_aa_length",
        "net_charge_pH7",
        "isoelectric_point",
        "gravy_kyte_doolittle",
        "hydrophobic_residue_fraction",
    ] + [c for c in metrics.columns if c.startswith("hydrophobic_moment_")]
    rows = []
    for col in numeric:
        vals = pd.to_numeric(metrics[col], errors="coerce").dropna()
        rows.append(
            {
                "metric": col,
                "n_valid": int(vals.shape[0]),
                "mean": vals.mean(),
                "std": vals.std(ddof=1),
                "median": vals.median(),
                "min": vals.min(),
                "max": vals.max(),
            }
        )
    rows.append(
        {
            "metric": "valid_standard_aa_fraction",
            "n_valid": int(metrics.shape[0]),
            "mean": metrics["valid_standard_aa"].mean(),
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    )
    return pd.DataFrame(rows)


def load_pairs(path: str, input_col: str, generated_col: str, id_col: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [input_col, generated_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing paired CSV column(s): {missing}")
    pair_id_col = id_col if id_col and id_col in df.columns else None
    return pd.DataFrame(
        {
            "pair_id": df[pair_id_col].astype(str) if pair_id_col else [f"pair_{i + 1}" for i in range(len(df))],
            "input_sequence": df[input_col].map(clean_sequence),
            "generated_sequence": df[generated_col].map(clean_sequence),
        }
    )


def compute_substitutions(pairs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    count = pd.DataFrame(0, index=AA_ORDER, columns=AA_ORDER, dtype=int)
    events = []
    unequal = 0
    invalid_pairs = 0
    compared_positions = 0
    unchanged_positions = 0
    substitution_events = 0

    for pair in pairs.itertuples(index=False):
        a = clean_sequence(pair.input_sequence)
        b = clean_sequence(pair.generated_sequence)
        if not a or not b:
            invalid_pairs += 1
            continue
        if len(a) != len(b):
            unequal += 1
        limit = min(len(a), len(b))
        for pos in range(limit):
            src, dst = a[pos], b[pos]
            if src not in AA_SET or dst not in AA_SET:
                continue
            compared_positions += 1
            if src == dst:
                unchanged_positions += 1
                continue
            count.loc[src, dst] += 1
            substitution_events += 1
            events.append(
                {
                    "pair_id": pair.pair_id,
                    "position_1based": pos + 1,
                    "input_residue": src,
                    "generated_residue": dst,
                    "substitution": f"{src}->{dst}",
                }
            )
    freq = count.astype(float)
    if substitution_events > 0:
        freq = freq / substitution_events
    stats = {
        "n_pairs": len(pairs),
        "n_empty_or_invalid_pairs": invalid_pairs,
        "n_length_mismatched_pairs": unequal,
        "n_compared_standard_positions": compared_positions,
        "n_unchanged_positions": unchanged_positions,
        "n_substitution_events": substitution_events,
    }
    return count, freq, pd.DataFrame(events), stats


def motif_spans(sequence: str, exact_motifs: Sequence[str], prosite_patterns: Sequence[object]) -> List[Tuple[int, int, str]]:
    spans = []
    for motif in exact_motifs:
        start = 0
        while motif:
            idx = sequence.find(motif, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(motif), motif))
            start = idx + 1
    for pattern in prosite_patterns:
        for match in pattern.finditer(sequence):
            spans.append((match.start(), match.end(), pattern.pattern))
    return sorted(spans)


def extract_logo_windows(
    seqs: pd.DataFrame,
    active_motif_file: Optional[str],
    motif_types: str,
    window: int,
) -> Tuple[List[str], pd.DataFrame]:
    if not active_motif_file:
        return [], pd.DataFrame()
    motifs = load_active_motifs(active_motif_file, enabled_types=None if motif_types == "all" else set(motif_types.split(",")))
    half = window // 2
    windows = []
    rows = []
    for row in seqs.itertuples(index=False):
        seq = clean_sequence(row.sequence)
        for start, end, motif_name in motif_spans(seq, motifs["exact"], motifs["prosite"]):
            center = (start + end - 1) // 2
            left = center - half
            chars = []
            for pos in range(left, left + window):
                chars.append(seq[pos] if 0 <= pos < len(seq) else "-")
            win = "".join(chars)
            windows.append(win)
            rows.append(
                {
                    "sequence_id": row.sequence_id,
                    "motif": motif_name,
                    "motif_start_1based": start + 1,
                    "motif_end_1based": end,
                    "motif_center_1based": center + 1,
                    "window": win,
                }
            )
    return windows, pd.DataFrame(rows)


def logo_frequency_table(windows: Sequence[str]) -> pd.DataFrame:
    if not windows:
        return pd.DataFrame()
    width = len(windows[0])
    rows = []
    for pos in range(width):
        counts = Counter(win[pos] for win in windows if len(win) == width and win[pos] in AA_SET)
        denom = sum(counts.values())
        for aa in AA_ORDER:
            rows.append(
                {
                    "position_in_window": pos + 1,
                    "residue": aa,
                    "count": counts.get(aa, 0),
                    "frequency": counts.get(aa, 0) / denom if denom else 0.0,
                }
            )
    return pd.DataFrame(rows)


def draw_logo(freq: pd.DataFrame, path: Path) -> None:
    if freq.empty:
        return
    positions = sorted(freq["position_in_window"].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(positions) * 0.42), 3.2))
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    for pos in positions:
        sub = freq[freq["position_in_window"] == pos].sort_values("frequency")
        y = 0.0
        for row in sub.itertuples(index=False):
            height = float(row.frequency)
            if height <= 0:
                continue
            text_path = TextPath((0, 0), row.residue, size=1, prop=fp)
            bbox = text_path.get_extents()
            sx = 0.8 / max(bbox.width, 1e-6)
            sy = height / max(bbox.height, 1e-6)
            trans = Affine2D().scale(sx, sy).translate(pos - 0.4, y)
            patch = PathPatch(text_path, transform=trans + ax.transData, color=AA_COLORS.get(row.residue, "#777777"), lw=0)
            ax.add_patch(patch)
            y += height
    ax.set_xlim(0.5, len(positions) + 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Position in motif-centered window")
    ax.set_ylabel("Residue frequency")
    ax.set_xticks(positions)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append("nan" if math.isnan(value) else f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    substitution_stats: Optional[Dict[str, int]],
    n_logo_windows: int,
    args,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# AmpDiff AMP property and transition analysis\n\n")
        handle.write(f"- Input sequences: {len(metrics)}\n")
        handle.write(f"- Valid standard-AA fraction: {metrics['valid_standard_aa'].mean():.4f}\n")
        handle.write("- Net charge was calculated with Biopython ProteinAnalysis.charge_at_pH(7.0).\n")
        handle.write("- Isoelectric point was calculated with Biopython ProteinAnalysis.isoelectric_point().\n")
        handle.write("- Hydrophobicity was calculated as Kyte-Doolittle GRAVY with Biopython ProteinAnalysis.gravy().\n")
        handle.write(f"- Amphipathicity was calculated as alpha-helical hydrophobic moment using the {args.amphipathicity_scale} scale and 100-degree residue spacing.\n")
        handle.write("\n## Property summary\n\n")
        handle.write(dataframe_to_markdown(summary))
        handle.write("\n\n")
        if substitution_stats is None:
            handle.write("## Substitution matrix\n\nNo paired input/generated CSV was provided; substitution analysis was not computed.\n\n")
        else:
            handle.write("## Substitution matrix\n\n")
            for key, value in substitution_stats.items():
                handle.write(f"- {key}: {value}\n")
            handle.write("\nThe substitution-frequency matrix is normalized by all non-identical standard-amino-acid substitution events.\n\n")
        if n_logo_windows == 0:
            handle.write("## Motif-centered logo\n\nNo motif-centered windows were available; sequence-logo analysis was not computed.\n")
        else:
            handle.write("## Motif-centered logo\n\n")
            handle.write(f"- Motif-centered windows: {n_logo_windows}\n")
            handle.write(f"- Window width: {args.logo_window}\n")
            handle.write("- Frequencies were computed after excluding padding positions from the denominator.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AmpDiff downstream AMP property and transition analysis")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--fasta", help="Generated AMP FASTA")
    input_group.add_argument("--csv", help="Generated AMP CSV")
    parser.add_argument("--seq-col", default="sequence", help="Sequence column for --csv")
    parser.add_argument("--id-col", default="sequence_id", help="ID column for --csv and paired CSV when present")
    parser.add_argument("--paired-csv", help="Optional paired input/generated CSV")
    parser.add_argument("--input-col", default="input_sequence", help="Input sequence column for --paired-csv")
    parser.add_argument("--generated-col", default="generated_sequence", help="Generated sequence column for --paired-csv")
    parser.add_argument("--active-motif-file", default=str(ROOT / "data/Generalizability/motif/AMP_act_Motifs.csv"))
    parser.add_argument("--motif-types", default="all", help="Comma-separated prosite,regular,merci or all")
    parser.add_argument("--logo-window", type=int, default=15)
    parser.add_argument("--amphipathicity-scale", choices=["kyte_doolittle", "eisenberg"], default="kyte_doolittle")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.fasta or args.csv).resolve().parent / "amp_property_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = load_sequences(args)
    metrics = compute_properties(seqs, args.amphipathicity_scale)
    summary = summarize_properties(metrics)
    metrics.to_csv(out_dir / "amp_property_metrics.csv", index=False)
    summary.to_csv(out_dir / "amp_property_summary.csv", index=False)

    substitution_stats = None
    if args.paired_csv:
        pairs = load_pairs(args.paired_csv, args.input_col, args.generated_col, args.id_col)
        counts, freq, events, substitution_stats = compute_substitutions(pairs)
        counts.to_csv(out_dir / "substitution_count_matrix.csv")
        freq.to_csv(out_dir / "substitution_frequency_matrix.csv")
        events.to_csv(out_dir / "substitution_events_long.csv", index=False)

    windows, logo_windows = extract_logo_windows(seqs, args.active_motif_file, args.motif_types, args.logo_window)
    if not logo_windows.empty:
        logo_windows.to_csv(out_dir / "motif_logo_windows.csv", index=False)
    freq = logo_frequency_table(windows)
    if not freq.empty:
        freq.to_csv(out_dir / "motif_logo_frequency_table.csv", index=False)
        draw_logo(freq, out_dir / "motif_sequence_logo.png")

    write_report(
        out_dir / "amp_property_report.md",
        metrics,
        summary,
        substitution_stats,
        len(windows),
        args,
    )

    print(f"Property metrics: {out_dir / 'amp_property_metrics.csv'}")
    print(f"Property summary: {out_dir / 'amp_property_summary.csv'}")
    if substitution_stats is not None:
        print(f"Substitution matrices: {out_dir / 'substitution_count_matrix.csv'}")
    if len(windows) > 0:
        print(f"Motif logo: {out_dir / 'motif_sequence_logo.png'}")
    print(f"Report: {out_dir / 'amp_property_report.md'}")


if __name__ == "__main__":
    main()
