#!/usr/bin/env python3
"""Create 90%-internal / 60%-vs-test CD-HIT filtered training sets."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "Generalizability"
WORK = ROOT / "dataset" / "Generalizability" / "work_cdhit60"

TRAIN_FILES = [
    DATA / "pretrain" / "pretrain.csv",
    DATA / "pretrain" / "pretrain_motif.csv",
    DATA / "finetune" / "finetune_de.csv",
    DATA / "finetune" / "finetune_inp.csv",
]

TEST_FILES = [
    DATA / "test" / "ampdiff_test_de.csv",
    DATA / "test" / "ampdiff_test_inp.csv",
]


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_fasta(seqs: Iterable[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for i, seq in enumerate(seqs):
            handle.write(f">seq_{i}|len={len(seq)}\n{seq}\n")


def read_fasta_sequences(path: Path) -> list[str]:
    seqs: list[str] = []
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if chunks:
                    seqs.append("".join(chunks))
                    chunks = []
            else:
                chunks.append(line)
    if chunks:
        seqs.append("".join(chunks))
    return seqs


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def cdhit_word_size(identity: float) -> str:
    if identity >= 0.7:
        return "5"
    if identity >= 0.6:
        return "4"
    if identity >= 0.5:
        return "3"
    return "2"


def output_paths(input_csv: Path) -> tuple[Path, Path]:
    out_csv = input_csv.with_name(f"{input_csv.stem}_60{input_csv.suffix}")
    out_fasta = input_csv.with_name(f"{input_csv.stem}_60.fasta")
    return out_csv, out_fasta


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)

    test_sequences: list[str] = []
    for test_file in TEST_FILES:
        _, rows = read_rows(test_file)
        test_sequences.extend(row["sequence"] for row in rows)
    test_sequences = sorted(set(test_sequences))
    test_fasta = WORK / "ampdiff_test_union.fasta"
    write_fasta(test_sequences, test_fasta)

    summary: list[dict[str, object]] = []
    for train_file in TRAIN_FILES:
        fieldnames, rows = read_rows(train_file)
        seq_to_row = {row["sequence"]: row for row in rows}

        raw_fasta = WORK / f"{train_file.stem}.fasta"
        internal_fasta = WORK / f"{train_file.stem}.cdhit90.fasta"
        final_prefix = WORK / f"{train_file.stem}.cdhit90_vs_test60"
        write_fasta(seq_to_row, raw_fasta)

        run([
            "cd-hit",
            "-i", str(raw_fasta),
            "-o", str(internal_fasta),
            "-c", "0.9",
            "-n", cdhit_word_size(0.9),
            "-d", "0",
            "-M", "0",
            "-T", "0",
        ])
        run([
            "cd-hit-2d",
            "-i", str(test_fasta),
            "-i2", str(internal_fasta),
            "-o", str(final_prefix),
            "-c", "0.6",
            "-n", cdhit_word_size(0.6),
            "-d", "0",
            "-M", "0",
            "-T", "0",
        ])

        kept_sequences = read_fasta_sequences(final_prefix)
        kept_rows = [seq_to_row[seq] for seq in kept_sequences]
        out_csv, out_fasta = output_paths(train_file)
        written = write_rows(out_csv, fieldnames, kept_rows)
        write_fasta(kept_sequences, out_fasta)
        summary.append({
            "input": str(train_file),
            "output_csv": str(out_csv),
            "output_fasta": str(out_fasta),
            "input_rows": len(rows),
            "after_internal_90": len(read_fasta_sequences(internal_fasta)),
            "after_vs_test_60": written,
        })

    for item in summary:
        print(item)


if __name__ == "__main__":
    main()
