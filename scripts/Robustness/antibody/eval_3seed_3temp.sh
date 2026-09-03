#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/wucy/WUCHUYA/AmpDiff"
PYTHON_BIN="${PYTHON_BIN:-/mnt/wucy/miniconda3/envs/Hudiff/bin/python}"
DATASET="${1:-chicken}"
SAMPLE_METHOD="${SAMPLE_METHOD:-FR}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

for temp in 0.6 1.0 1.4; do
  for seed in 2024 2025 2026; do
    sample_path="results/Robustness/Ab/${DATASET}/${DATASET}_${SAMPLE_METHOD}_Temp${temp}/Seed${seed}_${SAMPLE_METHOD}_Temp${temp}/sample_humanization_result.csv"
    if [[ ! -f "${sample_path}" ]]; then
      echo "Missing sample file: ${sample_path}" >&2
      exit 1
    fi
    "${PYTHON_BIN}" scripts/Robustness/antibody/eval.py --dataset "${DATASET}" "${sample_path}"
  done
done
