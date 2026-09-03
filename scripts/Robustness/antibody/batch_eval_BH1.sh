#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/wucy/WUCHUYA/AmpDiff"
PYTHON_BIN="${PYTHON_BIN:-/mnt/wucy/miniconda3/envs/Hudiff/bin/python}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PATH="/mnt/wucy/miniconda3/envs/biophi/bin:/mnt/wucy/miniconda3/envs/abnativ/bin:${PATH}"

cd "${ROOT_DIR}"

files=(
  "results/Robustness/Ab/BH1/BH1_FR_Temp0.6/Seed2024_FR_Temp0.6/sample_humanization_result.csv"
  "results/Robustness/Ab/BH1/BH1_FR_Temp0.6/Seed2025_FR_Temp0.6/sample_humanization_result.csv"
  "results/Robustness/Ab/BH1/BH1_FR_Temp0.6/Seed2026_FR_Temp0.6/sample_humanization_result.csv"
  "results/Robustness/Ab/BH1/BH1_FR_Temp1.0/Seed2024_FR_Temp1.0/sample_humanization_result.csv"
  "results/Robustness/Ab/BH1/BH1_FR_Temp1.0/Seed2025_FR_Temp1.0/sample_humanization_result.csv"
  "results/Robustness/Ab/BH1/BH1_FR_Temp1.0/Seed2026_FR_Temp1.0/sample_humanization_result.csv"
  "results/Robustness/Ab/BH1/BH1_FR_Temp1.4/Seed2024_FR_Temp1.4/sample_humanization_result.csv"
  "results/Robustness/Ab/BH1/BH1_FR_Temp1.4/Seed2025_FR_Temp1.4/sample_humanization_result.csv"
  "results/Robustness/Ab/BH1/BH1_FR_Temp1.4/Seed2026_FR_Temp1.4/sample_humanization_result.csv"
)

total=${#files[@]}
count=0

for file in "${files[@]}"; do
  count=$((count + 1))
  if [[ ! -f "${file}" ]]; then
    echo "Missing sample file: ${file}" >&2
    exit 1
  fi
  echo "[$count/$total] Evaluating ${file}"
  "${PYTHON_BIN}" scripts/Robustness/antibody/2B04_eval.py "$file" 2>&1
done

echo "All ${total} BH1 evaluation jobs finished."
