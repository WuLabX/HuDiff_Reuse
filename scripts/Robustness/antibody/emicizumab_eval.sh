#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/wucy/WUCHUYA/AmpDiff"
BASE_DIR="${ROOT_DIR}/results/Robustness/Ab/Emicizumab"
PYTHON="${PYTHON:-/mnt/wucy/miniconda3/envs/Hudiff/bin/python}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export T20_PAIR_CACHE_FPATH="${T20_PAIR_CACHE_FPATH:-${BASE_DIR}/t20_pair_cache.csv}"
export T20_PAIR_MAX_WORKERS="${T20_PAIR_MAX_WORKERS:-2}"

cd "${ROOT_DIR}"

for task in 1 2; do
  for temp in 0.6 1.0 1.4; do
    for seed in 2024 2025 2026; do
      sample_path="${BASE_DIR}/${task}/Emicizumab_FR_Temp${temp}/Seed${seed}_FR_Temp${temp}/sample_humanization_result.csv"
      if [[ ! -f "${sample_path}" ]]; then
        echo "Missing sample file: ${sample_path}" >&2
        exit 1
      fi
      echo "Evaluating ${sample_path}"
      "${PYTHON}" scripts/Robustness/antibody/emicizumab_eval.py "${sample_path}"
    done
  done
done
