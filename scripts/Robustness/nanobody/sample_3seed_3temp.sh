#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/wucy/WUCHUYA/AmpDiff"
PYTHON_BIN="${PYTHON_BIN:-/mnt/wucy/miniconda3/envs/Hudiff/bin/python}"
CKPT="${CKPT:-checkpoints/Robustness/nanobody/hudiffnb.pt}"
DATASET="${1:-shark349}"
SAMPLE_METHOD="${SAMPLE_METHOD:-gen}"
MODEL="${MODEL:-finetune_vh}"
SAMPLE_NUMBER="${SAMPLE_NUMBER:-1}"
TRY_NUMBER="${TRY_NUMBER:-10}"

case "${DATASET}" in
  shark349)
    DATA_FPATH="${DATA_FPATH:-data/Robustness/shark349.csv}"
    ;;
  HuAb348_H|Humab25_H)
    DATA_FPATH="${DATA_FPATH:-data/Robustness/Ab_to-Nb/${DATASET}.csv}"
    ;;
  *)
    DATA_FPATH="${DATA_FPATH:-data/Robustness/${DATASET}.csv}"
    ;;
esac

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

for temp in 0.6 1.0 1.4; do
  for seed in 2024 2025 2026; do
    out_dir="results/Robustness/Nb/${DATASET}/${DATASET}_${SAMPLE_METHOD}_Temp${temp}/Seed${seed}_${SAMPLE_METHOD}_Temp${temp}"
    "${PYTHON_BIN}" scripts/Robustness/nanobody/sample.py \
      --ckpt "${CKPT}" \
      --data_fpath "${DATA_FPATH}" \
      --sample_method "${SAMPLE_METHOD}" \
      --sample_number "${SAMPLE_NUMBER}" \
      --try_number "${TRY_NUMBER}" \
      --temperature "${temp}" \
      --seed "${seed}" \
      --model "${MODEL}" \
      --output_dir "${out_dir}"
  done
done
