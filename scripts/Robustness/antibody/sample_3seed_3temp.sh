#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/wucy/WUCHUYA/AmpDiff"
PYTHON_BIN="${PYTHON_BIN:-/mnt/wucy/miniconda3/envs/Hudiff/bin/python}"
CKPT="${CKPT:-checkpoints/Robustness/antibody/hudiffab.pt}"
SAMPLE_METHOD="${SAMPLE_METHOD:-FR}"
SAMPLE_NUMBER="${SAMPLE_NUMBER:-1}"
TRY_NUMBER="${TRY_NUMBER:-1}"
DATASET="${1:-chicken}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

case "${DATASET}" in
  chicken)
    DATA_FPATH="data/Robustness/chicken/chicken_hudiff_input.csv"
    ;;
  rabbit)
    DATA_FPATH="data/Robustness/rabbit/rabbit_hudiff_input.csv"
    ;;
  BH1)
    DATA_FPATH="data/Robustness/BH1/bH1.csv"
    ;;
  Emicizumab)
    DATA_FPATH="data/Robustness/Emicizumab/1_Emicizumab.csv"
    ;;
  *)
    echo "Unsupported antibody dataset: ${DATASET}" >&2
    exit 1
    ;;
esac

for temp in 0.6 1.0 1.4; do
  out_dir="results/Robustness/Ab/${DATASET}/${DATASET}_${SAMPLE_METHOD}_Temp${temp}"
  "${PYTHON_BIN}" scripts/Robustness/antibody/sample.py \
    --ckpt "${CKPT}" \
    --ckpt_version finetune \
    --data_fpath "${DATA_FPATH}" \
    --data_name "${DATASET}" \
    --sample_method "${SAMPLE_METHOD}" \
    --sample_number "${SAMPLE_NUMBER}" \
    --try_number "${TRY_NUMBER}" \
    --temperature "${temp}" \
    --n_rounds 3 \
    --seed 2024 \
    --output_dir "${out_dir}"
done
