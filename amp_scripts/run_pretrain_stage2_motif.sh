#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PRETRAIN_CKPT="${1:-checkpoints/pretrain/stage1_all.pt}"

python amp_scripts/amp_train.py \
  --data_path data/pretrain/ampdiff_pretrain_motif.csv \
  --config_path configs/amp_pretrain_stage2_motif.yml \
  --log_path logs/amp_pretrain \
  --train_loss fr \
  --motif_type prosite,regular,merci \
  --mode de \
  --scorer none \
  --require_active_motif True \
  --min_optimizable 1 \
  --init_checkpoint "$PRETRAIN_CKPT" \
  --ckpt_subdir pretrain \
  --ckpt_name stage2_motif.pt
