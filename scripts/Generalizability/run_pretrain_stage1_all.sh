#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python scripts/Generalizability/amp_train.py \
  --data_path data/Generalizability/pretrain/ampdiff_pretrain_all.csv \
  --config_path configs/Generalizability/amp_pretrain_stage1_all.yml \
  --log_path logs/amp_pretrain \
  --train_loss fr \
  --motif_type prosite,regular,merci \
  --mode de \
  --scorer none \
  --require_active_motif False \
  --min_optimizable 1 \
  --ckpt_subdir pretrain \
  --ckpt_name stage1_all.pt
