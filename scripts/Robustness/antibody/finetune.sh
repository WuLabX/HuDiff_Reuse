#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/wucy/WUCHUYA/AmpDiff"
PYTHON_BIN="${PYTHON_BIN:-/mnt/wucy/miniconda3/envs/Hudiff/bin/python}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
"${PYTHON_BIN}" scripts/Robustness/antibody/finetune.py "$@"
