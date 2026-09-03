#!/usr/bin/env bash
set -euo pipefail

cd /mnt/wucy/WUCHUYA/AmpDiff

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_DISABLE_DYNAMO=1

PY=/mnt/wucy/miniconda3/envs/Hudiff/bin/python
SAMPLE=scripts/Generalizability/amp_sample.py

DE_FASTA=data/Generalizability/test/ampdiff_test_de.fasta
INP_FASTA=data/Generalizability/test/ampdiff_test_inp.fasta

PRETRAIN_CKPT=checkpoints/Generalizability/pretrain/stage2_motif.pt
DE_CKPT=checkpoints/Generalizability/de/de.pt
INP_CKPT=checkpoints/Generalizability/inp/inp.pt

DE_OUT=results/Generalizability/ampdiff_test_de
INP_OUT=results/Generalizability/ampdiff_test_inp
WORK=results/Generalizability/_generation_runs

TEMPERATURES=(0.6 1.0 1.4)
SEEDS=(2024 2025 2026)

mkdir -p "$DE_OUT" "$INP_OUT" "$WORK"

write_header() {
  local final_tsv="$1"
  printf "task\tmodel\tmode\tcheckpoint\tsubgroup\tseed_name\tseed_sequence\tvariant_name\tvariant_sequence\tsampling_seed\ttemperature\n" > "$final_tsv"
}

append_variants() {
  local sample_csv="$1"
  local final_tsv="$2"
  local task="$3"
  local model_name="$4"
  local mode="$5"
  local ckpt="$6"
  local subgroup="$7"

  "$PY" - "$sample_csv" "$final_tsv" "$task" "$model_name" "$mode" "$ckpt" "$subgroup" <<'PY'
import csv
import sys

sample_csv, final_tsv, task, model_name, mode, ckpt, subgroup = sys.argv[1:]
seed_sequences = {}
rows = []

with open(sample_csv, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        if row["type"] == "seed":
            seed_sequences[row["name"]] = row["sequence"]
            continue
        if row["type"] != "generated":
            continue
        name = row["name"]
        seed_name = name.split("_amp_sample_", 1)[0]
        rows.append({
            "task": task,
            "model": model_name,
            "mode": mode,
            "checkpoint": ckpt,
            "subgroup": subgroup,
            "seed_name": seed_name,
            "seed_sequence": seed_sequences.get(seed_name, ""),
            "variant_name": name,
            "variant_sequence": row["sequence"],
            "sampling_seed": row["seed"],
            "temperature": row["temperature"],
        })

fieldnames = [
    "task", "model", "mode", "checkpoint", "subgroup", "seed_name",
    "seed_sequence", "variant_name", "variant_sequence", "sampling_seed",
    "temperature",
]
with open(final_tsv, "a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writerows(rows)

print(f"[append] {final_tsv}: subgroup={subgroup} variants={len(rows)}")
PY
}

run_one() {
  local task="$1"
  local model_name="$2"
  local ckpt="$3"
  local input_fasta="$4"
  local mode="$5"
  local out_dir="$6"
  local hemolysis_flag=()

  if [[ "$mode" == "inp" ]]; then
    hemolysis_flag=(--hemolysis-guidance)
  fi

  local run_root="$WORK/${task}_${model_name}"
  local final_tsv="$out_dir/${model_name}_${task}_variants.tsv"
  rm -rf "$run_root"
  mkdir -p "$run_root"
  write_header "$final_tsv"

  for temp in "${TEMPERATURES[@]}"; do
    for sample_seed in "${SEEDS[@]}"; do
      local subgroup="Temp${temp}_Seed${sample_seed}"
      local run_dir="$run_root/$subgroup"
      mkdir -p "$run_dir"

      echo "[run] task=${task} model=${model_name} mode=${mode} temp=${temp} seed=${sample_seed} ckpt=${ckpt}"
      "$PY" "$SAMPLE" \
        --ckpt "$ckpt" \
        --input_fasta "$input_fasta" \
        --mode "$mode" \
        --guidance_scorer pepnet \
        "${hemolysis_flag[@]}" \
        --independent_scorers none \
        --num_samples 1 \
        --batch_size 1 \
        --n_rounds 1 \
        --temperature "$temp" \
        --sample_order shuffle \
        --seed "$sample_seed" \
        --output "$run_dir"

      local sample_csv
      sample_csv=$(find "$run_dir" -name sample_amp_result.csv -print | sort | head -n 1)
      append_variants "$sample_csv" "$final_tsv" "$task" "$model_name" "$mode" "$ckpt" "$subgroup"
    done
  done

  local total
  total=$(tail -n +2 "$final_tsv" | wc -l)
  echo "[write] ${final_tsv}: ${total} variants across 9 subgroups"
}

run_one "de"  "pretrain" "$PRETRAIN_CKPT" "$DE_FASTA"  "de"  "$DE_OUT"
run_one "de"  "de"       "$DE_CKPT"       "$DE_FASTA"  "de"  "$DE_OUT"
run_one "de"  "inp"      "$INP_CKPT"      "$DE_FASTA"  "de"  "$DE_OUT"

run_one "inp" "pretrain" "$PRETRAIN_CKPT" "$INP_FASTA" "inp" "$INP_OUT"
run_one "inp" "de"       "$DE_CKPT"       "$INP_FASTA" "inp" "$INP_OUT"
run_one "inp" "inp"      "$INP_CKPT"      "$INP_FASTA" "inp" "$INP_OUT"

echo "[done] main generation experiments complete"
