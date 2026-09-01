# AmpDiff
AmpDiff is a diffusion-based workflow for antimicrobial peptide (AMP) design and optimization. The repository contains the code used to prepare AMP datasets, train motif-aware pretraining and finetuning models, sample DE/INP peptide variants, and evaluate generated peptides with external AMP activity and hemolysis predictors.


![pipeline](doc/process.svg)

## Overview
This repository contains AmpDiff, the cross-domain generalizability component of a Reusability Report on HuDiff. AmpDiff transfers the HuDiff-Nb diffusion framework from nanobody humanization to antimicrobial peptide design to evaluate the architectural reusability of HuDiff beyond its original biological domain.

## Repository Layout

```text
AmpDiff/
  amp_scripts/              Training, finetuning, sampling, and auxiliary analysis entrypoints
  configs/                  AmpDiff training and finetuning YAML configs
  data/                     Processed pretrain, finetune, test, and motif data
  dataset/                  Dataset classes and dataset preparation scripts
    processing/             Reproducible data preparation scripts
    raw_data/               Merged raw AMP source tables
    work/                   CD-HIT and split-building working files
  evaluation/               Paper-facing predictor and AMP-metric evaluation scripts
  model/                    Diffusion model components
  results/                  Generated AmpDiff result tables
  utils/                    Shared tokenizer, motif, training, loss, and scorer utilities
```

## Environment

Create the conda environment from the provided file:

```bash
conda env create -f environment.yaml
conda activate AmpDiff
```

Install the external activity and hemolysis predictors under the same parent directory as `AmpDiff`:

```text
WUCHUYA/
  AmpDiff/
  PepNet/
  AMPpred-MFA/
  iAMP-Attenpred/
  UniDL4BioPep/
  hemopi2/
```

Download links:

- PepNet: [https://github.com/Harkool/PepNet](https://github.com/Harkool/PepNet)
- AMPpred-MFA: [https://github.com/Jiangle525/AMPpred-MFA](https://github.com/Jiangle525/AMPpred-MFA)
- iAMP-Attenpred: [https://github.com/xingwxzz/iAMP-Attenpred](https://github.com/xingwxzz/iAMP-Attenpred)
- UniDL4BioPep: [https://github.com/dzjxzyd/UniDL4BioPep](https://github.com/dzjxzyd/UniDL4BioPep)
- HemoPI2: [https://github.com/raghavagps/hemopi2](https://github.com/raghavagps/hemopi2)

The default local layout used by the scripts is `/mnt/wucy/WUCHUYA/<predictor-name>`. All predictor paths can also be overridden with CLI flags in the evaluation and sampling scripts.

## Data Preparation

The main dataset preparation script builds the pretraining, finetuning, and held-out test splits from the merged raw AMP tables:

```bash
python dataset/processing/prepare_ampdiff_datasets.py \
  --raw-dir dataset/raw_data \
  --out-dir data \
  --work-dir dataset/work \
  --motif-dir data/motif
```

Outputs are written to:

```text
data/pretrain/
data/finetune/
data/test/
data/dataset_report.json
data/dataset_report.md
```

For simple sequence-column cleaning and deduplication, use:

```bash
python dataset/processing/prepare_amp_pretrain.py \
  --input data/pretrain/AMP_51345.csv \
  --output data/pretrain/AMP_51345_clean.csv
```

## Training

Stage 1 pretraining on the broad AMP sequence pool:

```bash
bash amp_scripts/run_pretrain_stage1_all.sh
```

Stage 2 motif-focused pretraining:

```bash
bash amp_scripts/run_pretrain_stage2_motif.sh
```

Direct Python entrypoint:

```bash
python amp_scripts/amp_train.py \
  --data_path data/pretrain/ampdiff_pretrain_all.csv \
  --config_path configs/amp_pretrain_stage1_all.yml \
  --mode de \
  --log_path logs/amp_pretrain \
  --scorer none
```

## Finetuning

Finetuning uses activity-supervised AMP records and optional activity/hemolysis guidance:

```bash
python amp_scripts/amp_finetune.py \
  --data_path data/finetune/ampdiff_finetune_de.csv \
  --config_path configs/amp_finetune.yml \
  --mode de \
  --scorer pepnet \
  --log_path logs/amp_finetune
```

Use `--mode inp` for the hemolysis-aware inpainting task when the config and checkpoint are set accordingly.

## Sampling

Generate optimized variants from seed AMP FASTA files:

```bash
python amp_scripts/amp_sample.py \
  --ckpt checkpoints/de/de.pt \
  --input_fasta data/test/ampdiff_test_de.fasta \
  --mode de \
  --guidance_scorer pepnet \
  --independent_scorers auto \
  --eval-hemolysis \
  --output results/ampdiff_test_de/sample_run
```

For INP sampling, switch `--mode inp` and use the corresponding checkpoint and test FASTA.

## Evaluation

Run the external activity predictors and HemoPI2 on generated FASTA/CSV/TSV files:

```bash
python evaluation/activity_hemolysis_eval.py \
  --input results/ampdiff_test_de/de_de_variants.tsv \
  --seq-col variant_sequence \
  --id-col variant_name \
  --output-dir logs/evaluation/de_de_predictors
```

The output includes per-sequence scores, the mean activity score when multiple activity predictors are loaded, and a dual-hit flag based on activity >= 0.80 and hemolysis < 0.40.

Compute paper-facing AMP metrics:

```bash
python evaluation/amp_metrics_eval.py \
  --input results/ampdiff_test_de/de_de_variants.tsv \
  --seq-col variant_sequence \
  --id-col variant_name \
  --train data/pretrain/ampdiff_pretrain.csv \
  --output-dir logs/evaluation/de_de_metrics
```

`amp_metrics_eval.py` reports net charge at pH 7.4, Kyte-Doolittle GRAVY, alpha-helical hydrophobic moment, CHBI, novelty, dual-hit rate, and desirability. Helicity and aggregation propensity are imported from input/reference columns when supplied; otherwise they are left as `NaN` and recorded in the notes file.

## Results

Curated generated sequence tables are stored in:

```text
results/ampdiff_test_de/
results/ampdiff_test_inp/
```

Each TSV contains seed sequence metadata and generated `variant_sequence` records suitable for the evaluation scripts above.

## Citation

If you use AmpDiff code, model checkpoints, processed data, or generated designs in a publication, please cite the AmpDiff paper:

```bibtex
@article{ampdiff,
  title = {Pushing adaptive autoregressive diffusion to its limits from antibody humanization to antimicrobial peptide design},
  author = {Wu, Chuya},
  journal = {Nat. Mach. Intell.},
  year = {2026}
}
```
