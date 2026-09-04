# AmpDiff

AmpDiff is a paper-support repository for studying the reusability of HuDiff-style diffusion models across biological sequence design tasks. It contains two modules:

- `Generalizability`: antimicrobial peptide (AMP) design with AmpDiff.
- `Robustness`: HuDiff antibody and nanobody robustness experiments integrated into the same repository structure.

![pipeline](doc/process.svg)

## Contents

- [Overview](#overview)
- [Experimental Setup](#experimental-setup)
- [System Requirements](#system-requirements)
- [Reproduction](#reproduction)
- [Generalizability](#generalizability)
- [Robustness](#robustness)
- [Citation](#citation)

## Overview

AmpDiff adapts the HuDiff diffusion framework from antibody and nanobody humanization to AMP design. The repository is arranged so that AMP generalizability experiments and HuDiff robustness experiments share the same high-level module layout while keeping their data, checkpoints, model code, utilities, scripts, evaluation code, and results separated under `Generalizability` and `Robustness`.

The `Generalizability` module supports AMP dataset preparation, pretraining, finetuning, guided sampling, external AMP activity and hemolysis scoring, and AMP paper metrics. The `Robustness` module supports HuDiff-style antibody and nanobody robustness experiments with three random seeds and three sampling temperatures.

## Experimental Setup

All experiments were run on Linux servers with NVIDIA A100 GPUs. The default robustness and sampling scripts use the following random seeds and temperatures:

- Random seeds: `2024`, `2025`, `2026`
- Sampling temperatures: `0.6`, `1.0`, `1.4`

Outputs are written to `results/Generalizability/` for AMP experiments and `results/Robustness/` for HuDiff robustness experiments.

## System Requirements

The codebase was developed with Python 3.9, PyTorch 1.13.0, and CUDA 11.6. Create the conda environment with:

```bash
conda env create -f environment.yaml
conda activate AmpDiff
```

The environment file installs the main conda and pip dependencies, including `python-lmdb`, `abnumber`, `pandas`, `scipy`, `scikit-learn`, `tqdm`, `tensorboard`, `easydict`, `pyyaml`, `sequence-models`, `einops`, `matplotlib`, and `seaborn`.

Install the AMP external predictors under the same parent directory as `AmpDiff`:

```text
WUCHUYA/
  AmpDiff/
  PepNet/
  AMPpred-MFA/
  iAMP-Attenpred/
  UniDL4BioPep/
  hemopi2/
```

Predictor links:

- [PepNet](https://github.com/Harkool/PepNet)
- [AMPpred-MFA](https://github.com/Jiangle525/AMPpred-MFA)
- [iAMP-Attenpred](https://github.com/xingwxzz/iAMP-Attenpred)
- [UniDL4BioPep](https://github.com/dzjxzyd/UniDL4BioPep)
- [HemoPI2](https://github.com/raghavagps/hemopi2)

The robustness evaluation stack may also require HuDiff-related tools such as AbNatiV, BioPhi/OASis, ABLSTM, ANARCI, and `abnumber`.

## Reproduction

To reproduce the original HuDiff antibody and nanobody experiments, use the official [TencentAI4S/HuDiff](https://github.com/TencentAI4S/HuDiff) repository. This AmpDiff repository focuses on reusing the HuDiff framework for AMP generalizability and on robustness experiments using the three seeds `2024`, `2025`, `2026` and the three temperatures `0.6`, `1.0`, `1.4`.

## Generalizability

Resources:

- [Generalizability raw data](https://doi.org/10.5281/zenodo.22231112)
- [Generalizability raw_data file list](dataset/Generalizability/raw_data/README.md)
- [Generalizability checkpoints](https://doi.org/10.5281/zenodo.22231112)
- [Generalizability checkpoint README](checkpoints/Generalizability/README.md)

Raw-data retraining path:

1. Download [Generalizability raw data](https://doi.org/10.5281/zenodo.22231112) and place the files under `dataset/Generalizability/raw_data/`.
2. Build the processed AMP datasets:

```bash
python dataset/Generalizability/processing/prepare_ampdiff_datasets.py \
  --raw-dir dataset/Generalizability/raw_data \
  --out-dir data/Generalizability \
  --work-dir dataset/Generalizability/work \
  --motif-dir data/Generalizability/motif
```

3. Train and finetune the AMP checkpoints:

```bash
bash scripts/Generalizability/run_pretrain_stage1_all.sh
bash scripts/Generalizability/run_pretrain_stage2_motif.sh
python scripts/Generalizability/amp_finetune.py \
  --data_path data/Generalizability/finetune/ampdiff_finetune_de.csv \
  --config_path configs/Generalizability/amp_finetune.yml \
  --mode de
```

Warm-start path:

1. Download [Generalizability checkpoints](https://doi.org/10.5281/zenodo.22231112).
2. Place the checkpoint files under `checkpoints/Generalizability/`.

After either path, sample AMP variants:

```bash
bash scripts/Generalizability/run_main_generation.sh
```

Evaluate AMP outputs:

```bash
python evaluation/Generalizability/activity_hemolysis_eval.py \
  --input results/Generalizability/ampdiff_test_de/de_de_variants.tsv \
  --seq-col variant_sequence \
  --id-col variant_name

python evaluation/Generalizability/amp_metrics_eval.py \
  --input results/Generalizability/ampdiff_test_de/de_de_variants.tsv \
  --seq-col variant_sequence \
  --id-col variant_name \
  --train data/Generalizability/pretrain/ampdiff_pretrain.csv
```

## Robustness

Resources:

- [Robustness raw data](https://huggingface.co/cloud77/HuDiff)
- [Robustness raw_data file list](dataset/Robustness/raw_data/README.md)
- [Robustness checkpoints](https://huggingface.co/cloud77/HuDiff)
- [Robustness checkpoint README](checkpoints/Robustness/README.md)

Raw-data retraining path:

1. Download [Robustness raw data](https://huggingface.co/cloud77/HuDiff) and place the release-data files under `dataset/Robustness/raw_data/`.
2. Train and finetune the HuDiff robustness models:

```bash
bash scripts/Robustness/antibody/train.sh
bash scripts/Robustness/antibody/finetune.sh
bash scripts/Robustness/nanobody/train.sh
bash scripts/Robustness/nanobody/finetune.sh
```

Warm-start path:

1. Download [Robustness checkpoints](https://huggingface.co/cloud77/HuDiff).
2. Place the checkpoint files under `checkpoints/Robustness/`.

After either path, run antibody robustness sampling and evaluation with three seeds and three temperatures:

```bash
bash scripts/Robustness/antibody/sample_3seed_3temp.sh chicken
bash scripts/Robustness/antibody/eval_3seed_3temp.sh chicken
```

Supported antibody datasets are `chicken`, `rabbit`, `BH1`, and `Emicizumab`.

Run nanobody or heavy-chain robustness sampling and evaluation with the same seed-temperature grid:

```bash
bash scripts/Robustness/nanobody/sample_3seed_3temp.sh shark349
bash scripts/Robustness/nanobody/eval_3seed_3temp.sh shark349
```

Supported nanobody/heavy-chain datasets include `shark349`, `HuAb348_H`, and `Humab25_H`.

Robustness benchmark inputs are stored under `data/Robustness/`. Generated robustness outputs are stored under `results/Robustness/`; HuDiff `My_Data` outputs are not copied into this repository.

## Citation

```bibtex
@article{ampdiff,
  title = {Pushing adaptive autoregressive diffusion to its limits from antibody humanization to antimicrobial peptide design},
  author = {Wu, Chuya},
  journal = {Nat. Mach. Intell.},
  year = {2026}
}
```
