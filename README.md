# AmpDiff

AmpDiff is organized as a paper-support repository with two experiment modules:

- `Generalizability`: the original AmpDiff antimicrobial peptide design experiments.
- `Robustness`: HuDiff antibody and nanobody robustness experiments copied into this repository.

![pipeline](doc/process.svg)

## Layout

```text
AmpDiff/
  scripts/
    Generalizability/        AMP train/finetune/sample/eval scripts
    Robustness/
      antibody/              eval.py, sample.py, train.py, finetune.py, sample_for_anti_cdr.py
      nanobody/              eval.py, sample.py, train.py, finetune.py, sample_for_anti_cdr.py
  configs/
    Generalizability/
    Robustness/
  data/
    Generalizability/
    Robustness/
  dataset/
    Generalizability/
    Robustness/
  evaluation/
    Generalizability/
    Robustness/
  model/
    Generalizability/
    Robustness/
  utils/
    Generalizability/
    Robustness/
  checkpoints/
    Generalizability/
    Robustness/
  results/
    Generalizability/
    Robustness/
```

## AMP External Predictors

Install these predictors under the same parent directory as `AmpDiff`:

- PepNet: [https://github.com/Harkool/PepNet](https://github.com/Harkool/PepNet)
- AMPpred-MFA: [https://github.com/Jiangle525/AMPpred-MFA](https://github.com/Jiangle525/AMPpred-MFA)
- iAMP-Attenpred: [https://github.com/xingwxzz/iAMP-Attenpred](https://github.com/xingwxzz/iAMP-Attenpred)
- UniDL4BioPep: [https://github.com/dzjxzyd/UniDL4BioPep](https://github.com/dzjxzyd/UniDL4BioPep)
- HemoPI2: [https://github.com/raghavagps/hemopi2](https://github.com/raghavagps/hemopi2)

## Generalizability

Resources:

- [Generalizability raw data](https://doi.org/10.5281/zenodo.22231112)
- [Generalizability raw_data file list](dataset/Generalizability/raw_data/README.md)
- [Generalizability checkpoints](https://doi.org/10.5281/zenodo.22231112)
- [Generalizability checkpoint README](checkpoints/Generalizability/README.md)

Raw-data retraining path:

1. Download [Generalizability raw data](https://doi.org/10.5281/zenodo.22231112) and place the files under `dataset/Generalizability/raw_data/`.
2. Build the processed AMP train/test datasets:

```bash
python dataset/Generalizability/processing/prepare_ampdiff_datasets.py \
  --raw-dir dataset/Generalizability/raw_data \
  --out-dir data/Generalizability \
  --work-dir dataset/Generalizability/work \
  --motif-dir data/Generalizability/motif
```

3. Train and finetune the Generalizability checkpoints:

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

The robustness script directories are intentionally compact. Each submodule has five Python entrypoints plus shell wrappers:

```text
eval.py
sample.py
train.py
finetune.py
sample_for_anti_cdr.py
```

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

After either path, run robustness sampling and evaluation.

Antibody 3 seed x 3 temperature sampling and evaluation:

```bash
bash scripts/Robustness/antibody/sample_3seed_3temp.sh chicken
bash scripts/Robustness/antibody/eval_3seed_3temp.sh chicken
```

Supported antibody datasets are `chicken`, `rabbit`, `BH1`, and `Emicizumab`.

Nanobody/heavy-chain 3 seed x 3 temperature sampling and evaluation:

```bash
bash scripts/Robustness/nanobody/sample_3seed_3temp.sh shark349
bash scripts/Robustness/nanobody/eval_3seed_3temp.sh shark349
```

Supported nanobody/heavy-chain datasets include `shark349`, `HuAb348_H`, and `Humab25_H`.

Robustness input data lives in:

```text
data/Robustness/chicken/
data/Robustness/rabbit/
data/Robustness/BH1/
data/Robustness/Emicizumab/
data/Robustness/shark349.csv
data/Robustness/Ab_to-Nb/HuAb348_H.csv
data/Robustness/Ab_to-Nb/Humab25_H.csv
```

Robustness outputs are generated under `results/Robustness/`. HuDiff `My_Data` outputs are not copied into this repository.

## Citation

```bibtex
@article{ampdiff,
  title = {Pushing adaptive autoregressive diffusion to its limits from antibody humanization to antimicrobial peptide design},
  author = {Wu, Chuya},
  journal = {Nat. Mach. Intell.},
  year = {2026}
}
```
