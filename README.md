# CLIP-SDMG
This repository contains the source code of CLIP-SDMG [CLIP-SDMG:CLIP knowledge distillation based on semantic decoupling and  mask generation].



## Install

Python >= 3.8 and PyTorch >= 2.3 are required (the code uses the unified `torch.amp` API).

```
pip install -r requirements-training.txt
```



## Dataset preparation

### Conceptual Captions 3M 

OpenCLIP reads a CSV file with two columns: a path to an image, and a text caption. 

Our downloaded CC3M training set contains 2.89M images, and our CC3M validation set contains 13K images.


The generated `cc3m_train.csv` is:
```
title   filepath
XXXXXX  train/X/X.jpg
...     ...
```

The generated `cc3m_val.csv` is:
```
title   filepath
XXXXXX  val/X/X.jpg
...     ...
```

## Training

Run the following command from the repository root (`src` is imported as a namespace package):

```bash
python -m src.training.main_kd \
  --train-data /path/to/cc3m_train.csv \
  --data-root /path/to/train/images \
  --val-data /path/to/cc3m_val.csv \
  --val-data-root /path/to/val/images \
  --model ViT-T-16 \
  --t-model ViT-B-16 \
  --t-model-checkpoint /path/to/ViT_B_16_cc3m_12m_ep32.pt \
  --batch-size 64 \
  --epochs 32 \
  --precision amp \
  --logs ./logs
```

- Dataset and checkpoint paths have no default values; pass them explicitly.
- If the image paths in the CSV are already absolute, `--data-root` / `--val-data-root` can be omitted.
- Logs, TensorBoard events and checkpoints are written to `--logs/<experiment-name>/`.
- `--report-to tensorboard` is the default; use `--report-to wandb` or `--report-to tensorboard,wandb` for Weights & Biases.
- On Windows PowerShell, replace the trailing `\` line continuations with a backtick.

## Evaluation

```bash
python -m src.training.main_kd \
  --val-data /path/to/cc3m_val.csv \
  --val-data-root /path/to/val/images \
  --t-model-checkpoint /path/to/ViT_B_16_cc3m_12m_ep32.pt \
  --resume ./logs/<experiment-name>/checkpoints/epoch_latest.pt \
  --logs ./logs \
  --name eval
```

The student checkpoint is loaded from `--resume`; without `--train-data` the script runs evaluation only.

## Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` / `--t-model` | `ViT-T-16` / `ViT-B-16` | Student / teacher backbone names (see `src/open_clip/model_configs`). |
| `--alpha_fd_loss` | `2000` | Weight of the feature-distillation loss. |
| `--alpha_pgs_loss` | `1` | Weight of the progressive global-semantic loss. |
| `--alpha_dab_loss` | `1` | Weight of the attention-based shallow imitation loss. |
| `--alpha_gen_loss` | `1` | Weight of the masked-generation loss. |
| `--precision` | `amp` | One of `amp`, `amp_bf16`, `bf16`, `fp16`, `fp32`. |
| `--workers` | `4` | DataLoader worker processes. |
| `--val-frequency` | `1` | Run retrieval evaluation every N epochs. |
| `--save-frequency` | `1` | Save a checkpoint every N epochs. |
| `--deterministic` | off | Enables deterministic cuDNN and disables `cudnn.benchmark`. |
| `--grad-clip-norm` | none | Optional gradient clipping norm. |

Evaluation uses chunked retrieval-metric computation, so peak memory stays bounded on the 13K CC3M validation set.

## Tests

```bash
pip install pytest
python -m pytest -q
```

The test suite covers MLM batched decoding, PGS loss equivalence, sparse transformer collection, chunked retrieval metrics, and an end-to-end CPU forward/loss check.



## Supplementary Materials

Further supplementary experiments and theoretical details can be found at the following link.

[https://ars.els-cdn.com/content/image/1-s2.0-S0950705125012353-mmc1.pdf](https://ars.els-cdn.com/content/image/1-s2.0-S0950705125012353-mmc1.pdf)



## ViT-B/16 as the teacher

| [CC3M+12M teacher](https://github.com/MrS2203/CLIP-SDMG/releases/download/v1.0/ViT_B_16_cc3m_12m_ep32.pt) | [laion400m teacher](https://github.com/MrS2203/CLIP-SDMG/releases/download/v1.0/ViT_B_16-laion400m_e32.pt) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |



## Acknowledgement

Our codebase is built over [open_clip](https://github.com/mlfoundations/open_clip), an open-source codebase to run CLIP models.

We would appreciate it if our paper and repo are helpful to you!
```
@article{zhang2025clip,
  title={CLIP-SDMG: CLIP Knowledge Distillation based on semantic decoupling and mask generation},
  author={Zhang, Qian and Si, Zhicheng and Shao, Mingwen and Liang, Hong},
  journal={Knowledge-Based Systems},
  pages={114194},
  year={2025},
  publisher={Elsevier}
}
```
