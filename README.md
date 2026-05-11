# SAVAR

SAVAR is an open-source inference and training core codebase for Visual AutoRegressive (VAR) models with Jina-CLIP-v2 conditions.

## Project Structure

This directory isolates the clean, core VAR k=4 logic.

- `infer.py`: Lightweight inference script with `num_samples=1` by default. It takes a VAR checkpoint, Condition checkpoint, and generates an image from a given text prompt.
- `train.py`: The main VAR model training script.
- `train_condition.py`: The condition model training script to align text embeddings (Jina-CLIP-v2) and image tokens.
- `condition_model.py`: Architecture for the condition alignment model.
- `models/`: Contains the core VAR and VQVAE model architectures.
- `utils/`: Contains various utilities for data loading, logging, and argument parsing.

## Setup

Please make sure you have the required dependencies installed (e.g. `torch`, `torchvision`, `transformers`, `numpy`).

## Inference

Run `infer.py` to decode an image from text. It defaults to generating a single best sample without calculating PSNR/LPIPS/MS-SSIM selection metrics.

```bash
python infer.py --var_ckpt path/to/var.pth --condition_ckpt path/to/cond.pth --kodak_dir path/to/images
```

## Training

### Train Condition Model
You need to train the CondAlign model first using Jina-CLIP-v2 1024-dim features:

```bash
torchrun --nproc_per_node=8 train_condition.py \
    --data_path /path/to/imagenet
```

### Train VAR
Train the main VAR model conditioned on the previously trained condition model:

```bash
torchrun --nproc_per_node=8 train.py \
    --data_path /path/to/imagenet \
    --condition_ckpt path/to/cond.pth
```
