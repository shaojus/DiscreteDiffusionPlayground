# Discrete Diffusion Playground

This repo benchmarks discrete generative models on analytically tractable 2D binary-encoded distributions, including GMM and checkerboard data.

## Setup

Install options:
```bash
pip install -r requirements.txt
pip install -e .
```

or with micromamba/conda:
```bash
micromamba env create -f environment.yaml
micromamba activate playground
```

## Training Workflow

`playground/train.py` is Hydra-driven. Main config is `conf/config.yaml` with grouped defaults:
- `model: ...`
- `data: ...`
- `optimizer: ...`
- `run: ...`
- `checkpoint: ...`

Typical override examples:
```bash
python playground/train.py model=sedd train.steps=20000 eval.interval=1000
python playground/train.py data=checkerboard model=ar optimizer.lr=1e-4
```

## Where Metrics Are Calculated

Evaluation metrics are computed during the evaluation step in `playground/train.py`. Generated binary sequences are decoded back into 2D coordinates, binned on a fixed evaluation grid, and compared against the target distribution.

The main metric code lives in:

```text
playground/metrics.py
```

Checkpoints:
- Save path is controlled by `train.save_path` (default `auto`).
- With `auto`, checkpoints are named from model config (and include run id if available).
- Checkpoints now store: `model`, `cfg`, `optimizer`, and `step`.

Modes:
```bash
# 1) Train from scratch (default)
python playground/train.py data=gmm train.steps=20000

# 2) Eval-only from checkpoint
python playground/train.py run.mode=eval checkpoint.path=ckpts/your_ckpt.pt wandb.enabled=false

# 3) Continue training from checkpoint
python playground/train.py run.mode=train checkpoint.path=ckpts/your_ckpt.pt checkpoint.resume=true train.steps=2000
```

## Sweeps (W&B + Slurm)

1. Setup: define the parameter grid in `sweep.yaml`, and set `WANDB_ENTITY` / `WANDB_PROJECT` in `.env`.
2. Launch:
```bash
python launch_sweep.py
```

`launch_sweep.py` creates the W&B sweep and submits `run_sweep.sh` via `sbatch`.  
`run_sweep.sh` activates `playground_test`, runs from the repo root, and starts `wandb agent "$SWEEP_ID"`.
