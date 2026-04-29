# Discrete Diffusion Playground

This repo benchmarks discrete generative models on analytically tractable 2D binary-encoded distributions, including GMM and checkerboard data.

## Setup

Python assumptions:
- Python 3.10+
- CUDA is optional; device is controlled by `DEVICE` env var (`cuda` by default, `cpu` for local smoke tests)

Dependencies used by the current workflow:
- `torch`
- `hydra-core`
- `omegaconf`
- `wandb`
- `matplotlib`
- `numpy`
- `tqdm`

Install options:
```bash
pip install -r requirements.txt
pip install -e .
```

or with micromamba/conda:
```bash
micromamba env create -f environment.yaml
micromamba activate playground_test
```

## Training Workflow

`playground/train.py` is Hydra-driven. Main config is `conf/config.yaml` with grouped defaults:
- `model: ...`
- `data: ...`
- `optimizer: ...`

Typical override examples:
```bash
python playground/train.py model=sedd train.steps=20000 eval.interval=1000
python playground/train.py data=checkerboard model=ar optimizer.lr=1e-4
```

Smoke tests:
```bash
python playground/train.py data=gmm train.steps=2 eval.interval=1 eval.n_samples=16 wandb.enabled=false
python playground/train.py data=checkerboard train.steps=2 eval.interval=1 eval.n_samples=16 wandb.enabled=false
python playground/train.py model=ar data=gmm train.steps=2 eval.interval=1 eval.n_samples=16 wandb.enabled=false
```

Checkpoints are controlled by `train.save_path` (default `ckpts/model.pt`).  
In sweeps, `sweep.yaml` sets per-run paths via:
- `train.save_path=ckpts/${envvar:WANDB_RUN_ID}.pt`

## Sweeps (W&B + Slurm)

1. Setup: define the parameter grid in `sweep.yaml`, and set `WANDB_ENTITY` / `WANDB_PROJECT` in `.env`.
2. Launch:
```bash
python launch_sweep.py
```

`launch_sweep.py` creates the W&B sweep and submits `run_sweep.sh` via `sbatch`.  
`run_sweep.sh` activates `playground_test`, runs from the repo root, and starts `wandb agent "$SWEEP_ID"`.
