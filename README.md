# Discrete Diffusion Playground

Set up the environment: 
```bash
micromamba create -f environment.yaml
```
Run an experiment with custom train steps and eval interval:
```bash
python playground/train.py model=sedd train.steps=20000 eval.interval=1000
```
Running Hyperparameter Sweeps
1. **Setup**: Define your grid in `sweep.yaml` and set `WANDB_ENTITY` / `WANDB_PROJECT` in `.env`.
2. **Launch**: Run the launcher to create the sweep and submit Slurm jobs:
   ```bash
   python launch_sweep.py
