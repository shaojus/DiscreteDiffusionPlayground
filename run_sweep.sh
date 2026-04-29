#!/bin/bash
#SBATCH --job-name=sedd_sw_c
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --partition=any_gpu
#SBATCH --mem=50G
#SBATCH --array=0-1
#SBATCH --output=/net/galaxy/home/koes/jshao2/test/DiscreteDiffusionPlayground/logs/sedd_sweep_c-%A_%a.out

# Activate environment
eval "$(micromamba shell hook --shell=bash)"
micromamba activate playground_test

# Use the original submission directory (repo root), not Slurm spool path.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/net/galaxy/home/koes/jshao2/test/DiscreteDiffusionPlayground}"
cd "$PROJECT_DIR"

# Force W&B files into writable project-local directories on Slurm nodes.
export WANDB_DIR="$PROJECT_DIR/wandb"
export WANDB_CACHE_DIR="$PROJECT_DIR/.cache/wandb"
export WANDB_CONFIG_DIR="$PROJECT_DIR/.config/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

echo "Worker starting for Sweep: $SWEEP_ID"
echo "Entity: $WANDB_ENTITY | Project: $WANDB_PROJECT"
echo "PWD: $(pwd)"
echo "WANDB_DIR: $WANDB_DIR"

# Start the W&B agent
# It automatically picks up SWEEP_ID from the environment
wandb agent "$SWEEP_ID"

exit 0
