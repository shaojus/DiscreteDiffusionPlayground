#!/bin/bash
#SBATCH --job-name=ar_sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --partition=any_gpu
#SBATCH --mem=50G
#SBATCH --array=0-3
#SBATCH --output=/net/galaxy/home/koes/jshao2/DiscreteDiffusionPlayground/logs/ar_sweep-%A_%a.out

# Activate environment
eval "$(micromamba shell hook --shell=bash)"
micromamba activate playground

echo "Worker starting for Sweep: $SWEEP_ID"
echo "Entity: $WANDB_ENTITY | Project: $WANDB_PROJECT"

# Start the W&B agent
# It automatically picks up SWEEP_ID from the environment
wandb agent "$SWEEP_ID"

exit 0