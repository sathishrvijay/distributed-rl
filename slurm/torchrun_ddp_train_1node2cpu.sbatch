#!/bin/bash
#SBATCH --job-name=torchrun-ddp-train
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=torchrun_ddp_%j.out
#SBATCH --error=torchrun_ddp_%j.err

# Partition notes:
# - 'debug' partition: max 2 hours, good for testing and short runs
# - For longer training runs, change to: --partition=compute (max 5 days)

# Activate virtual environment
source slurm_setup/.venv/bin/activate

# Change to workspace directory
cd $HOME/Development/distributed-rl

# Run torchrun with 2 worker processes on CPU
torchrun --nproc_per_node=2 torchrun/ddp_train.py \
  --num-samples 10240 \
  --batch-size 32 \
  --epochs 3

