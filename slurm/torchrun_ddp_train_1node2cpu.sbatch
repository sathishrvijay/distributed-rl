#!/bin/bash
#SBATCH --job-name=torchrun-ddp-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=torchrun_ddp_%j.out
#SBATCH --error=torchrun_ddp_%j.err

# Activate virtual environment
source slurm_setup/.venv/bin/activate

# Change to workspace directory
cd /Users/vsathish/Development/distributed-rl

# Run torchrun with 2 worker processes on CPU
torchrun --nproc_per_node=2 torchrun/ddp_train.py \
  --num-samples 10240 \
  --batch-size 32 \
  --epochs 3

