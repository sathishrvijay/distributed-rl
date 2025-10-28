#!/bin/bash
#SBATCH --job-name=torchrun-ddp-1n2g
#SBATCH --partition=a100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=torchrun_ddp_1n2g_%j.out
#SBATCH --error=torchrun_ddp_1n2g_%j.err

# Notes:
# - Requests 2 GPUs on a single node in the a100 partition
# - Ensure your venv includes a CUDA-enabled PyTorch build
#   (pip install per https://pytorch.org/get-started/locally/)

# Activate virtual environment
source slurm_setup/.venv/bin/activate

# Optional NCCL diagnostics (uncomment for debugging)
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1  # Uncomment if IB issues occur on this cluster

# Change to workspace directory
cd "$HOME/Development/distributed-rl"

# Run torchrun with 2 GPU workers on a single node
torchrun --nproc_per_node=2 torchrun/ddp_train.py \
  --use-gpu 1 \
  --num-samples 10240 \
  --batch-size 32 \
  --epochs 3


