#!/bin/bash
#SBATCH --job-name=torchrun-ddp-1n2g
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=$HOME/Development/distributed-rl
#SBATCH --output=slurm_logs/torchrun_ddp_1n2g_%j.out
#SBATCH --error=slurm_logs/torchrun_ddp_1n2g_%j.err

# Notes:
# - Requests 2 GPUs on a single node in the a100 partition
# - Ensure your venv includes a CUDA-enabled PyTorch build
#   (pip install per https://pytorch.org/get-started/locally/)
# GPUs still need CPU help: host-side ops, launching CUDA kernels, NCCL comms, data loading/decoding, pinned-memory copies, logging, etc

mkdir -p "$HOME/Development/distributed-rl/slurm_logs"

# Activate virtual environment
source slurm_setup/.venv/bin/activate
export OMP_NUM_THREADS=1

# Optional NCCL diagnostics (uncomment for debugging)
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1  # Uncomment if IB issues occur on this cluster

# Change to workspace directory (redundant with --chdir, kept for clarity)
cd "$HOME/Development/distributed-rl"

# Run torchrun with 2 GPU workers on a single node
torchrun --nproc_per_node=2 torchrun/ddp_train.py \
  --use-gpu 1 \
  --num-samples 10240 \
  --batch-size 32 \
  --epochs 3


