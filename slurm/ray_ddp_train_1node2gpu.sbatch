#!/bin/bash
#SBATCH --job-name=ray-ddp-1n2gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/ray_ddp_1n2gpu_%j.out
#SBATCH --error=slurm_logs/ray_ddp_1n2gpu_%j.err

# Notes:
# - Requests 2 GPUs on a single node in the a100 partition
# - Ray cluster is started with GPU support
# - Ensure your venv includes a CUDA-enabled PyTorch build

# Change to workspace directory
cd "$HOME/Development/distributed-rl"

# Create log directory
mkdir -p slurm_logs

# Activate virtual environment
source slurm_setup/.venv/bin/activate
export OMP_NUM_THREADS=1

# Optional: Enable NCCL debug (uncomment for troubleshooting)
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1

# Start Ray cluster on this node with GPU support
echo "Starting Ray cluster with 2 GPUs..."
ray start --head --port=6379 --num-cpus=4 --num-gpus=2

# Wait for Ray to initialize
sleep 2

# Print GPU info
nvidia-smi

# Run Ray Train DDP with 2 GPU workers
echo "Running Ray Train DDP training on GPUs..."
python ray_train/ddp_train.py \
  --num-workers 2 \
  --use-gpu 1 \
  --num-samples 10240 \
  --batch-size 32 \
  --epochs 3

# Stop Ray cluster
echo "Stopping Ray cluster..."
ray stop

echo "Training completed!"

