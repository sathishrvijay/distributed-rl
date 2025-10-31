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

# Disable Ray log deduplication to see all worker output
export RAY_DEDUP_LOGS=0

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

echo ""
echo "Training script completed. Checking Ray logs..."
# Display recent worker logs for debugging
RAY_SESSION_DIR=$(ls -td /tmp/ray/session_* 2>/dev/null | head -1)
if [ -n "$RAY_SESSION_DIR" ]; then
  echo "Ray session directory: $RAY_SESSION_DIR"
  LOG_ROOT="$RAY_SESSION_DIR/logs"
  ARCHIVE_ROOT="$HOME/Development/distributed-rl/slurm_logs/ray_sessions/$SLURM_JOB_ID"
  mkdir -p "$ARCHIVE_ROOT"

  if [ -d "$LOG_ROOT" ]; then
    echo "Copying Ray logs to $ARCHIVE_ROOT"
    cp -r "$LOG_ROOT" "$ARCHIVE_ROOT/" 2>/dev/null || true

    if [ -d "$ARCHIVE_ROOT/logs" ]; then
      INSPECT_ROOT="$ARCHIVE_ROOT/logs"
    else
      INSPECT_ROOT="$LOG_ROOT"
    fi

    echo "Available log files (inspect root: $INSPECT_ROOT):"
    ls "$INSPECT_ROOT"

    echo ""
    echo "Collecting trainer/worker logs..."
    mapfile -t RAY_LOG_FILES < <(find "$INSPECT_ROOT" -maxdepth 2 -type f \
      \( -name "worker-*.out" -o -name "worker-*.log" \
         -name "trainer_*.out" -o -name "trainer_*.log" \
         -name "python-core-worker*.log" -o -name "python-core-driver*.log" \
         -name "python-ray_driver*.log" -o -name "python-worker-*.log" \
      \) 2>/dev/null | sort)

    if [ ${#RAY_LOG_FILES[@]} -gt 0 ]; then
      for file in "${RAY_LOG_FILES[@]}"; do
        [ -f "$file" ] || continue
        echo ""
        echo "Printing log file: $file"
        echo "--- Tail of $file ---"
        tail -n 200 "$file"
      done
    else
      echo "No matching trainer/worker logs found. Listing top-level logs for reference:"
      find "$INSPECT_ROOT" -maxdepth 2 -type f -print
    fi
  else
    echo "No logs directory found under Ray session."
  fi
else
  echo "No Ray session directory found."
fi

# Stop Ray cluster
echo "Stopping Ray cluster..."
ray stop

echo "Training completed!"

