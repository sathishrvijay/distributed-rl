#!/bin/bash
#SBATCH --job-name=ray-ddp-1n2cpu
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/ray_ddp_1n2cpu_%j.out
#SBATCH --error=slurm_logs/ray_ddp_1n2cpu_%j.err

# Notes:
# - Requests 2 CPUs on debug partition for 2 Ray workers
# - Ray cluster is started/stopped within the script

# Change to workspace directory
cd "$HOME/Development/distributed-rl"

# Create log directory
mkdir -p slurm_logs

# Activate virtual environment
source slurm_setup/.venv/bin/activate

# Disable Ray log deduplication to see all worker output
export RAY_DEDUP_LOGS=0

# Start Ray cluster on this node
echo "Starting Ray cluster..."
ray start --head --port=6379 --num-cpus=2

# Wait for Ray to initialize
sleep 2

# Run Ray Train DDP with 2 CPU workers
echo "Running Ray Train DDP training..."
python ray_train/ddp_train.py \
  --num-workers 2 \
  --use-gpu 0 \
  --num-samples 10240 \
  --batch-size 32 \
  --epochs 3

echo ""
echo "Training script completed. Checking Ray logs..."
# Display recent worker logs for debugging
RAY_SESSION_DIR=$(ls -td /tmp/ray/session_* 2>/dev/null | head -1)
if [ -n "$RAY_SESSION_DIR" ]; then
  echo "Ray session directory: $RAY_SESSION_DIR"
  if [ -d "$RAY_SESSION_DIR/logs" ]; then
    echo "Available log files:"
    ls "$RAY_SESSION_DIR/logs"
    for pattern in "worker-*.out" "trainer_*.out" "python-core-worker*.log" "python-ray_driver*.log"; do
      for file in "$RAY_SESSION_DIR"/logs/$pattern; do
        [ -f "$file" ] || continue
        echo ""
        echo "--- Tail of $file ---"
        tail -n 200 "$file"
      done
    done
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

