# SLURM Setup for Distributed RL

This directory contains the infrastructure for running distributed RL training jobs on SLURM clusters.

## Directory Structure

```
slurm_setup/
├── .venv/                      # Virtual environment (created by install_deps.sh)
├── pyproject.toml              # Python dependencies specification
├── install_deps.sh             # One-time setup script
├── submit_slurm_job.py         # Job submission helper
└── README.md                   # This file
```

## One-Time Setup

Before submitting any SLURM jobs, you need to create the virtual environment **once** on the SLURM cluster:

```bash
# SSH into your SLURM cluster
ssh your-cluster.edu

# Navigate to the repository
cd ~/distributed-rl/slurm_setup

# Run the installation script
bash install_deps.sh
```

This will:
1. Create a virtual environment at `slurm_setup/.venv` using `uv`
2. Install all dependencies from `pyproject.toml`
3. Make the environment available to all SLURM worker nodes (via shared filesystem)

**Note**: You only need to run this once. All subsequent SLURM jobs will use this pre-built environment.

## Submitting Jobs

### Using the Helper Script (Recommended)

The helper script validates scripts and provides clear feedback:

```bash
# Submit a single job
python submit_slurm_job.py ../slurm/torchrun_ddp_train_1node2cpu.sbatch

# Submit multiple jobs
python submit_slurm_job.py ../slurm/job1.sbatch ../slurm/job2.sbatch
```

### Manual Submission

You can also submit jobs directly using `sbatch`:

```bash
sbatch ../slurm/torchrun_ddp_train_1node2cpu.sbatch
```

## Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

View job output:
```bash
cat torchrun_ddp_<job_id>.out
cat torchrun_ddp_<job_id>.err
```

Cancel a job:
```bash
scancel <job_id>
```

## Available SLURM Scripts

- `../slurm/torchrun_ddp_train_1node2cpu.sbatch` - Single node, 2 CPU workers, torchrun DDP training
- `../slurm/ray_cluster.sbatch` - Multi-node Ray cluster training
- `../slurm/gpu_test.sbatch` - GPU availability test

## Environment Details

The virtual environment includes:
- **PyTorch 2.5.0** with CPU/CUDA support
- **Ray 2.37.0** for distributed training
- **Accelerate 0.33.0+** for HuggingFace training
- **Gymnasium** for RL environments
- **TensorBoard** for logging
- All dependencies from `../requirements.txt`

## Updating Dependencies

To add or update dependencies:

1. Edit `pyproject.toml` or `../requirements.txt`
2. Re-run the installation script:
   ```bash
   cd slurm_setup
   bash install_deps.sh
   ```

## Troubleshooting

### "uv: command not found"

The `uv` package manager should be available in your Hermit environment. If not:
```bash
source activate-hermit  # or equivalent for your cluster
```

### Virtual Environment Not Found

Make sure you've run `install_deps.sh` at least once on the SLURM cluster before submitting jobs.

### Permission Denied

Make the scripts executable:
```bash
chmod +x install_deps.sh submit_slurm_job.py
```

### Jobs Fail with Import Errors

Check that the virtual environment was properly created and all dependencies installed:
```bash
source slurm_setup/.venv/bin/activate
python -c "import torch; import ray; print('Success!')"
```

