## HF Accelerate DDP Training

### Local CPU (2 processes)

```bash
accelerate launch --cpu --num_processes 2 ./hf_accelerate/distr_train.py \
  --num-samples 10240 --batch-size 32 --epochs 3
```

Optional persistent config:

```bash
accelerate config  # choose CPU, 2 processes
accelerate launch ./hf_accelerate/distr_train.py --num-samples 10240 --batch-size 32 --epochs 3
```

### Notes
- Uses `Accelerator.prepare(...)` to handle process group setup, device placement, and dataloader sharding.
- Rank-safe logging via `accelerator.print`.
- Default: no mixed precision (CPU). Enable mp/strategies later via `accelerate config`.

### Multi-node (SLURM) sketch (future)

Two machines, 4 procs each (total 8). Replace `<MASTER_IP>` appropriately.

Machine 0 (rank 0):

```bash
accelerate launch --num_machines 2 --machine_rank 0 --num_processes 4 \
  --main_process_ip <MASTER_IP> --main_process_port 29500 \
  ./hf_accelerate/distr_train.py --num-samples 10240 --batch-size 32 --epochs 3
```

Machine 1 (rank 1):

```bash
accelerate launch --num_machines 2 --machine_rank 1 --num_processes 4 \
  --main_process_ip <MASTER_IP> --main_process_port 29500 \
  ./hf_accelerate/distr_train.py --num-samples 10240 --batch-size 32 --epochs 3
```

SLURM tip: you can derive ranks and world size from SLURM env vars or set them in an Accelerate YAML config and call `accelerate launch` within your `.sbatch` script.



