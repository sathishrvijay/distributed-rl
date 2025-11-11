# torchrun Distributed Training Examples

Two reference scripts live here:

- `ddp_train.py`: standard DistributedDataParallel (DDP) that runs on CPU or GPU.
- `fsdp_train.py`: Fully Sharded Data Parallel (FSDP) for large models on CUDA GPUs.

## DDP quick start

```bash
# 2 CPU workers
torchrun --nproc_per_node=2 ./torchrun/ddp_train.py --num-samples 10240 --batch-size 32 --epochs 3

# Verbose logging from every rank
torchrun --nproc_per_node=2 ./torchrun/ddp_train.py --num-samples 10240 --batch-size 32 --epochs 3 --verbose

# GPUs (sets NCCL backend automatically when CUDA is available)
torchrun --nproc_per_node=2 ./torchrun/ddp_train.py --use-gpu 1 --num-samples 10240 --batch-size 32 --epochs 3
```

CLI flags:
- `--use-gpu 1` switches from CPU/gloo to GPU/NCCL (falls back to CPU if CUDA missing).
- `--verbose` mirrors internal state on every rank; otherwise only rank 0 logs progress.
- `--num-samples`, `--batch-size`, `--epochs`, `--lr` control the synthetic workload.

## FSDP quick start (CUDA only)

```bash
# 2 GPUs on one node
torchrun --nproc_per_node=2 ./torchrun/fsdp_train.py --num-samples 10240 --batch-size 32 --epochs 3

# Extra logging
torchrun --nproc_per_node=2 ./torchrun/fsdp_train.py --num-samples 10240 --batch-size 32 --epochs 3 --verbose
```

Multi-node example:

```bash
# Node 0
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
  --master_addr=<MASTER_IP> --master_port=29500 \
  ./torchrun/fsdp_train.py --num-samples 10240 --batch-size 32 --epochs 3

# Node 1
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
  --master_addr=<MASTER_IP> --master_port=29500 \
  ./torchrun/fsdp_train.py --num-samples 10240 --batch-size 32 --epochs 3
```

FSDP specifics:
- Always requires NVIDIA GPUs with NCCL; the script raises if CUDA is unavailable.
- Parameters and optimizer states are sharded across ranks, so per-GPU memory stays low.

## What the scripts demonstrate

- torchrun bootstraps rank/world size via environment variables set per process.
- `DistributedSampler` shards the synthetic dataset consistently across ranks.
- Only rank 0 emits progress by default; use `--verbose` to echo diagnostics everywhere.
- Cleanup is handled with `torch.distributed.destroy_process_group()` in both scripts.

