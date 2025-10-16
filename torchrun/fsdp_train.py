import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel


def setup_distributed():
    """Initialize the distributed process group."""
    # torchrun sets these environment variables automatically
    # FSDP requires NCCL backend with CUDA GPUs
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    return rank, world_size


def cleanup_distributed():
    """Clean up the distributed process group."""
    torch.distributed.destroy_process_group()


def resolve_device() -> torch.device:
    """Resolve device for FSDP training."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FSDP requires CUDA GPUs. "
            "Ensure you're running on a machine with NVIDIA GPUs."
        )
    
    # Use the local rank to assign GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return torch.device(f"cuda:{local_rank}")


class SyntheticDataset(Dataset):
    """Synthetic dataset that generates random data."""
    
    def __init__(self, num_samples: int, input_dim: int, num_classes: int, seed: int = 42):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        # Use a fixed seed for reproducibility across workers
        torch.manual_seed(seed)
        # Pre-generate all data
        self.inputs = torch.randn(num_samples, input_dim)
        self.targets = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {"inputs": self.inputs[idx], "target": self.targets[idx]}


def train_loop_per_worker(rank: int, world_size: int, config: dict) -> None:
    """Main training function run by each worker process."""
    device = resolve_device()
    verbose = config.get("verbose", False)
    
    if verbose:
        print(f"[Rank {rank}/{world_size}] Starting training on device: {device}", flush=True)
    
    # Create dataset - same dataset on all workers, but will be sharded by DistributedSampler
    dataset = SyntheticDataset(num_samples=config["num_samples"], input_dim=32, num_classes=2, seed=42)
    
    # Create DistributedSampler to shard data across workers
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    
    if verbose:
        print(f"[Rank {rank}] Dataset size: {len(dataset)}, Samples per worker: {len(sampler)}", flush=True)
    
    # Initialize model
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    
    if verbose:
        # Print initial weight before FSDP
        first_param = next(model.parameters())
        print(f"[Rank {rank}] Initial weight sample BEFORE FSDP: {first_param.flatten()[:3]}", flush=True)
    
    # Move model to device and wrap with FSDP
    # FSDP shards model parameters, gradients, and optimizer states across workers
    # Unlike DDP which replicates the full model, FSDP reduces memory per GPU
    model = model.to(device)
    
    # Wrap with FSDP and specify device_id for proper initialization
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = FullyShardedDataParallel(model, device_id=local_rank)
    
    if verbose:
        # Check if FSDP wrapper was applied
        model_type = type(model).__name__
        print(f"[Rank {rank}] Model type after FSDP wrap: {model_type}, device: {device}", flush=True)
        
        # Note: After FSDP wrapping, parameters are sharded and accessing them directly
        # requires special handling. For simplicity, we'll skip weight printing here.
        print(f"[Rank {rank}] Model wrapped with FSDP - parameters are now sharded across workers", flush=True)
    
    # Initialize optimizer (no wrapping required!)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    step = 0
    num_epochs = config["epochs"]
    batch_size = config.get("batch_size", 32)
    
    for epoch in range(num_epochs):
        # Set epoch for sampler to ensure different shuffling each epoch
        sampler.set_epoch(epoch)
        
        # Create a fresh DataLoader for each epoch
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=False)
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract inputs and targets from batch
            inputs = batch["inputs"].to(device, dtype=torch.float32)
            targets = batch["target"].to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                # Only rank 0 prints summary metrics (unless verbose mode)
                if rank == 0 or verbose:
                    loss_val = float(loss.item())
                    if verbose:
                        print(f"[Rank {rank}] Epoch {epoch}, Step {step}: Loss = {loss_val:.4f}", flush=True)
                    else:
                        if rank == 0:
                            print(f"Epoch {epoch}, Step {step}: Loss = {loss_val:.4f}", flush=True)
            
            step += 1
    
    if rank == 0:
        print(f"\nTraining completed! Total steps: {step}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP Training with torchrun")
    parser.add_argument("--num-samples", type=int, default=10240, help="Total number of training samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per worker")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs from all workers")
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size = setup_distributed()
    
    if rank == 0:
        print(f"Starting FSDP distributed training with {world_size} workers...")
        print(f"Configuration: {args.num_samples} samples, {args.batch_size} batch size, {args.epochs} epochs")
        print(f"Backend: NCCL with CUDA GPUs")
    
    try:
        # Run training
        config = {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "verbose": args.verbose
        }
        train_loop_per_worker(rank=rank, world_size=world_size, config=config)
    finally:
        # Clean up distributed resources
        cleanup_distributed()


# Commands to run:
# === Single-node, Multi-GPU Training ===
# Basic run with 2 GPUs, 10240 samples (automatically sharded), 3 epochs:
# torchrun --nproc_per_node=2 ./torchrun/fsdp_train.py --num-samples 10240 --batch-size 32 --epochs 3
#
# With verbose mode to see FSDP behavior:
# torchrun --nproc_per_node=2 ./torchrun/fsdp_train.py --num-samples 10240 --batch-size 32 --epochs 3 --verbose
#
# === Multi-node (for SLURM) ===
# Node 0 (master):
# torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=<MASTER_IP> --master_port=29500 \
#   ./torchrun/fsdp_train.py --num-samples 10240 --batch-size 32 --epochs 3
#
# Node 1 (worker):
# torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 --master_addr=<MASTER_IP> --master_port=29500 \
#   ./torchrun/fsdp_train.py --num-samples 10240 --batch-size 32 --epochs 3

