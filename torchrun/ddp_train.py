import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


def setup_distributed(backend: str = "gloo"):
    """Initialize the distributed process group.

    backend:
        "gloo" for CPU training, "nccl" for GPU training.
    """
    # torchrun sets these environment variables automatically
    torch.distributed.init_process_group(backend=backend)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    return rank, world_size


def cleanup_distributed():
    """Clean up the distributed process group."""
    torch.distributed.destroy_process_group()


def resolve_device(use_gpu: bool) -> torch.device:
    """Resolve device for DDP training."""
    if use_gpu and torch.cuda.is_available():
        # Use the local rank to assign GPU
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


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
    device = resolve_device(config["use_gpu"])
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
        # Print initial weight before DDP
        first_param = next(model.parameters())
        print(f"[Rank {rank}] Initial weight sample BEFORE DDP: {first_param.flatten()[:3]}", flush=True)
    
    # Move model to device and wrap with DDP
    model = model.to(device)
    if device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    if verbose:
        # Check if DDP wrapper was applied
        model_type = type(model).__name__
        print(f"[Rank {rank}] Model type after DDP wrap: {model_type}, device: {device}", flush=True)
        
        # Print initial weight after DDP (should be synchronized across workers)
        first_param = next(model.parameters())
        print(f"[Rank {rank}] Initial weight sample AFTER DDP: {first_param.flatten()[:3]}", flush=True)
    
    # Initialize optimizer (no wrapping required like for Ray Train!)
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
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        
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
                        # Show current weight sample to verify DDP weight synchronization
                        first_param = next(model.parameters())
                        weight_sample = first_param.flatten()[:3]
                        print(f"[Rank {rank}] Epoch {epoch}, Step {step}: Loss = {loss_val:.4f}, Weight sample: {weight_sample}", flush=True)
                    else:
                        if rank == 0:
                            print(f"Epoch {epoch}, Step {step}: Loss = {loss_val:.4f}", flush=True)
            
            step += 1
    
    if rank == 0:
        print(f"\nTraining completed! Total steps: {step}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Native DDP Training with torchrun")
    parser.add_argument("--use-gpu", type=int, default=0, help="Whether to use GPU (1) or not (0)")
    parser.add_argument("--num-samples", type=int, default=10240, help="Total number of training samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per worker")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs from all workers")
    args = parser.parse_args()
    
    # Setup distributed training
    ddp_backend = "nccl" if bool(args.use_gpu) and torch.cuda.is_available() else "gloo"
    rank, world_size = setup_distributed(backend=ddp_backend)
    
    if rank == 0:
        print(f"Starting distributed training with {world_size} workers...")
        print(f"Configuration: {args.num_samples} samples, {args.batch_size} batch size, {args.epochs} epochs")
    
    try:
        # Run training
        config = {"use_gpu": bool(args.use_gpu), "num_samples": args.num_samples, "batch_size": args.batch_size,
                  "epochs": args.epochs, "lr": args.lr, "verbose": args.verbose}
        train_loop_per_worker(rank=rank, world_size=world_size, config=config)
    finally:
        # Clean up distributed resources
        cleanup_distributed()


# Commands to run:
# Basic run with 2 workers on CPU, 10240 samples (automatically sharded), 3 epochs:
# torchrun --nproc_per_node=2 ./torchrun/ddp_train.py --num-samples 10240 --batch-size 32 --epochs 3
#
# With verbose mode to see data sharding and weight synchronization:
# torchrun --nproc_per_node=2 ./torchrun/ddp_train.py --num-samples 10240 --batch-size 32 --epochs 3 --verbose
#
# On a machine with GPUs:
# torchrun --nproc_per_node=2 ./torchrun/ddp_train.py --use-gpu 1 --num-samples 10240 --batch-size 32 --epochs 3

