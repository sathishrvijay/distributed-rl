import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import ray.data
import ray.train
from ray.train import ScalingConfig, report, get_context
from ray.train.torch import TorchTrainer, prepare_model, prepare_optimizer


def resolve_device() -> torch.device:
    """Resolve device for DDP training. Note: DDP doesn't support MPS, so we only use CUDA or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_training_dataset(num_samples: int, input_dim: int, num_classes: int) -> ray.data.Dataset:
    """Create a Ray dataset with synthetic training data."""
    data = []
    for _ in range(num_samples):
        inputs = torch.randn(input_dim).tolist()
        target = torch.randint(0, num_classes, (1,)).item()
        data.append({"inputs": inputs, "target": target})
    return ray.data.from_items(data)


def train_loop_per_worker(config: dict) -> None:
    device = resolve_device()
    verbose = config.get("verbose", False)
    rank = get_context().get_world_rank()
    world_size = get_context().get_world_size()
    batch_size = config.get("batch_size", 32)

    # Always print device assignment for verification
    print(f"[Rank {rank}/{world_size}] Assigned device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[Rank {rank}] GPU name: {torch.cuda.get_device_name(device)}", flush=True)

    # Get dataset shard for this worker - Ray automatically shards the data
    dataset_shard = ray.train.get_dataset_shard("train")
    
    if verbose:
        print(f"[Rank {rank}] Received dataset shard (data automatically sharded across workers)", flush=True)

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

    # Wrap model with DDP - this enables gradient synchronization across workers
    model = prepare_model(model).to(device)

    if verbose:
        # Check if DDP wrapper was applied
        model_type = type(model).__name__
        print(f"[Rank {rank}] Model type after prepare_model: {model_type}, device: {device}", flush=True)
        
        # Print initial weight after DDP (should be synchronized across workers)
        first_param = next(model.parameters())
        print(f"[Rank {rank}] Initial weight sample AFTER DDP: {first_param.flatten()[:3]}", flush=True)

    # Initialize optimizer and wrap it for better performance
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    optimizer = prepare_optimizer(optimizer)

    loss_fn = nn.CrossEntropyLoss()

    # Training loop - iterate over dataset batches
    step = 0
    num_epochs = config.get("epochs", 1)
    
    for epoch in range(num_epochs):
        # Create a fresh iterator for each epoch
        dataloader = dataset_shard.iter_torch_batches(batch_size=batch_size, dtypes=torch.float32)
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract inputs and targets from batch
            inputs = torch.as_tensor(batch["inputs"], dtype=torch.float32, device=device)
            targets = torch.as_tensor(batch["target"], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                metrics = {"loss": float(loss.item()), "step": step, "epoch": epoch, "rank": rank}
                report(metrics)
                if verbose:
                    # Show current weight sample to verify DDP weight synchronization
                    first_param = next(model.parameters())
                    weight_sample = first_param.flatten()[:3]
                    print(f"[Rank {rank}] Epoch {epoch}, Step {step}: Loss = {metrics['loss']:.4f}, Weight sample: {weight_sample}", flush=True)
            
            step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers for distributed training")
    parser.add_argument("--use-gpu", type=int, default=0, help="Whether to use GPU (1) or not (0)")
    parser.add_argument("--address", type=str, default=None, help="Ray cluster address")
    parser.add_argument("--num-samples", type=int, default=10240, help="Total number of training samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per worker")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs from all workers")
    args = parser.parse_args()

    ray.init(address=args.address)

    # Print configuration
    print(f"Ray Train Configuration:")
    print(f"  Number of workers: {args.num_workers}")
    print(f"  Use GPU: {bool(args.use_gpu)}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
    print(f"  Samples: {args.num_samples}, Batch size: {args.batch_size}, Epochs: {args.epochs}")

    # Create dataset once - will be automatically sharded across workers
    print(f"\nCreating training dataset with {args.num_samples} samples...")
    train_dataset = create_training_dataset(
        num_samples=args.num_samples,
        input_dim=32,
        num_classes=2
    )
    print(f"Dataset created. Starting distributed training with {args.num_workers} workers...")

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config={
            "lr": args.lr,
            "verbose": args.verbose,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
        datasets={"train": train_dataset},  # Ray automatically shards this across workers
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=bool(args.use_gpu),
        ),
    )
    result = trainer.fit()
    print("Training completed!")
    print(f"Final metrics: {result.metrics}")

# Commands to run:
# Basic run with 2 workers, 10240 samples (automatically sharded), 3 epochs:
# python3.11 ./ray-train/ddp_train.py --num-workers 2 --use-gpu 0
#
# With custom dataset size and batch size:
# python3.11 ./ray-train/ddp_train.py --num-workers 2 --num-samples 20480 --batch-size 64 --epochs 5
#
# With verbose mode to see data sharding and weight synchronization:
# RAY_DEDUP_LOGS=0 python3.11 ./ray-train/ddp_train.py --num-workers 2 --use-gpu 0 --verbose


