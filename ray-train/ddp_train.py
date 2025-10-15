import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import ray
from ray.train import ScalingConfig, report, get_context
from ray.train.torch import TorchTrainer, prepare_model, prepare_optimizer


def resolve_device() -> torch.device:
    """Resolve device for DDP training. Note: DDP doesn't support MPS, so we only use CUDA or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_loop_per_worker(config: dict) -> None:
    device = resolve_device()
    verbose = config.get("verbose", False)
    rank = get_context().get_world_rank()

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

    for step in range(config["steps"]):
        inputs = torch.randn(1024, 32, device=device)
        targets = torch.randint(0, 2, (1024,), device=device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            metrics = {"loss": float(loss.item()), "step": step, "rank": rank}
            report(metrics)
            if verbose:
                # Show current weight sample to verify DDP weight synchronization
                first_param = next(model.parameters())
                weight_sample = first_param.flatten()[:3]
                print(f"[Rank {rank}] Step {step}: Loss = {metrics['loss']:.4f}, Weight sample: {weight_sample}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-gpu", type=int, default=0)
    parser.add_argument("--address", type=str, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs from all workers")
    args = parser.parse_args()

    ray.init(address=args.address)

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config={"steps": args.steps, "lr": args.lr, "verbose": args.verbose},
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=bool(args.use_gpu),
        ),
    )
    result = trainer.fit()
    print("Training completed!")
    print(f"Final metrics: {result.metrics}")

# Command to run:
# python3.11 ./ray-train/ddp_train.py --num-workers 2 --use-gpu 0 --steps 50
# With verbose mode (prints from all workers):
# RAY_DEDUP_LOGS=0 python3.11 ./ray-train/ddp_train.py --num-workers 2 --use-gpu 0 --steps 50 --verbose


