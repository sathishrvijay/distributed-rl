import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import ray
from ray.train import ScalingConfig, report
from ray.train.torch import TorchTrainer


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_loop_per_worker(config: dict) -> None:
    device = resolve_device()

    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    for step in range(config["steps"]):
        inputs = torch.randn(1024, 32, device=device)
        targets = torch.randint(0, 2, (1024,), device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            report({"loss": float(loss.item()), "step": step})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-gpu", type=int, default=0)
    parser.add_argument("--address", type=str, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    ray.init(address=args.address)

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config={"steps": args.steps, "lr": args.lr},
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=bool(args.use_gpu),
        ),
    )
    result = trainer.fit()
    print(result)

# Command to run:
# python3.11 ./ray-train/min_train.py --num-workers 2 --use-gpu 0 --steps 50"
