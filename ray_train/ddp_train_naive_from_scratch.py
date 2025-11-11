"""
From-scratch synchronous data-parallel training using:
  - PyTorch (autograd, optimizers, tensors — no torch.nn.parallel.DistributedDataParallel)
  - Ray Core (actors for a simple parameter server and workers)

It demonstrates:
  1) Sharding data across workers
  2) Computing local gradients with PyTorch per worker
  3) Averaging gradients and applying SGD in a centralized parameter server
  4) Broadcasting updated weights back to workers each step

Run locally:
  $ python ray_train/ddp_train_blocking_from_scratch.py --num-workers 2 --epochs 2 --verbose
"""

import time
import argparse
import asyncio
import math
import ray
import torch
import torch.nn as nn
from typing import List, Tuple


NUM_WORKERS = 2
INPUT_DIM = 32
HIDDEN_DIM = 64
NUM_CLASSES = 10
NUM_SAMPLES = 1024
EPOCHS = 2
BATCH_SIZE = 128
LR = 1e-2
SEED = 1337
DEVICE = "cpu"  # change to "cuda" for GPUs if available


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, n: int, input_dim: int, num_classes: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, input_dim, generator=g)
        W = torch.randn(input_dim, num_classes, generator=g)
        logits = self.x @ W
        self.y = torch.argmax(logits, dim=1)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# Helper functions
def get_params(model: nn.Module) -> List[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def set_params(model: nn.Module, new_params: List[torch.Tensor]):
    with torch.no_grad():
        for p, np in zip(model.parameters(), new_params):
            p.copy_(np)


def get_grads(model: nn.Module) -> List[torch.Tensor]:
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p))
        else:
            grads.append(p.grad.detach().clone())
    return grads


def zero_grads(model: nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


@ray.remote
class ParameterServer:
    """Naive synchronous Parameter Server (PS).

    Each step, it:
      1) Waits for `num_workers` gradient lists
      2) Averages them
      3) Applies SGD to the master weights
      4) Returns the updated weights to *each* caller

    This implements a barrier using an asyncio.Condition to coordinate arrivals.
    """

    def __init__(self, model_init_state: List[torch.Tensor], lr: float, num_workers: int, verbose: bool = False):
        self.num_workers = num_workers
        self.lr = lr
        self.verbose = verbose
        # Master parameters the PS owns and updates
        self.params = [p.clone() for p in model_init_state]

        # Step buffers / barrier state
        self._pending: List[List[torch.Tensor]] = []
        self._arrivals = 0
        self._last_broadcast: List[torch.Tensor] = [p.clone() for p in self.params]
        self._step = 0
        self._cond = asyncio.Condition()

    def _sgd_update(self, avg_grads: List[torch.Tensor]):
        with torch.no_grad():
            for p, g in zip(self.params, avg_grads):
                p -= self.lr * g

    async def report_and_get_weights(self, grads: List[torch.Tensor], rank: int) -> List[torch.Tensor]:
        # Barrier step id for this call
        async with self._cond:
            call_step = self._step
            # Arrive and deposit grads
            self._pending.append([g.clone() for g in grads])
            self._arrivals += 1
            if self.verbose:
                print(f"[PS] step={call_step} arrival {self._arrivals}/{self.num_workers} from worker {rank}")

            # Last to arrive performs the aggregation and update
            if self._arrivals == self.num_workers:
                avg_grads: List[torch.Tensor] = []
                for i in range(len(self._pending[0])):
                    stack = torch.stack([grads_i[i] for grads_i in self._pending], dim=0)
                    avg_grads.append(stack.mean(dim=0))
                self._sgd_update(avg_grads)
                self._last_broadcast = [p.clone() for p in self.params]
                if self.verbose:
                    print(f"[PS] step={call_step} aggregated grads and updated params; broadcasting to workers")
                # Reset for next step and notify waiting callers
                self._pending.clear()
                self._arrivals = 0
                self._step += 1
                self._cond.notify_all()
            else:
                # Wait until this step completes
                while self._step == call_step:
                    await self._cond.wait()

            return [p.clone() for p in self._last_broadcast]


@ray.remote
class Worker:
    def __init__(
        self,
        rank: int,
        world_size: int,
        device: str,
        seed: int,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_samples: int,
        verbose: bool = False,
    ):
        torch.manual_seed(seed + rank)
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.model = MLP(input_dim, hidden_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.verbose = verbose
        self._step = 0

        # Create *this worker's* data shard
        ds = SyntheticDataset(num_samples, input_dim, num_classes, seed=seed)
        self.indices = list(range(rank, len(ds), world_size))
        self.data = ds.x[self.indices].to(self.device)
        self.targets = ds.y[self.indices].to(self.device)

    def set_params(self, params: List[torch.Tensor]):
        set_params(self.model, [p.to(self.device) for p in params])

    def train_one_epoch(self, ps_handle, batch_size: int) -> Tuple[float, float]:
        """Returns (epoch_loss, epoch_acc)."""
        n = self.data.size(0)
        num_batches = math.ceil(n / batch_size)
        if self.verbose:
            print(f"[Worker {self.rank}] starting training with {num_batches} batches...")
        epoch_loss = 0.0
        correct = 0
        total = 0

        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, n)
            x = self.data[start:end]
            y = self.targets[start:end]

            # Forward
            logits = self.model(x)
            loss = self.criterion(logits, y)

            # Backward (local grads)
            zero_grads(self.model)
            loss.backward()

            # Extract grads -> CPU for Ray transport simplicity
            grads = [g.detach().cpu() for g in get_grads(self.model)]

            # Push grads to PS & get updated weights (synchronous barrier)
            if self.verbose:
                g0_shape = tuple(grads[0].shape) if len(grads) > 0 else ()
                print(f"[Worker {self.rank}] step={self._step} sending grads (first={g0_shape})")
            new_params = ray.get(ps_handle.report_and_get_weights.remote(grads, self.rank))

            # Load new weights
            self.set_params(new_params)
            if self.verbose:
                print(f"[Worker {self.rank}] step={self._step} received updated weights")
            self._step += 1

            # Track metrics
            epoch_loss += loss.item() * (end - start)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += (end - start)

        return epoch_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="From-scratch Ray synchronous data-parallel training")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Number of Ray worker actors")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size per worker")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate for PS SGD")
    parser.add_argument("--device", type=str, default=DEVICE, help='Device string, e.g., "cpu" or "cuda"')
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--input-dim", type=int, default=INPUT_DIM, help="Model input dimensionality")
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM, help="Hidden layer size")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="Number of classes")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Total synthetic samples")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging of PS/worker comms")
    return parser.parse_args()


def main():
    args = parse_args()
    ray.init(ignore_reinit_error=True, num_cpus=max(args.num_workers, 1))

    # Seed master model to initialize PS
    torch.manual_seed(args.seed)
    master_model = MLP(args.input_dim, args.hidden_dim, args.num_classes).to(args.device)
    init_params = [p.detach().cpu().clone() for p in master_model.parameters()]

    ps = ParameterServer.remote(init_params, args.lr, args.num_workers, args.verbose)
    workers = [
        Worker.remote(
            rank=i,
            world_size=args.num_workers,
            device=args.device,
            seed=args.seed,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            num_samples=args.num_samples,
            verbose=args.verbose,
        )
        for i in range(args.num_workers)
    ]

    # Broadcast initial weights directly to workers
    if args.verbose:
        print("[Main] Broadcasting initial parameters to workers")
    ray.get([w.set_params.remote(init_params) for w in workers])

    for epoch in range(args.epochs):
        results = ray.get([w.train_one_epoch.remote(ps, args.batch_size) for w in workers])
        # Aggregate metrics across workers (mean loss/acc)
        mean_loss = sum(r[0] for r in results) / len(results)
        mean_acc = sum(r[1] for r in results) / len(results)
        print(f"Epoch {epoch+1}/{args.epochs} | loss={mean_loss:.4f} | acc={mean_acc:.4f}")

    ray.shutdown()


if __name__ == "__main__":
    main()
