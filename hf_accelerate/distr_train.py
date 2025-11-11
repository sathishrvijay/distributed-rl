import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed


class SyntheticDataset(Dataset):
    """Synthetic dataset that generates random data.

    Matches the behavior of the torchrun examples for apples-to-apples testing.
    """

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


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )


def train_loop_per_worker(accelerator: Accelerator, config: dict) -> None:
    """Main training function run by each worker process.
    
    Args:
        accelerator: Accelerator instance handling distributed environment
        config: Training configuration dictionary
    """
    verbose = config.get("verbose", False)

    if accelerator.is_main_process:
        accelerator.print(
            f"Starting Accelerate training with {accelerator.num_processes} processes on device {accelerator.device}"
        )
        accelerator.print(
            f"Configuration: {config['num_samples']} samples, {config['batch_size']} batch size, {config['epochs']} epochs"
        )
    
    # Start timing
    start_time = time.time()

    # Dataset/Dataloader (Accelerate will replace sampler for distributed runs)
    dataset = SyntheticDataset(num_samples=config["num_samples"], input_dim=32, num_classes=2, seed=42)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=False)

    # Model/optimizer/loss
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    # Prepare objects for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Optional: show a small weight sample to verify synchronization
    if verbose:
        # accelerator.print(f"Initial weight sample: {first_param.flatten()[:3]}") -> only prints from rank 0 by default
        first_param = next(model.parameters())
        print(f"[Rank {accelerator.process_index}] Initial weight sample: {first_param.flatten()[:3]}", flush=True)

    step = 0
    for epoch in range(config["epochs"]):
        # No need to set sampler epoch; Accelerate manages shuffling per epoch internally
        for batch in dataloader:
            # With device_placement=True (default), tensors are moved to the correct device.
            inputs = batch["inputs"].to(accelerator.device, dtype=torch.float32)
            targets = batch["target"].to(accelerator.device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            accelerator.backward(loss)
            optimizer.step()

            if step % 10 == 0:
                if verbose:
                    # Print from all ranks showing which rank is reporting
                    print(
                        f"[Rank {accelerator.process_index}] Epoch {epoch}, Step {step}: Loss = {float(loss.item()):.4f}",
                        flush=True
                    )
                else:
                    if accelerator.is_main_process:
                        accelerator.print(
                            f"Epoch {epoch}, Step {step}: Loss = {float(loss.item()):.4f}"
                        )
            step += 1

    # End timing and report
    end_time = time.time()
    total_time = end_time - start_time
    
    if accelerator.is_main_process:
        accelerator.print(f"\nTraining completed! Total steps: {step}")
        accelerator.print(f"Total training time: {total_time:.2f}s")
        accelerator.print(f"Time per step: {total_time / step:.4f}s")
    
    if verbose:
        print(f"[Rank {accelerator.process_index}] Total time: {total_time:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="DDP training using Hugging Face Accelerate (CPU-friendly)")
    parser.add_argument("--num-samples", type=int, default=10240, help="Total number of training samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per worker")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs from all workers")
    args = parser.parse_args()
    
    # Setup Accelerator (handles process group setup, device placement, data sharding)
    # torchrun spawns this script on each worker, so Accelerator() is initialized per process
    accelerator = Accelerator()
    set_seed(42)
    
    # Build config dict
    config = {
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "verbose": args.verbose,
    }
    
    # Run training loop
    train_loop_per_worker(accelerator=accelerator, config=config)


if __name__ == "__main__":
    main()


# Commands to run: (use --verbose to see output from all ranks)
# === Local CPU, 2 processes (recommended) ===
# torchrun --nproc_per_node=2 ./hf_accelerate/distr_train.py \
#   --num-samples 10240 --batch-size 32 --epochs 3
#
# Alternative (using accelerate launch, may not work on macOS):
# accelerate launch --cpu --num_processes 2 ./hf_accelerate/distr_train.py \
#   --num-samples 10240 --batch-size 32 --epochs 3
#
# === Multi-node (SLURM) ===
# Use torchrun with multi-node flags:
# torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
#   --master_addr=<MASTER_IP> --master_port=29500 \
#   ./hf_accelerate/distr_train.py --num-samples 10240 --batch-size 32 --epochs 3
#
# Or use accelerate launch (may require YAML config):
# accelerate launch --num_machines 2 --machine_rank 0 --num_processes 4 \
#   --main_process_ip <MASTER_IP> --main_process_port 29500 \
#   ./hf_accelerate/distr_train.py --num-samples 10240 --batch-size 32 --epochs 3
