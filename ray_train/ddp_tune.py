from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray_train.ddp_train import train_loop_per_worker, create_training_dataset


# Create a fixed dataset once per trial script run. If you want to tune dataset
# size, move this into the param_space or build inside a small factory.
train_dataset = create_training_dataset(num_samples=10240, input_dim=32, num_classes=2)


# Base trainer; Tune will clone this object and apply nested param overrides.
base_trainer = TorchTrainer(
    train_loop_per_worker,
    train_loop_config={
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": 5,
        "verbose": False,
    },
    datasets={"train": train_dataset},
    scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
)


# Tune on nested fields in place (minimal boilerplate).
param_space = {
    "train_loop_config": {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        # Keep epochs/verbose fixed for a quick demo; can also be tuned.
    },
    # Expose distributed resources if desired (kept fixed for minimal example):
    # "scaling_config": {"num_workers": tune.choice([1, 2, 4]), "use_gpu": False},
}


def main() -> None:
    # Quick start:
    #   python ./ray_train/ddp_tune.py
    # To target a running cluster, set env before running:
    #   RAY_ADDRESS="ray://<head>:10001" python ./ray_train/ddp_tune.py
    tuner = tune.Tuner(
        base_trainer,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=8,
            scheduler=ASHAScheduler(time_attr="epoch", max_t=5, grace_period=2),
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="loss", mode="min")
    print("Best metrics:", best.metrics)


if __name__ == "__main__":
    # For cluster usage, you can set RAY_ADDRESS env var instead of calling ray.init here.
    main()

# Commands to run:
# Basic run:
# python -m ray_train.ddp_tune
#
# To target a running cluster, set env before running:
# RAY_ADDRESS="ray://<head>:10001" python -m ray_train.ddp_tune

