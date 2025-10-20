import os

from ray import tune
from ray.air import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger import TBXLoggerCallback
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray_train.ddp_train import train_loop_per_worker, create_training_dataset


if __name__ == "__main__":
    # Build dataset and base trainer in the driver process.
    train_dataset = create_training_dataset(num_samples=10240, input_dim=32, num_classes=2)

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
        },
    }

    # ASHAScheduler: Asynchronous successive halving scheduler.
    scheduler = ASHAScheduler(time_attr="epoch", max_t=5, grace_period=2)

    # Configure TensorBoard logging for all trials.
    results_dir = os.path.expanduser(os.getenv("RAY_RESULTS_DIR", "~/ray_results"))

    tuner = tune.Tuner(
        base_trainer,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=8,
            scheduler=scheduler,
        ),
        run_config=RunConfig(
            name="ddp_tune_tb",
            storage_path=results_dir,
            callbacks=[TBXLoggerCallback()],
            log_to_file=True,
        ),
    )

    results = tuner.fit()

    best = results.get_best_result(metric="loss", mode="min")
    print("Best metrics:", best.metrics)

    # Commands to run:
    # Basic local run with 2 workers (CPU):
    #   python -m ray_train.ddp_tune_wtensorboard
    #
    # Launch TensorBoard (after or during run) to view per-trial metrics:
    #   tensorboard --logdir ~/ray_results/ddp_tune_tb
    #
    # Notes:
    # - Customize the result root by exporting RAY_RESULTS_DIR to override ~/ray_results.
    # - Metrics reported via ray.train.report in train_loop_per_worker (e.g., loss/epoch/step)
    #   are recorded by the TensorBoard logger per trial.


