import os
import warnings
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env


def setup_warning_suppression():
    """Suppress Gymnasium deprecation warnings in main and worker processes."""
    # Set environment variable to propagate suppression to Ray workers.
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:gymnasium"
    
    # Suppress warnings in the main process.
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.vector")
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.core")
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs")


def make_cartpole_env(env_ctx):
    """Create CartPole env with Gymnasium warnings suppressed."""
    import warnings as worker_warnings
    worker_warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
    
    import gymnasium as gym
    from gymnasium import logger as gym_logger
    gym_logger.set_level(gym_logger.ERROR)
    
    return gym.make("CartPole-v1")


class ResultMetrics:
    """Unified accessor for RLlib result metrics.

    get(name, split=None) supports:
      - name: "return_mean" with split in {"train", "eval"}
      - name: "timesteps"
      - name: "sps"
      - name: any explicit key path present in result (e.g., "infos/custom")
    """

    _RETURN_TRAIN_KEYS = [
        "env_runners/episode_return_mean",
        "episode_return_mean",
        "episode_reward_mean",
    ]
    _RETURN_EVAL_KEYS = [
        "evaluation/episode_return_mean",
        "evaluation/episode_reward_mean",
        "evaluation/env_runners/episode_return_mean",
    ]
    _TIMESTEPS_KEYS = [
        "num_env_steps_sampled",
        "env_runners/num_env_steps_sampled",
        "timesteps_total",
        "num_steps_sampled",
    ]
    _SPS_KEYS = [
        "env_runners/samples_per_second",
        "samples_per_second",
        "sample_throughput",
    ]
    _TIME_ITER_KEYS = [
        "time_this_iter_s",
        "env_runners/time_this_iter_s",
    ]
    _SAMPLES_ITER_KEYS = [
        "num_env_steps_sampled_this_iter",
        "env_runners/num_env_steps_sampled_this_iter",
        "samples_this_iter",
    ]

    def __init__(self, result: dict):
        self._result = result
        self._flat = self._flatten_result(result)

    def get(self, name: str, split: str | None = None):
        if name == "return_mean":
            if split == "eval":
                return self._first_present(self._flat, self._RETURN_EVAL_KEYS)
            # default to train
            return self._first_present(self._flat, self._RETURN_TRAIN_KEYS)
        if name == "timesteps":
            return self._first_present(self._flat, self._TIMESTEPS_KEYS)
        if name in ("sps", "samples_per_second"):
            return self._first_present(self._flat, self._SPS_KEYS)
        if name in ("time_iter", "time_this_iter_s"):
            return self._first_present(self._flat, self._TIME_ITER_KEYS)
        if name in ("samples_this_iter", "steps_this_iter"):
            return self._first_present(self._flat, self._SAMPLES_ITER_KEYS)
        # Fallback: treat as direct key path or alias
        return self._flat.get(name)

    @staticmethod
    def _flatten_result(nested: dict, prefix: str = "") -> dict:
        flat = {}
        for k, v in nested.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            if isinstance(v, dict):
                flat.update(ResultMetrics._flatten_result(v, key))
            else:
                flat[key] = v
        return flat

    @staticmethod
    def _first_present(flat: dict, candidates: list):
        for c in candidates:
            if c in flat and flat[c] is not None:
                return flat[c]
        return None


if __name__ == "__main__":
    """PPO with CartPole-v1 environment using Ray Train. Should converge in <50 iterations with an average eval return of 475+ (max score is 500)
    """
    setup_warning_suppression()
    ray.init()
    
    register_env("cartpole_no_warnings", make_cartpole_env)
    num_gpus = 1 if torch.cuda.is_available() else 0

    algo = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .environment("cartpole_no_warnings")
        .framework("torch")
        .env_runners(num_env_runners=2, num_envs_per_env_runner=1)
        .evaluation(
            evaluation_interval=1,
            evaluation_num_env_runners=1,
            evaluation_parallel_to_training=False,
        )
        .debugging(logger_config={"type": "ray.tune.logger.NoopLogger", "logdir": None})
        .resources(num_gpus=num_gpus)
    ).build()

    for i in range(50):
        result = algo.train()
        metrics = ResultMetrics(result)
        train_ret = metrics.get("return_mean", split="train")
        eval_ret = metrics.get("return_mean", split="eval")
        total_ts = metrics.get("timesteps")
        throughput = metrics.get("sps") or (
            (metrics.get("samples_this_iter") or 0)
            / (metrics.get("time_iter") or 1)
        )
        print(
            {
                "iter": result.get("training_iteration", i),
                "train_return": train_ret,
                "eval_return": eval_ret,
                "timesteps": total_ts,
                "sps": throughput,
            }
        )


# Command to run:
# python ./ray-train/ppo_cartpole.py