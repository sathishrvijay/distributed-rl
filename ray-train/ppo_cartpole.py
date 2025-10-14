import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig


if __name__ == "__main__":
    ray.init()
    num_gpus = 1 if torch.cuda.is_available() else 0

    algo = (
        PPOConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .env_runners(num_env_runners=2)
        .resources(num_gpus=num_gpus)
    ).build()

    for i in range(10):
        result = algo.train()
        reward_mean = (
            result.get("episode_return_mean")
            or result.get("env_runners/episode_return_mean")
            or result.get("episode_reward_mean")
        )
        print({"iter": i, "reward_mean": reward_mean})


