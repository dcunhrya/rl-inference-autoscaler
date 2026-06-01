"""Shared PPO / RLlib configuration for local Ray and Modal training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_inference_autoscaler.autoscaler_env import AutoscalerEnv


@dataclass
class TrainingSettings:
    """Hyperparameters and paths used by Ray and Modal entrypoints."""

    num_iterations: int = 50
    num_env_runners: int = 4
    num_envs_per_runner: int = 1
    train_batch_size: int = 4000
    lr: float = 3e-4
    gamma: float = 0.99
    clip_param: float = 0.2
    entropy_coeff: float = 0.01
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "autoscaler-ppo"
    mlflow_tracking_uri: str | None = None
    env_config: dict[str, Any] | None = None
    stop_reward_mean: float | None = None


def env_config_dict(settings: TrainingSettings) -> dict[str, Any]:
    return dict(settings.env_config or {})


def build_ppo_config(settings: TrainingSettings):
    """
    Build a Ray RLlib PPOConfig for AutoscalerEnv.

    Requires optional dependency: ``uv sync --extra train``.
    """
    from ray.rllib.algorithms.ppo import PPOConfig

    env_cfg = env_config_dict(settings)
    checkpoint_path = Path(settings.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    config = (
        PPOConfig()
        .environment(env=AutoscalerEnv, env_config=env_cfg)
        .env_runners(
            num_env_runners=settings.num_env_runners,
            num_envs_per_env_runner=settings.num_envs_per_runner,
        )
        .training(
            lr=settings.lr,
            gamma=settings.gamma,
            train_batch_size_per_learner=settings.train_batch_size,
            clip_param=settings.clip_param,
            entropy_coeff=settings.entropy_coeff,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
        )
        .evaluation(evaluation_interval=10, evaluation_duration=5)
        .checkpointing(export_native_model_files=True)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )
    return config, checkpoint_path


def validate_ppo_config(settings: TrainingSettings | None = None) -> dict[str, Any]:
    """Smoke-check that RLlib config builds without starting Ray."""
    settings = settings or TrainingSettings(num_iterations=1, num_env_runners=1)
    _config, checkpoint_path = build_ppo_config(settings)
    return {
        "algorithm": "PPO",
        "checkpoint_dir": str(checkpoint_path),
        "num_env_runners": settings.num_env_runners,
        "env_class": AutoscalerEnv.__name__,
    }
