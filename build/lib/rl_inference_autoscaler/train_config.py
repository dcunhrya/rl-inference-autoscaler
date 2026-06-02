"""Shared PPO / DQN RLlib configuration for local Ray and Modal training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_inference_autoscaler.autoscaler_env import AutoscalerEnv


@dataclass
class TrainingSettings:
    """Hyperparameters and paths used by Ray and Modal PPO entrypoints."""

    num_iterations: int = 50
    num_env_runners: int = 4
    num_envs_per_runner: int = 1
    train_batch_size: int = 4000
    lr: float = 3e-4
    gamma: float = 0.99
    clip_param: float = 0.2
    entropy_coeff: float = 0.01
    checkpoint_dir: str = "checkpoints/ppo"
    experiment_name: str = "autoscaler-ppo"
    mlflow_tracking_uri: str | None = None
    env_config: dict[str, Any] | None = None
    stop_reward_mean: float | None = None


@dataclass
class DQNSettings:
    """Hyperparameters and paths used by Ray and Modal DQN entrypoints."""

    num_iterations: int = 50
    num_env_runners: int = 4
    num_envs_per_runner: int = 1
    train_batch_size: int = 32
    lr: float = 5e-4
    gamma: float = 0.99
    replay_buffer_capacity: int = 50_000
    learning_starts: int = 1_000
    target_update_freq: int = 500
    double_q: bool = True
    dueling: bool = True
    n_step: int = 1
    epsilon_schedule: list[tuple[int, float]] | None = None
    checkpoint_dir: str = "checkpoints/dqn"
    experiment_name: str = "autoscaler-dqn"
    mlflow_tracking_uri: str | None = None
    env_config: dict[str, Any] | None = None
    stop_reward_mean: float | None = None


def env_config_dict(settings: TrainingSettings | DQNSettings) -> dict[str, Any]:
    return dict(settings.env_config or {})


def _ensure_checkpoint_dir(checkpoint_dir: str) -> Path:
    checkpoint_path = Path(checkpoint_dir).resolve()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def build_ppo_config(settings: TrainingSettings):
    """
    Build a Ray RLlib PPOConfig for AutoscalerEnv.

    Requires optional dependency: ``uv sync --extra train``.
    """
    from ray.rllib.algorithms.ppo import PPOConfig

    env_cfg = env_config_dict(settings)
    checkpoint_path = _ensure_checkpoint_dir(settings.checkpoint_dir)

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


def _dqn_replay_buffer_config(settings: DQNSettings) -> dict[str, Any]:
    """Episode replay buffer for DQN on RLlib's new API stack."""
    return {
        "type": "PrioritizedEpisodeReplayBuffer",
        "capacity": settings.replay_buffer_capacity,
        "alpha": 0.6,
        "beta": 0.4,
    }


def build_dqn_config(settings: DQNSettings):
    """
    Build a Ray RLlib DQNConfig for AutoscalerEnv.

    DQN uses RLlib's new API stack (PrioritizedEpisodeReplayBuffer). The old
    stack rejects episode buffers and hits a Ray bug with MultiAgentPrioritizedReplayBuffer.

    Requires optional dependency: ``uv sync --extra train``.
    """
    from ray.rllib.algorithms.dqn import DQNConfig

    env_cfg = env_config_dict(settings)
    checkpoint_path = _ensure_checkpoint_dir(settings.checkpoint_dir)
    epsilon = settings.epsilon_schedule or [(0, 1.0), (50_000, 0.05)]

    config = (
        DQNConfig()
        .environment(env=AutoscalerEnv, env_config=env_cfg)
        .env_runners(
            num_env_runners=settings.num_env_runners,
            num_envs_per_env_runner=settings.num_envs_per_runner,
        )
        .training(
            lr=settings.lr,
            gamma=settings.gamma,
            train_batch_size_per_learner=settings.train_batch_size,
            double_q=settings.double_q,
            dueling=settings.dueling,
            n_step=settings.n_step,
            num_steps_sampled_before_learning_starts=settings.learning_starts,
            target_network_update_freq=settings.target_update_freq,
            epsilon=epsilon,
            replay_buffer_config=_dqn_replay_buffer_config(settings),
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }
        )
        .evaluation(evaluation_interval=10, evaluation_duration=5)
        .checkpointing(export_native_model_files=True)
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
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


def validate_dqn_config(settings: DQNSettings | None = None) -> dict[str, Any]:
    """Smoke-check that RLlib DQN config builds and validates without starting Ray."""
    settings = settings or DQNSettings(num_iterations=1, num_env_runners=1)
    config, checkpoint_path = build_dqn_config(settings)
    config.validate()
    return {
        "algorithm": "DQN",
        "checkpoint_dir": str(checkpoint_path),
        "num_env_runners": settings.num_env_runners,
        "env_class": AutoscalerEnv.__name__,
    }
