"""Shared Ray / MLflow helpers for PPO and DQN training entrypoints."""

from __future__ import annotations

import logging
import os
from typing import Any

from rl_inference_autoscaler.train_config import DQNSettings, TrainingSettings

logger = logging.getLogger(__name__)


def init_ray_local(*, num_cpus: int | None = 4) -> None:
    """
    Initialize Ray for local training.

    Disables Ray's ``uv run`` worker replication (Ray 2.55+). That path packages
    the repo into ``/tmp/ray/...`` and runs ``uv sync`` without ``--extra train``,
    so workers install only base deps and fail with ``No module named 'ray'``.
    """
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

    import ray

    if ray.is_initialized():
        return

    init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
    if num_cpus is not None:
        init_kwargs["num_cpus"] = num_cpus

    ray.init(**init_kwargs)


def setup_mlflow(
    settings: TrainingSettings | DQNSettings,
    params: dict[str, Any],
):
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed; metrics only in Ray result dict")
        return None
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    mlflow.start_run(run_name=settings.experiment_name)
    mlflow.log_params(params)
    return mlflow


def log_metrics(mlflow, metrics: dict[str, Any], iteration: int) -> None:
    logger.info("train iter %s: %s", iteration, metrics)
    if mlflow is not None:
        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, float(value), step=iteration)


def finish_mlflow_run(mlflow, summary: dict[str, Any]) -> None:
    if mlflow is not None:
        import mlflow

        mlflow.log_dict(summary, "training_summary.json")
        mlflow.end_run()


def _env_mlflow_params(settings: TrainingSettings | DQNSettings) -> dict[str, Any]:
    cfg = dict(settings.env_config or {})
    keys = (
        "traffic_mode",
        "reward_mode",
        "cost_alpha",
        "latency_beta",
        "queue_penalty_gamma",
        "churn_penalty_delta",
        "pending_penalty_eta",
        "util_band_penalty_zeta",
    )
    return {k: cfg[k] for k in keys if k in cfg}


def ppo_mlflow_params(settings: TrainingSettings) -> dict[str, Any]:
    return {
        "algorithm": "PPO",
        "num_iterations": settings.num_iterations,
        "num_env_runners": settings.num_env_runners,
        "lr": settings.lr,
        "gamma": settings.gamma,
        "clip_param": settings.clip_param,
        **_env_mlflow_params(settings),
    }


def dqn_mlflow_params(settings: DQNSettings) -> dict[str, Any]:
    return {
        "algorithm": "DQN",
        "num_iterations": settings.num_iterations,
        "num_env_runners": settings.num_env_runners,
        "lr": settings.lr,
        "gamma": settings.gamma,
        "replay_buffer_capacity": settings.replay_buffer_capacity,
        "learning_starts": settings.learning_starts,
        "target_update_freq": settings.target_update_freq,
        "double_q": settings.double_q,
        "dueling": settings.dueling,
        **_env_mlflow_params(settings),
    }


def extract_ppo_metrics(result: dict[str, Any], iteration: int) -> dict[str, Any]:
    env_runners = result.get("env_runners") or {}
    learners = result.get("learners") or {}
    return {
        "iteration": iteration,
        "episode_return_mean": env_runners.get("episode_return_mean"),
        "episode_len_mean": env_runners.get("episode_len_mean"),
        "policy_entropy": learners.get("entropy"),
    }


def extract_dqn_metrics(result: dict[str, Any], iteration: int) -> dict[str, Any]:
    env_runners = result.get("env_runners") or {}
    learners = result.get("learners") or {}
    default_learner = learners.get("default_policy") or learners.get("default_policy_id") or {}
    if not default_learner and learners:
        default_learner = next(iter(learners.values()), {})

    metrics: dict[str, Any] = {
        "iteration": iteration,
        "episode_return_mean": env_runners.get("episode_return_mean"),
        "episode_len_mean": env_runners.get("episode_len_mean"),
        "num_env_steps_sampled": result.get("num_env_steps_sampled_lifetime")
        or env_runners.get("num_env_steps_sampled"),
    }

    for key in ("td_error", "exploration_epsilon", "mean_q"):
        value = default_learner.get(key) or learners.get(key)
        if value is not None:
            metrics[key] = value

    return metrics


def should_stop_training(
    result: dict[str, Any],
    stop_reward_mean: float | None,
) -> bool:
    if stop_reward_mean is None:
        return False
    env_runners = result.get("env_runners") or {}
    mean_return = env_runners.get("episode_return_mean")
    return mean_return is not None and mean_return >= stop_reward_mean
