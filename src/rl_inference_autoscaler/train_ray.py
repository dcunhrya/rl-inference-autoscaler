"""Ray RLlib training entrypoint (local or inside Modal worker)."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from rl_inference_autoscaler import register_env
from rl_inference_autoscaler.train_config import TrainingSettings, build_ppo_config

logger = logging.getLogger(__name__)


def _init_ray_local(*, num_cpus: int | None = None) -> None:
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


def _setup_mlflow(settings: TrainingSettings):
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed; metrics only in Ray result dict")
        return None
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    mlflow.start_run(run_name=settings.experiment_name)
    mlflow.log_params(
        {
            "num_iterations": settings.num_iterations,
            "num_env_runners": settings.num_env_runners,
            "lr": settings.lr,
            "gamma": settings.gamma,
            "clip_param": settings.clip_param,
        }
    )
    return mlflow


def _log_result(mlflow, result: dict[str, Any], iteration: int) -> None:
    env_runners = result.get("env_runners") or {}
    learners = result.get("learners") or {}
    metrics = {
        "iteration": iteration,
        "episode_return_mean": env_runners.get("episode_return_mean"),
        "episode_len_mean": env_runners.get("episode_len_mean"),
        "policy_entropy": learners.get("entropy"),
    }
    logger.info("train iter %s: %s", iteration, metrics)
    if mlflow is not None:
        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, float(value), step=iteration)


def run_training(
    settings: TrainingSettings | None = None,
    *,
    init_ray: bool = True,
    shutdown_ray: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run PPO training with Ray RLlib.

    Parameters
    ----------
    dry_run:
        If True, only build config and return metadata (no ``algo.train()``).
    """
    settings = settings or TrainingSettings()
    register_env()

    if dry_run:
        config, checkpoint_path = build_ppo_config(settings)
        return {
            "dry_run": True,
            "checkpoint_dir": str(checkpoint_path),
            "config_built": True,
        }

    import ray

    if init_ray:
        _init_ray_local()

    config, checkpoint_path = build_ppo_config(settings)
    algo = config.build_algo()
    mlflow = _setup_mlflow(settings)
    last_result: dict[str, Any] = {}

    try:
        for i in range(settings.num_iterations):
            last_result = algo.train()
            _log_result(mlflow, last_result, i)
            env_runners = last_result.get("env_runners") or {}
            mean_return = env_runners.get("episode_return_mean")
            if (
                settings.stop_reward_mean is not None
                and mean_return is not None
                and mean_return >= settings.stop_reward_mean
            ):
                logger.info("stop: episode_return_mean >= %s", settings.stop_reward_mean)
                break

        checkpoint = algo.save(str(checkpoint_path / "final"))
        summary = {
            "checkpoint": str(checkpoint),
            "checkpoint_dir": str(checkpoint_path),
            "iterations": settings.num_iterations,
            "last_result_keys": list(last_result.keys()),
        }
        if mlflow is not None:
            import mlflow

            mlflow.log_dict(summary, "training_summary.json")
            mlflow.end_run()
        return summary
    finally:
        algo.stop()
        if shutdown_ray and ray.is_initialized():
            ray.shutdown()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train autoscaler PPO with Ray RLlib")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--num-env-runners", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--experiment-name", type=str, default="autoscaler-ppo")
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="e.g. file:./mlruns or http://localhost:5000",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config only; do not call ray.init or train",
    )
    parser.add_argument(
        "--traffic-mode",
        choices=["auto", "synthetic", "csv"],
        default="auto",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use num_env_runners=0 (single-process rollouts; best for Mac debugging)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    num_env_runners = 0 if args.local else args.num_env_runners
    settings = TrainingSettings(
        num_iterations=args.iterations,
        num_env_runners=num_env_runners,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        env_config={"traffic_mode": args.traffic_mode},
    )
    summary = run_training(settings, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
