"""Ray RLlib PPO training entrypoint (local or inside Modal worker)."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from rl_inference_autoscaler import register_env
from rl_inference_autoscaler.train_common import (
    extract_ppo_metrics,
    finish_mlflow_run,
    init_ray_local,
    log_metrics,
    ppo_mlflow_params,
    setup_mlflow,
    should_stop_training,
)
from rl_inference_autoscaler.train_config import TrainingSettings, build_ppo_config

logger = logging.getLogger(__name__)


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
        _config, checkpoint_path = build_ppo_config(settings)
        return {
            "dry_run": True,
            "algorithm": "PPO",
            "checkpoint_dir": str(checkpoint_path),
            "config_built": True,
        }

    import ray

    if init_ray:
        init_ray_local()

    config, checkpoint_path = build_ppo_config(settings)
    algo = config.build_algo()
    mlflow = setup_mlflow(settings, ppo_mlflow_params(settings))
    last_result: dict[str, Any] = {}

    try:
        for i in range(settings.num_iterations):
            last_result = algo.train()
            log_metrics(mlflow, extract_ppo_metrics(last_result, i), i)
            if should_stop_training(last_result, settings.stop_reward_mean):
                logger.info(
                    "stop: episode_return_mean >= %s", settings.stop_reward_mean
                )
                break

        checkpoint = algo.save(str(checkpoint_path / "final"))
        summary = {
            "algorithm": "PPO",
            "checkpoint": str(checkpoint),
            "checkpoint_dir": str(checkpoint_path),
            "iterations": settings.num_iterations,
            "last_result_keys": list(last_result.keys()),
        }
        finish_mlflow_run(mlflow, summary)
        return summary
    finally:
        algo.stop()
        if shutdown_ray and ray.is_initialized():
            ray.shutdown()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train autoscaler PPO with Ray RLlib")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--num-env-runners", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/ppo")
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
    parser.add_argument(
        "--reward-mode",
        choices=["balanced", "cost_sensitive", "latency_sensitive"],
        default=None,
        help="Apply reward preset (R8)",
    )
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--churn-penalty-delta", type=float, default=None)
    parser.add_argument("--pending-penalty-eta", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    num_env_runners = 0 if args.local else args.num_env_runners
    env_config: dict = {"traffic_mode": args.traffic_mode}
    if args.reward_mode:
        env_config["reward_mode"] = args.reward_mode
    if args.max_steps_per_episode is not None:
        env_config["max_steps_per_episode"] = args.max_steps_per_episode
    if args.churn_penalty_delta is not None:
        env_config["churn_penalty_delta"] = args.churn_penalty_delta
    if args.pending_penalty_eta is not None:
        env_config["pending_penalty_eta"] = args.pending_penalty_eta
    settings = TrainingSettings(
        num_iterations=args.iterations,
        num_env_runners=num_env_runners,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        env_config=env_config,
        lr=args.lr if args.lr is not None else TrainingSettings().lr,
    )
    summary = run_training(settings, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
