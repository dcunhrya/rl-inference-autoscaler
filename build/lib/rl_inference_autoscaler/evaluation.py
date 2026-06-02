"""Evaluate RLlib checkpoints and load MLflow training curves."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from rl_inference_autoscaler.autoscaler_env import AutoscalerEnv
from rl_inference_autoscaler.baselines import (
    PolicyFn,
    _step_penalties,
    do_nothing_policy,
    evaluate_policy,
    fixed_replica_policy,
    greedy_policy,
    ideal_replica_count,
)
from rl_inference_autoscaler.train_common import init_ray_local

PPO_MLFLOW_METRICS = (
    "episode_return_mean",
    "episode_len_mean",
    "policy_entropy",
    "iteration",
)
DQN_MLFLOW_METRICS = (
    "episode_return_mean",
    "episode_len_mean",
    "td_error",
    "num_env_steps_sampled",
    "exploration_epsilon",
    "mean_q",
    "iteration",
)


def resolve_checkpoint_path(
    primary: str | Path,
    *fallbacks: str | Path,
) -> Path | None:
    """Return the first existing checkpoint path, or None."""
    for candidate in (primary, *fallbacks):
        path = Path(candidate)
        if path.exists():
            return path
    return None


def evaluate_rllib_checkpoint(
    checkpoint_path: str | Path,
    env: AutoscalerEnv | None = None,
    *,
    episodes: int = 20,
    seed: int = 0,
    record_trajectory_episode: int | None = None,
) -> dict[str, Any]:
    """Roll out a saved Ray RLlib policy (PPO or DQN)."""
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    init_ray_local()
    algo = Algorithm.from_checkpoint(str(path))
    env = env or AutoscalerEnv()
    returns: list[float] = []
    costs: list[float] = []
    latencies: list[float] = []
    trajectory: dict[str, list] | None = None

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            total = 0.0
            ep_cost = 0.0
            ep_latency = 0.0
            truncated = False
            record = record_trajectory_episode is not None and ep == record_trajectory_episode
            if record:
                trajectory = {
                    "rps": [],
                    "active_replicas": [],
                    "ideal_replicas": [],
                    "actions": [],
                }

            while not truncated:
                action = algo.compute_single_action(
                    obs,
                    explore=False,
                    prev_action=None,
                    prev_reward=None,
                )
                if isinstance(action, tuple):
                    action = action[0]
                action = int(action)
                obs, reward, _term, truncated, info = env.step(action)
                total += reward
                step_cost, step_lat = _step_penalties(env, info)
                ep_cost += step_cost
                ep_latency += step_lat
                if record and trajectory is not None:
                    trajectory["rps"].append(float(obs[0]))
                    trajectory["active_replicas"].append(float(info["active_replicas"]))
                    trajectory["ideal_replicas"].append(
                        ideal_replica_count(float(obs[0]), env)
                    )
                    trajectory["actions"].append(action)

            returns.append(total)
            costs.append(ep_cost)
            latencies.append(ep_latency)
    finally:
        algo.stop()
        if ray.is_initialized():
            ray.shutdown()

    result: dict[str, Any] = {
        "episode_returns": returns,
        "episode_return_mean": float(np.mean(returns)),
        "episode_return_std": float(np.std(returns)),
        "mean_cost_penalty": float(np.mean(costs)),
        "mean_latency_penalty": float(np.mean(latencies)),
        "episodes": episodes,
    }
    if trajectory is not None:
        result["trajectory"] = trajectory
    return result


def evaluate_ppo_checkpoint(
    checkpoint_path: str | Path,
    env: AutoscalerEnv | None = None,
    *,
    episodes: int = 20,
    seed: int = 0,
    record_trajectory_episode: int | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for ``evaluate_rllib_checkpoint``."""
    return evaluate_rllib_checkpoint(
        checkpoint_path,
        env,
        episodes=episodes,
        seed=seed,
        record_trajectory_episode=record_trajectory_episode,
    )


def load_mlflow_training_metrics(
    db_path: str | Path = "mlflow.db",
    *,
    experiment_name: str = "autoscaler-ppo",
    prefer_iterations: int | None = 50,
    metric_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load per-iteration metrics from the MLflow SQLite store."""
    db_path = Path(db_path)
    if not db_path.is_file():
        return {"run_id": None, "metrics": {}}

    if metric_keys is None:
        if experiment_name.startswith("autoscaler-dqn"):
            metric_keys = DQN_MLFLOW_METRICS
        else:
            metric_keys = PPO_MLFLOW_METRICS

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT experiment_id FROM experiments WHERE name = ?",
            (experiment_name,),
        )
        row = cur.fetchone()
        if row is None:
            return {"run_id": None, "metrics": {}}
        exp_id = row[0]

        cur = conn.execute(
            """
            SELECT r.run_uuid, r.start_time
            FROM runs r
            WHERE r.experiment_id = ? AND r.status = 'FINISHED'
            ORDER BY r.start_time DESC
            """,
            (exp_id,),
        )
        runs = cur.fetchall()
        if not runs:
            return {"run_id": None, "metrics": {}}

        run_id = runs[0][0]
        if prefer_iterations is not None:
            for rid, _ in runs:
                cur = conn.execute(
                    "SELECT value FROM params WHERE run_uuid = ? AND key = 'num_iterations'",
                    (rid,),
                )
                prow = cur.fetchone()
                if prow and int(float(prow[0])) == prefer_iterations:
                    run_id = rid
                    break

        metrics: dict[str, list[tuple[int, float]]] = {}
        for key in metric_keys:
            cur = conn.execute(
                """
                SELECT step, value FROM metrics
                WHERE run_uuid = ? AND key = ?
                ORDER BY step
                """,
                (run_id, key),
            )
            rows = [(int(s), float(v)) for s, v in cur.fetchall()]
            if rows:
                metrics[key] = rows

        return {"run_id": run_id, "metrics": metrics, "experiment_name": experiment_name}
    finally:
        conn.close()


def run_benchmark_suite(
    *,
    ppo_checkpoint_path: str | Path = "checkpoints/ppo/final",
    dqn_checkpoint_path: str | Path = "checkpoints/dqn/final",
    include_ppo: bool = True,
    include_dqn: bool = True,
    env_config: dict[str, Any] | None = None,
    episodes: int = 20,
    seed: int = 0,
    fixed_replicas: int = 4,
    mlflow_db: str | Path = "mlflow.db",
    trajectory_episode: int = 0,
) -> dict[str, Any]:
    """Evaluate trained policies and baselines under identical env settings."""
    base_cfg = dict(env_config or {})
    base_cfg.setdefault("traffic_mode", "auto")

    def _env(extra: dict[str, Any] | None = None) -> AutoscalerEnv:
        cfg = {**base_cfg, **(extra or {})}
        return AutoscalerEnv(config=cfg)

    results: dict[str, Any] = {
        "episodes": episodes,
        "seed": seed,
        "env_config": base_cfg,
        "policies": {},
        "trajectory_episode": trajectory_episode,
    }

    policies: list[tuple[str, PolicyFn, dict[str, Any] | None]] = [
        ("do_nothing", do_nothing_policy, None),
        ("fixed_replica", fixed_replica_policy, {"initial_replicas": float(fixed_replicas)}),
        ("greedy", greedy_policy, None),
    ]

    for name, fn, extra in policies:
        record_ep = trajectory_episode if name == "greedy" else None
        results["policies"][name] = evaluate_policy(
            fn,
            _env(extra),
            episodes=episodes,
            seed=seed,
            record_trajectory_episode=record_ep,
        )

    if include_ppo:
        ppo_ckpt = resolve_checkpoint_path(
            ppo_checkpoint_path,
            "checkpoints/final",
        )
        if ppo_ckpt is not None:
            results["policies"]["ppo"] = evaluate_rllib_checkpoint(
                ppo_ckpt,
                _env(),
                episodes=episodes,
                seed=seed,
                record_trajectory_episode=trajectory_episode,
            )
        else:
            results["policies"]["ppo"] = {
                "error": f"missing checkpoint: {ppo_checkpoint_path} (and fallbacks)"
            }

    if include_dqn:
        dqn_ckpt = resolve_checkpoint_path(dqn_checkpoint_path)
        if dqn_ckpt is not None:
            results["policies"]["dqn"] = evaluate_rllib_checkpoint(
                dqn_ckpt,
                _env(),
                episodes=episodes,
                seed=seed,
                record_trajectory_episode=trajectory_episode,
            )
        else:
            results["policies"]["dqn"] = {
                "error": f"missing checkpoint: {dqn_checkpoint_path}"
            }

    results["mlflow"] = {
        "ppo": load_mlflow_training_metrics(
            mlflow_db,
            experiment_name="autoscaler-ppo",
        ),
        "dqn": load_mlflow_training_metrics(
            mlflow_db,
            experiment_name="autoscaler-dqn",
        ),
    }
    return results
