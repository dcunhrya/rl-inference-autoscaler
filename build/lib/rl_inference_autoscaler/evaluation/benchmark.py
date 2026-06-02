"""Evaluate RLlib checkpoints and load MLflow training curves."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rl_inference_autoscaler.env import AutoscalerEnv
from rl_inference_autoscaler.policies.baselines import (
    PolicyFn,
    _step_penalties,
    do_nothing_policy,
    evaluate_policy,
    fixed_replica_policy,
    greedy_policy,
    ideal_replica_count,
    target_utilization_policy,
)
from rl_inference_autoscaler.training.runtime import init_ray_local

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
    """Return the first existing checkpoint path (absolute), or None."""
    for candidate in (primary, *fallbacks):
        path = Path(candidate).resolve()
        if path.exists():
            return path
    return None


def _checkpoint_version(path: Path) -> tuple[float, dict[str, Any]]:
    meta_path = path / "rllib_checkpoint.json"
    if not meta_path.is_file():
        return 1.0, {}
    meta = json.loads(meta_path.read_text())
    version = float(str(meta.get("checkpoint_version", "1.0")).split(".")[0])
    return version, meta


def _rl_module_checkpoint_path(checkpoint_path: Path) -> Path:
    return (
        checkpoint_path
        / "learner_group"
        / "learner"
        / "rl_module"
        / "default_policy"
    )


def _action_from_rl_module_output(output: dict[str, Any]) -> int:
    import torch

    action = output["actions"]
    if isinstance(action, torch.Tensor):
        if action.ndim == 0:
            return int(action.item())
        return int(action[0].item())
    if isinstance(action, (list, tuple, np.ndarray)):
        return int(action[0])
    return int(action)


def _load_new_stack_action_fn(checkpoint_path: Path) -> Callable[[np.ndarray], int]:
    """Load a new-API-stack RLModule for local inference (no Ray workers)."""
    import torch
    from ray.rllib.core.rl_module.rl_module import RLModule

    module_path = _rl_module_checkpoint_path(checkpoint_path)
    if not module_path.is_dir():
        raise FileNotFoundError(f"RLModule checkpoint not found: {module_path}")

    module = RLModule.from_checkpoint(str(module_path))

    def compute_action(obs: np.ndarray) -> int:
        batch = {"obs": torch.tensor(np.asarray([obs], dtype=np.float32))}
        with torch.no_grad():
            output = module.forward_inference(batch)
        return _action_from_rl_module_output(output)

    return compute_action


def _load_old_stack_action_fn(checkpoint_path: Path) -> Callable[[np.ndarray], int]:
    """Load an old-API-stack Algorithm checkpoint via Ray."""
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm

    from rl_inference_autoscaler import register_env

    register_env()
    init_ray_local()
    algo = Algorithm.from_checkpoint(str(checkpoint_path))

    def compute_action(obs: np.ndarray) -> int:
        action = algo.compute_single_action(
            obs,
            explore=False,
            prev_action=None,
            prev_reward=None,
        )
        if isinstance(action, tuple):
            action = action[0]
        return int(action)

    compute_action._algo = algo  # type: ignore[attr-defined]
    compute_action._uses_ray = True  # type: ignore[attr-defined]
    return compute_action


def _rollout_with_action_fn(
    action_fn: Callable[[np.ndarray], int],
    env: AutoscalerEnv,
    *,
    episodes: int,
    seed: int,
    record_trajectory_episode: int | None,
) -> dict[str, Any]:
    returns: list[float] = []
    costs: list[float] = []
    latencies: list[float] = []
    churns: list[float] = []
    pendings: list[float] = []
    trajectory: dict[str, list] | None = None

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        ep_cost = 0.0
        ep_latency = 0.0
        ep_churn = 0.0
        ep_pending = 0.0
        truncated = False
        record = record_trajectory_episode is not None and ep == record_trajectory_episode
        if record:
            trajectory = {
                "rps": [],
                "active_replicas": [],
                "ideal_replicas": [],
                "actions": [],
                "cost_penalty": [],
                "latency_penalty": [],
            }

        while not truncated:
            action = action_fn(obs)
            obs, reward, _term, truncated, info = env.step(action)
            total += reward
            step_cost = float(info.get("cost_penalty", _step_penalties(env, info)[0]))
            step_lat = float(info.get("latency_penalty", _step_penalties(env, info)[1]))
            ep_cost += step_cost
            ep_latency += step_lat
            ep_churn += float(info.get("churn_penalty", 0.0))
            ep_pending += float(info.get("pending_penalty", 0.0))
            if record and trajectory is not None:
                trajectory["rps"].append(float(obs[0]))
                trajectory["active_replicas"].append(float(info["active_replicas"]))
                trajectory["ideal_replicas"].append(
                    ideal_replica_count(float(obs[0]), env)
                )
                trajectory["actions"].append(action)
                trajectory["cost_penalty"].append(step_cost)
                trajectory["latency_penalty"].append(step_lat)

        returns.append(total)
        costs.append(ep_cost)
        latencies.append(ep_latency)
        churns.append(ep_churn)
        pendings.append(ep_pending)

    result: dict[str, Any] = {
        "episode_returns": returns,
        "episode_return_mean": float(np.mean(returns)),
        "episode_return_std": float(np.std(returns)),
        "mean_cost_penalty": float(np.mean(costs)),
        "mean_latency_penalty": float(np.mean(latencies)),
        "mean_churn_penalty": float(np.mean(churns)),
        "mean_pending_penalty": float(np.mean(pendings)),
        "episodes": episodes,
    }
    if trajectory is not None:
        result["trajectory"] = trajectory
    return result


def evaluate_rllib_checkpoint(
    checkpoint_path: str | Path,
    env: AutoscalerEnv | None = None,
    *,
    episodes: int = 20,
    seed: int = 0,
    record_trajectory_episode: int | None = None,
) -> dict[str, Any]:
    """Roll out a saved Ray RLlib policy (new API stack RLModule, or legacy Algorithm)."""
    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    env = env or AutoscalerEnv()
    version, _meta = _checkpoint_version(path)
    uses_ray = False
    algo = None

    module_path = _rl_module_checkpoint_path(path)
    try:
        if module_path.is_dir():
            action_fn = _load_new_stack_action_fn(path)
        elif version >= 2.0:
            action_fn = _load_new_stack_action_fn(path)
        else:
            action_fn = _load_old_stack_action_fn(path)
            uses_ray = True
            algo = getattr(action_fn, "_algo", None)

        return _rollout_with_action_fn(
            action_fn,
            env,
            episodes=episodes,
            seed=seed,
            record_trajectory_episode=record_trajectory_episode,
        )
    finally:
        if algo is not None:
            algo.stop()
        if uses_ray:
            import ray

            if ray.is_initialized():
                ray.shutdown()


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
    prefer_iterations: int | None = 100,
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
        ("target_utilization", target_utilization_policy, None),
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
