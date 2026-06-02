"""Heuristic and greedy baselines for comparing against the learned PPO policy."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from rl_inference_autoscaler.autoscaler_env import AutoscalerEnv

PolicyFn = Callable[[np.ndarray, AutoscalerEnv], int]


def do_nothing_policy(state: np.ndarray, env: AutoscalerEnv) -> int:
    """Always hold (action 1)."""
    return 1


def fixed_replica_policy(state: np.ndarray, env: AutoscalerEnv) -> int:
    """Always hold; replica count is set via env ``initial_replicas`` on reset."""
    return 1


def target_utilization_policy(
    state: np.ndarray,
    env: AutoscalerEnv,
    *,
    target_util: float = 0.7,
) -> int:
    """Reactive scaler: scale up if utilization above target, down if well below."""
    _rps, util, replicas, _delta = state
    if util > target_util and replicas < env.max_replicas:
        return 2
    if util < target_util * 0.5 and replicas > 1:
        return 0
    return 1


def ideal_replica_count(rps: float, env: AutoscalerEnv) -> int:
    """Ground-truth capacity for current RPS (instant, no cold-start)."""
    needed = int(math.ceil(float(rps) / max(env.throughput_per_replica, 1e-6)))
    return int(np.clip(needed, 1, env.max_replicas))


def greedy_policy(state: np.ndarray, env: AutoscalerEnv) -> int:
    """
    Greedy scaler: knows next-step RPS and moves toward required capacity
    (ignores cold-start delay).
    """
    rps, _util, replicas, _delta = state
    next_rps = env.traffic.peek_next_rps(env._rng, float(rps))
    needed = ideal_replica_count(next_rps, env)
    effective = int(replicas) + len(env._pending_boots)
    if effective < needed:
        return 2
    if int(replicas) > needed:
        return 0
    return 1


# Backward-compatible alias
oracle_policy = greedy_policy


def _step_penalties(env: AutoscalerEnv, info: dict) -> tuple[float, float]:
    cost = env.cost_alpha * float(info["active_replicas"])
    latency = env.latency_beta * (
        float(info["overload_rps"])
        + float(info["dropped_requests"])
        + env.queue_penalty_gamma * float(info["queue_depth"])
    )
    return cost, latency


def evaluate_policy(
    policy: PolicyFn,
    env: AutoscalerEnv | None = None,
    *,
    episodes: int = 20,
    seed: int = 0,
    record_trajectory_episode: int | None = None,
) -> dict[str, float | list[float] | dict]:
    """Roll out a policy; return returns, cost/latency breakdown, optional trace."""
    env = env or AutoscalerEnv()
    returns: list[float] = []
    costs: list[float] = []
    latencies: list[float] = []
    trajectory: dict[str, list] | None = None

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
                "cost_penalty": [],
                "latency_penalty": [],
            }

        while not truncated:
            action = policy(obs, env)
            obs, reward, _term, truncated, info = env.step(action)
            total += reward
            step_cost = float(info.get("cost_penalty", _step_penalties(env, info)[0]))
            step_lat = float(info.get("latency_penalty", _step_penalties(env, info)[1]))
            ep_cost += step_cost
            ep_latency += step_lat
            if record and trajectory is not None:
                trajectory["rps"].append(float(obs[0]))
                trajectory["active_replicas"].append(float(info["active_replicas"]))
                trajectory["ideal_replicas"].append(
                    ideal_replica_count(float(obs[0]), env)
                )
                trajectory["actions"].append(int(action))
                trajectory["cost_penalty"].append(step_cost)
                trajectory["latency_penalty"].append(step_lat)

        returns.append(total)
        costs.append(ep_cost)
        latencies.append(ep_latency)

    result: dict[str, float | list[float] | dict] = {
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


def evaluate_baseline(
    env: AutoscalerEnv | None = None,
    *,
    episodes: int = 3,
    seed: int = 0,
) -> dict[str, float]:
    """Backward-compatible wrapper for target-utilization heuristic."""
    result = evaluate_policy(
        target_utilization_policy,
        env=env,
        episodes=episodes,
        seed=seed,
    )
    return {
        "episode_return_mean": result["episode_return_mean"],
        "episode_return_std": result["episode_return_std"],
        "episodes": episodes,
    }
