"""Heuristic baselines for comparing against the learned policy."""

from __future__ import annotations

import numpy as np

from rl_inference_autoscaler.autoscaler_env import AutoscalerEnv


def target_utilization_policy(
    state: np.ndarray,
    *,
    target_util: float = 0.7,
    scale_up_action: int = 2,
    scale_down_action: int = 0,
    hold_action: int = 1,
) -> int:
    """Scale up if utilization above target, down if well below."""
    _rps, util, replicas, _delta = state
    if util > target_util and replicas < 20:
        return scale_up_action
    if util < target_util * 0.5 and replicas > 1:
        return scale_down_action
    return hold_action


def evaluate_baseline(
    env: AutoscalerEnv | None = None,
    *,
    episodes: int = 3,
    seed: int = 0,
) -> dict[str, float]:
    env = env or AutoscalerEnv()
    returns: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        truncated = False
        while not truncated:
            action = target_utilization_policy(obs)
            obs, reward, _term, truncated, _info = env.step(action)
            total += reward
        returns.append(total)
    return {
        "episode_return_mean": float(np.mean(returns)),
        "episode_return_std": float(np.std(returns)),
        "episodes": episodes,
    }
