"""Reward presets and env config resolution (R8)."""

from __future__ import annotations

from typing import Any

REWARD_MODE_PRESETS: dict[str, dict[str, float]] = {
    "balanced": {
        "cost_alpha": 0.1,
        "latency_beta": 1.0,
        "queue_penalty_gamma": 0.05,
    },
    "cost_sensitive": {
        "cost_alpha": 0.2,
        "latency_beta": 0.5,
        "queue_penalty_gamma": 0.05,
    },
    "latency_sensitive": {
        "cost_alpha": 0.05,
        "latency_beta": 2.0,
        "queue_penalty_gamma": 0.1,
    },
}


def apply_reward_mode(cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge preset hyperparameters when ``reward_mode`` is set (R8)."""
    out = dict(cfg)
    mode = out.get("reward_mode")
    if mode and mode in REWARD_MODE_PRESETS:
        for key, value in REWARD_MODE_PRESETS[mode].items():
            out.setdefault(key, value)
    return out
