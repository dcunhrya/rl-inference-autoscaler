"""RL inference autoscaler: Gymnasium simulator and Ray/Modal training."""

from rl_inference_autoscaler.env import AutoscalerEnv

__all__ = ["AutoscalerEnv", "register_env"]


def register_env() -> None:
    """Register Autoscaler-v0 with Gymnasium (idempotent)."""
    try:
        import gymnasium as gym
    except ImportError:
        return
    try:
        gym.spec("Autoscaler-v0")
        return
    except Exception:
        pass
    gym.register(
        id="Autoscaler-v0",
        entry_point="rl_inference_autoscaler.env.autoscaler:AutoscalerEnv",
    )
