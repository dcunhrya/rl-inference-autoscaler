"""Backward-compatible entrypoint; implementation lives in ``training.ppo``."""

from rl_inference_autoscaler.training.ppo import main, run_training

__all__ = ["main", "run_training"]
