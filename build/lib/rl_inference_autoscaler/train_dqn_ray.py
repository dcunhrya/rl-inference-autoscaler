"""Backward-compatible entrypoint; implementation lives in ``training.dqn``."""

from rl_inference_autoscaler.training.dqn import main, run_dqn_training

__all__ = ["main", "run_dqn_training"]
