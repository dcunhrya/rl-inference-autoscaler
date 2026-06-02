#!/usr/bin/env python3
"""Thin CLI for local Ray RLlib DQN training (see train_dqn_ray.py for logic)."""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from rl_inference_autoscaler.training.dqn import main

if __name__ == "__main__":
    main()
