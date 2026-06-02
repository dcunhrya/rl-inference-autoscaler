#!/usr/bin/env python3
"""Pinned benchmark entrypoint (E2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from rl_inference_autoscaler.evaluation import run_benchmark_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pinned autoscaler benchmark")
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO / "eval.json",
    )
    parser.add_argument("--output", type=Path, default=_REPO / "results" / "benchmark_full.json")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    benchmark = run_benchmark_suite(
        ppo_checkpoint_path=_REPO / cfg["ppo_checkpoint"],
        dqn_checkpoint_path=_REPO / cfg["dqn_checkpoint"],
        env_config={"traffic_mode": cfg.get("traffic_mode", "auto")},
        episodes=int(cfg.get("episodes", 20)),
        seed=int(cfg.get("seed", 42)),
        fixed_replicas=int(cfg.get("fixed_replicas", 4)),
        mlflow_db=_REPO / cfg.get("mlflow_db", "mlflow.db"),
        trajectory_episode=int(cfg.get("trajectory_episode", 0)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark, indent=2, default=str))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
