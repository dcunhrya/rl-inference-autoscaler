#!/usr/bin/env python3
"""Regenerate data/traffic_trace.csv with GPU-style bursty traffic."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from rl_inference_autoscaler.traffic import write_traffic_csv


def main() -> None:
    path = write_traffic_csv(_REPO / "data" / "traffic_trace.csv", steps=200, seed=42)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
