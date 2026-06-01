"""Traffic generators for AutoscalerEnv.

Data sources (in priority order when ``traffic_mode`` is ``auto``):

1. **CSV trace** — ``data/traffic_trace.csv`` (columns: ``step``, ``rps``). Checked in at
   repo root or path from env config ``traffic_csv_path``. Use this for reproducible
   evaluation or when you export real cluster metrics.
2. **Synthetic** — Random walk + Gaussian noise + 5% spike events. Default for training;
   no external download required. Models bursty inference traffic when no trace exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

TrafficMode = Literal["auto", "synthetic", "csv"]


def default_traffic_csv() -> Path:
    """Default bundled trace relative to repository root."""
    return Path(__file__).resolve().parents[2] / "data" / "traffic_trace.csv"


@dataclass
class TrafficGenerator:
    """Produces request rate (RPS) for each simulator timestep."""

    mode: TrafficMode = "auto"
    max_rps: float = 1000.0
    csv_path: Path | None = None
    spike_magnitude: float = 100.0
    spike_probability: float = 0.05
    noise_std: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.csv_path, str):
            self.csv_path = Path(self.csv_path)
        self._csv_rps: np.ndarray | None = None
        self._csv_index: int = 0
        self._resolved_mode: Literal["synthetic", "csv"] = "synthetic"
        self._resolve_mode()

    def _resolve_mode(self) -> None:
        if self.mode == "synthetic":
            self._resolved_mode = "synthetic"
            return
        path = Path(self.csv_path) if self.csv_path else default_traffic_csv()
        if self.mode == "csv" or (self.mode == "auto" and path.is_file()):
            self._load_csv(path)
            self._resolved_mode = "csv"
        else:
            self._resolved_mode = "synthetic"

    def _load_csv(self, path: Path) -> None:
        df = pd.read_csv(path)
        if "rps" not in df.columns:
            raise ValueError(f"traffic CSV must contain 'rps' column: {path}")
        self._csv_rps = np.clip(df["rps"].astype(float).to_numpy(), 0.0, self.max_rps)
        self._csv_index = 0

    @property
    def resolved_mode(self) -> str:
        return self._resolved_mode

    def reset(self, rng: np.random.Generator, initial_rps: float | None = None) -> float:
        """Return RPS at episode start."""
        self._csv_index = 0
        if self._resolved_mode == "csv" and self._csv_rps is not None:
            return float(self._csv_rps[0])
        return float(initial_rps if initial_rps is not None else rng.uniform(10.0, 50.0))

    def next_rps(self, rng: np.random.Generator, current_rps: float) -> float:
        """Advance one timestep and return new RPS."""
        if self._resolved_mode == "csv" and self._csv_rps is not None:
            idx = min(self._csv_index, len(self._csv_rps) - 1)
            rps = float(self._csv_rps[idx])
            self._csv_index += 1
            return rps

        spike = rng.choice(
            [0.0, self.spike_magnitude],
            p=[1.0 - self.spike_probability, self.spike_probability],
        )
        noise = rng.normal(0.0, self.noise_std)
        return float(np.clip(current_rps + noise + spike, 0.0, self.max_rps))
