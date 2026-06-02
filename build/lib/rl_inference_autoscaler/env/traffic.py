"""Traffic generators for AutoscalerEnv.

Data sources (in priority order when ``traffic_mode`` is ``auto``):

1. **CSV trace** — ``data/traffic_trace.csv`` (columns: ``step``, ``rps``). Checked in at
   repo root or path from env config ``traffic_csv_path``. Use this for reproducible
   evaluation or when you export real cluster metrics.
2. **Synthetic** — Random walk + Gaussian noise + multi-step burst events. Default when
   no trace exists; models bursty GPU inference traffic (job batches, queue buildup).

Regenerate the bundled trace::

    python scripts/generate_traffic_trace.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

TrafficMode = Literal["auto", "synthetic", "csv"]


def default_traffic_csv() -> Path:
    """Default bundled trace relative to repository root."""
    return Path(__file__).resolve().parents[2] / "data" / "traffic_trace.csv"


def generate_gpu_autoscaler_trace(
    steps: int = 200,
    *,
    max_rps: float = 1000.0,
    seed: int = 42,
    base_rps_range: tuple[float, float] = (25.0, 110.0),
    burst_count_range: tuple[int, int] = (10, 16),
    burst_peak_range: tuple[float, float] = (160.0, 720.0),
) -> np.ndarray:
    """
    Build a reproducible RPS trace mimicking GPU inference autoscaler load.

    Patterns: drifting baseline, jitter, multi-step ramps/plateaus/decays, and
    occasional sharp bursts (batch jobs / traffic spikes).
    """
    rng = np.random.default_rng(seed)
    rps = np.zeros(steps, dtype=np.float64)
    base = float(rng.uniform(*base_rps_range))

    for t in range(steps):
        base += rng.normal(0.0, 4.0)
        base = float(np.clip(base, *base_rps_range))
        micro_spike = 0.0
        if rng.random() < 0.12:
            micro_spike = rng.uniform(20.0, 90.0)
        rps[t] = base + rng.normal(0.0, 10.0) + micro_spike

    n_bursts = int(rng.integers(burst_count_range[0], burst_count_range[1] + 1))
    for _ in range(n_bursts):
        duration = int(rng.integers(4, 18))
        start = int(rng.integers(0, max(1, steps - duration)))
        peak = float(rng.uniform(*burst_peak_range))
        ramp = int(rng.integers(1, 4))
        tail = int(rng.integers(2, 6))

        for i in range(duration):
            t = start + i
            if t >= steps:
                break
            if i < ramp:
                factor = (i + 1) / ramp
            elif i < duration - tail:
                factor = 1.0 + rng.normal(0.0, 0.06)
            else:
                factor = max(0.25, 1.0 - (i - (duration - tail)) * 0.22)
            burst_rps = peak * factor + rng.normal(0.0, 15.0)
            rps[t] = max(rps[t], burst_rps)

    return np.clip(rps, 0.0, max_rps)


def write_traffic_csv(
    path: Path | None = None,
    *,
    steps: int = 200,
    seed: int = 42,
    max_rps: float = 1000.0,
) -> Path:
    """Write ``step,rps`` CSV and return the path."""
    path = Path(path) if path else default_traffic_csv()
    path.parent.mkdir(parents=True, exist_ok=True)
    rps = generate_gpu_autoscaler_trace(steps, max_rps=max_rps, seed=seed)
    df = pd.DataFrame({"step": np.arange(len(rps)), "rps": rps})
    df.to_csv(path, index=False)
    return path


@dataclass
class TrafficGenerator:
    """Produces request rate (RPS) for each simulator timestep."""

    mode: TrafficMode = "auto"
    max_rps: float = 1000.0
    csv_path: Path | None = None
    spike_magnitude: float = 150.0
    spike_probability: float = 0.08
    noise_std: float = 8.0
    spike_duration_range: tuple[int, int] = (3, 12)
    _spike_steps_left: int = field(default=0, init=False, repr=False)
    _spike_target: float = field(default=0.0, init=False, repr=False)
    _csv_rps: np.ndarray | None = field(default=None, init=False, repr=False)
    _csv_index: int = field(default=0, init=False, repr=False)
    _resolved_mode: Literal["synthetic", "csv"] = field(
        default="synthetic", init=False, repr=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.csv_path, str):
            self.csv_path = Path(self.csv_path)
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
        self._spike_steps_left = 0
        self._spike_target = 0.0
        if self._resolved_mode == "csv" and self._csv_rps is not None:
            return float(self._csv_rps[0])
        return float(initial_rps if initial_rps is not None else rng.uniform(20.0, 70.0))

    def _start_burst(self, rng: np.random.Generator, current_rps: float) -> None:
        lo, hi = self.spike_duration_range
        self._spike_steps_left = int(rng.integers(lo, hi + 1))
        uplift = rng.uniform(self.spike_magnitude * 0.4, self.spike_magnitude * 2.2)
        self._spike_target = float(np.clip(current_rps + uplift, 0.0, self.max_rps))

    def _sample_synthetic(self, rng: np.random.Generator, current_rps: float) -> float:
        noise = rng.normal(0.0, self.noise_std)
        drift = rng.normal(0.0, self.noise_std * 0.35)

        if self._spike_steps_left > 0:
            self._spike_steps_left -= 1
            # Blend toward spike level, then decay on last steps.
            weight = 0.55 if self._spike_steps_left > 0 else 0.25
            blended = (1.0 - weight) * current_rps + weight * self._spike_target
            return float(np.clip(blended + noise, 0.0, self.max_rps))

        if rng.random() < self.spike_probability:
            self._start_burst(rng, current_rps + drift)
            return float(
                np.clip(current_rps + drift + self._spike_target * 0.15, 0.0, self.max_rps)
            )

        if rng.random() < 0.1:
            drift += rng.uniform(15.0, 45.0)

        return float(np.clip(current_rps + drift + noise, 0.0, self.max_rps))

    def peek_next_rps(self, rng: np.random.Generator, current_rps: float) -> float:
        """Next RPS without advancing the CSV index (oracle / planning)."""
        if self._resolved_mode == "csv" and self._csv_rps is not None:
            idx = min(self._csv_index, len(self._csv_rps) - 1)
            return float(self._csv_rps[idx])
        state = rng.bit_generator.state
        spike_steps = self._spike_steps_left
        spike_target = self._spike_target
        try:
            return self._sample_synthetic(rng, current_rps)
        finally:
            rng.bit_generator.state = state
            self._spike_steps_left = spike_steps
            self._spike_target = spike_target

    def next_rps(self, rng: np.random.Generator, current_rps: float) -> float:
        """Advance one timestep and return new RPS."""
        if self._resolved_mode == "csv" and self._csv_rps is not None:
            idx = min(self._csv_index, len(self._csv_rps) - 1)
            rps = float(self._csv_rps[idx])
            self._csv_index += 1
            return rps
        return self._sample_synthetic(rng, current_rps)
