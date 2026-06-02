"""Repository and bundled data paths."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Project root (contains ``pyproject.toml`` and ``data/``)."""
    return Path(__file__).resolve().parents[3]


def default_traffic_csv() -> Path:
    """Bundled RPS trace at ``data/traffic_trace.csv``."""
    return repo_root() / "data" / "traffic_trace.csv"
