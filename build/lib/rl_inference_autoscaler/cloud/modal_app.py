"""
Modal cloud training wrapper.

All Modal-specific code lives here so ``train_ray.py`` / ``train_dqn_ray.py`` stay
platform-agnostic. Do not import this module in the core training path unless
launching on Modal.

Usage (when ready to train on Modal)::

    modal setup   # once
    modal run src/rl_inference_autoscaler/modal_train.py --iterations 100
    modal run src/rl_inference_autoscaler/modal_train.py --algorithm dqn --iterations 100

Tests verify this module defines an App and train function without calling ``.remote()``.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "rl-inference-autoscaler"
CHECKPOINT_VOLUME = "autoscaler-rl-checkpoints"
REPO_ROOT = Path(__file__).resolve().parents[3]

app = modal.App(APP_NAME)


def training_image() -> modal.Image:
    """Container image with Ray RLlib and project source."""
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "gymnasium>=1.3.0",
            "numpy>=2.4.6",
            "pandas>=3.0.3",
            "ray[rllib]>=2.40.0",
            "torch",
            "mlflow>=2.0.0",
        )
        .add_local_dir(
            REPO_ROOT / "src",
            remote_path="/root/src",
        )
        .add_local_file(
            REPO_ROOT / "pyproject.toml",
            remote_path="/root/pyproject.toml",
        )
        .add_local_dir(
            REPO_ROOT / "data",
            remote_path="/root/data",
        )
    )


checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME,
    create_if_missing=True,
)


@app.function(
    image=training_image(),
    cpu=8,
    timeout=4 * 60 * 60,
    volumes={"/checkpoints": checkpoint_volume},
)
def train_on_modal(
    algorithm: str = "ppo",
    iterations: int = 100,
    num_env_runners: int = 8,
    experiment_name: str | None = None,
    traffic_mode: str = "auto",
) -> dict:
    """Remote training worker; imports Ray training at runtime."""
    import sys

    sys.path.insert(0, "/root/src")
    algo = algorithm.lower()
    env_config = {
        "traffic_mode": traffic_mode,
        "traffic_csv_path": "/root/data/traffic_trace.csv",
    }

    if algo == "dqn":
        from rl_inference_autoscaler.training.config import DQNSettings
        from rl_inference_autoscaler.training.dqn import run_dqn_training

        settings = DQNSettings(
            num_iterations=iterations,
            num_env_runners=num_env_runners,
            checkpoint_dir="/checkpoints/dqn",
            experiment_name=experiment_name or "autoscaler-dqn-modal",
            mlflow_tracking_uri="file:/checkpoints/mlruns",
            env_config=env_config,
        )
        return run_dqn_training(settings, init_ray=True, shutdown_ray=True)

    if algo != "ppo":
        raise ValueError(f"unsupported algorithm: {algorithm!r} (use 'ppo' or 'dqn')")

    from rl_inference_autoscaler.training.config import TrainingSettings
    from rl_inference_autoscaler.training.ppo import run_training

    settings = TrainingSettings(
        num_iterations=iterations,
        num_env_runners=num_env_runners,
        checkpoint_dir="/checkpoints/ppo",
        experiment_name=experiment_name or "autoscaler-ppo-modal",
        mlflow_tracking_uri="file:/checkpoints/mlruns",
        env_config=env_config,
    )
    return run_training(settings, init_ray=True, shutdown_ray=True)


@app.local_entrypoint()
def main(
    algorithm: str = "ppo",
    iterations: int = 100,
    num_env_runners: int = 8,
    dry_run: bool = True,
):
    """
    Local CLI for Modal.

    Default ``dry_run=True`` prints instructions without submitting a job.
    Pass ``--no-dry-run`` to call ``train_on_modal.remote(...)``.
    """
    algo = algorithm.lower()
    if dry_run:
        print(
            "Modal dry run — no job submitted.\n"
            "To train on Modal:\n"
            f"  modal run {__file__} --no-dry-run --algorithm {algo} --iterations {iterations}\n"
            f"Checkpoints volume: {CHECKPOINT_VOLUME} mounted at /checkpoints/{algo}"
        )
        return

    result = train_on_modal.remote(
        algorithm=algo,
        iterations=iterations,
        num_env_runners=num_env_runners,
    )
    print(result)


def get_app() -> modal.App:
    """Helper for tests."""
    return app
