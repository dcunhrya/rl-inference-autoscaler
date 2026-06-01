import importlib
import sys

import pytest

from rl_inference_autoscaler import register_env
from rl_inference_autoscaler.train_config import TrainingSettings, validate_ppo_config


def test_register_env_idempotent():
    register_env()
    import gymnasium as gym

    spec = gym.spec("Autoscaler-v0")
    assert spec is not None


def test_validate_ppo_config_dry():
    pytest.importorskip("ray")
    meta = validate_ppo_config(TrainingSettings(num_env_runners=1))
    assert meta["algorithm"] == "PPO"
    assert meta["env_class"] == "AutoscalerEnv"


def test_train_ray_dry_run():
    pytest.importorskip("ray")
    from rl_inference_autoscaler.train_ray import run_training

    summary = run_training(TrainingSettings(num_iterations=1), dry_run=True)
    assert summary["dry_run"] is True
    assert summary["config_built"] is True


def test_modal_module_structure():
    pytest.importorskip("modal")
    modal_train = importlib.import_module("rl_inference_autoscaler.modal_train")
    assert hasattr(modal_train, "app")
    assert hasattr(modal_train, "train_on_modal")
    assert modal_train.APP_NAME == "rl-inference-autoscaler"
    assert modal_train.get_app().name == "rl-inference-autoscaler"


def test_modal_local_entrypoint_dry_run(capsys):
    pytest.importorskip("modal")
    from rl_inference_autoscaler import modal_train

    modal_train.main(iterations=10, dry_run=True)
    captured = capsys.readouterr()
    assert "dry run" in captured.out.lower() or "no job submitted" in captured.out.lower()
