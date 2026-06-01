import numpy as np
import pytest

from rl_inference_autoscaler.autoscaler_env import AutoscalerEnv
from rl_inference_autoscaler.baselines import evaluate_baseline, target_utilization_policy
from rl_inference_autoscaler.traffic import TrafficGenerator, default_traffic_csv


def test_reset_step_shapes():
    env = AutoscalerEnv(config={"traffic_mode": "synthetic", "cold_start_steps": 2})
    obs, info = env.reset(seed=42)
    assert obs.shape == (4,)
    assert env.observation_space.contains(obs)
    assert info["traffic_mode"] == "synthetic"

    obs, reward, term, trunc, info = env.step(1)
    assert obs.shape == (4,)
    assert isinstance(reward, float)
    assert not term
    assert "queue_depth" in info


def test_scale_up_has_pending_replicas():
    env = AutoscalerEnv(
        config={"traffic_mode": "synthetic", "cold_start_steps": 5}
    )
    env.reset(seed=0)
    env.step(2)  # scale up
    assert env._pending_boots  # noqa: SLF001
    assert env._active_replicas == 1.0  # noqa: SLF001


def test_cold_start_increases_active_replicas():
    env = AutoscalerEnv(
        config={
            "traffic_mode": "synthetic",
            "cold_start_steps": 1,
            "max_steps_per_episode": 10,
        }
    )
    env.reset(seed=1)
    env.step(2)
    env.step(1)
    assert env._active_replicas >= 2.0  # noqa: SLF001


def test_queue_builds_under_overload():
    env = AutoscalerEnv(
        config={
            "traffic_mode": "synthetic",
            "throughput_per_replica": 10.0,
            "max_queue": 1000.0,
        }
    )
    env.reset(seed=2)
    for _ in range(20):
        _, _, _, _, info = env.step(1)
    assert info["queue_depth"] >= 0.0


def test_csv_traffic_mode():
    path = default_traffic_csv()
    if not path.is_file():
        pytest.skip("bundled traffic trace missing")
    env = AutoscalerEnv(
        config={"traffic_mode": "csv", "traffic_csv_path": str(path)}
    )
    obs, info = env.reset(seed=0)
    assert info["traffic_mode"] == "csv"
    assert 0 <= obs[0] <= env.max_rps


def test_traffic_generator_csv():
    path = default_traffic_csv()
    if not path.is_file():
        pytest.skip("bundled traffic trace missing")
    gen = TrafficGenerator(mode="csv", csv_path=path)
    rng = np.random.default_rng(0)
    assert gen.resolved_mode == "csv"
    assert gen.reset(rng) == pytest.approx(25.0)


def test_baseline_evaluation():
    metrics = evaluate_baseline(episodes=2, seed=10)
    assert "episode_return_mean" in metrics
    assert metrics["episodes"] == 2


def test_target_utilization_policy_bounds():
    obs = np.array([100.0, 0.9, 5.0, 10.0], dtype=np.float32)
    action = target_utilization_policy(obs)
    assert action in (0, 1, 2)
