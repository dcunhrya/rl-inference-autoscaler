import numpy as np

from rl_inference_autoscaler.env import AutoscalerEnv
from rl_inference_autoscaler.policies.baselines import (
    do_nothing_policy,
    evaluate_policy,
    greedy_policy,
    ideal_replica_count,
)


def test_do_nothing_holds_replicas():
    env = AutoscalerEnv(config={"traffic_mode": "synthetic", "cold_start_steps": 3})
    env.reset(seed=0)
    env.step(2)
    assert env._active_replicas == 1.0  # noqa: SLF001


def test_fixed_replica_initial_count():
    env = AutoscalerEnv(
        config={"traffic_mode": "synthetic", "initial_replicas": 4.0}
    )
    obs, _ = env.reset(seed=1)
    assert obs[2] == 4.0


def test_greedy_policy_valid_action():
    env = AutoscalerEnv(config={"traffic_mode": "synthetic"})
    obs, _ = env.reset(seed=2)
    action = greedy_policy(obs, env)
    assert action in (0, 1, 2)


def test_ideal_replica_count():
    env = AutoscalerEnv(config={"throughput_per_replica": 50.0})
    assert ideal_replica_count(75.0, env) == 2
    assert ideal_replica_count(50.0, env) == 1


def test_evaluate_policy_runs():
    metrics = evaluate_policy(
        do_nothing_policy,
        AutoscalerEnv(config={"traffic_mode": "synthetic", "max_steps_per_episode": 10}),
        episodes=2,
        seed=0,
    )
    assert len(metrics["episode_returns"]) == 2
    assert metrics["episode_return_mean"] < 0
