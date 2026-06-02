"""Heuristic and baseline policies for comparison."""

from rl_inference_autoscaler.policies.baselines import (
    PolicyFn,
    do_nothing_policy,
    evaluate_baseline,
    evaluate_policy,
    fixed_replica_policy,
    greedy_policy,
    ideal_replica_count,
    oracle_policy,
    target_utilization_policy,
)

__all__ = [
    "PolicyFn",
    "do_nothing_policy",
    "evaluate_baseline",
    "evaluate_policy",
    "fixed_replica_policy",
    "greedy_policy",
    "ideal_replica_count",
    "oracle_policy",
    "target_utilization_policy",
]
