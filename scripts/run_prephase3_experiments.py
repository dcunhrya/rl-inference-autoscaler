#!/usr/bin/env python3
"""
Run achievable experiments from project_info/experiments_backlog.md (sections 1–5, before Phase 3).

Lightweight eval only (no re-training). Outputs JSON under results/experiments/.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import numpy as np

from rl_inference_autoscaler.autoscaler_env import AutoscalerEnv
from rl_inference_autoscaler.baselines import (
    evaluate_policy,
    fixed_replica_policy,
    greedy_policy,
)
from rl_inference_autoscaler.evaluation import (
    evaluate_rllib_checkpoint,
    resolve_checkpoint_path,
    run_benchmark_suite,
)

OUT_DIR = _REPO / "results" / "experiments"


def _oracle_gap(trajectory: dict) -> dict[str, float]:
    active = np.asarray(trajectory["active_replicas"])
    ideal = np.asarray(trajectory["ideal_replicas"])
    diff = np.abs(active - ideal)
    return {
        "mean_abs_gap": float(np.mean(diff)),
        "max_abs_gap": float(np.max(diff)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
    }


def _eval_greedy(env_cfg: dict, *, episodes: int, seed: int) -> dict:
    env = AutoscalerEnv(config=env_cfg)
    return evaluate_policy(
        greedy_policy,
        env,
        episodes=episodes,
        seed=seed,
        record_trajectory_episode=0,
    )


def run_r1_reward_sweep(*, episodes: int = 10, seed: int = 42) -> dict:
    """R1: α/β/γ grid — greedy eval only (fast calibration)."""
    grid = [
        {"cost_alpha": 0.1, "latency_beta": 1.0, "queue_penalty_gamma": 0.05},
        {"cost_alpha": 0.2, "latency_beta": 0.5, "queue_penalty_gamma": 0.05},
        {"cost_alpha": 0.05, "latency_beta": 2.0, "queue_penalty_gamma": 0.1},
        {"cost_alpha": 0.15, "latency_beta": 1.5, "queue_penalty_gamma": 0.05},
        {"cost_alpha": 0.1, "latency_beta": 0.5, "queue_penalty_gamma": 0.1},
    ]
    rows = []
    for cfg in grid:
        env_cfg = {"traffic_mode": "auto", **cfg}
        res = _eval_greedy(env_cfg, episodes=episodes, seed=seed)
        gap = _oracle_gap(res["trajectory"]) if "trajectory" in res else {}
        rows.append(
            {
                "config": cfg,
                "episode_return_mean": res["episode_return_mean"],
                "mean_cost_penalty": res["mean_cost_penalty"],
                "mean_latency_penalty": res["mean_latency_penalty"],
                **gap,
            }
        )
    return {"experiment": "R1", "episodes": episodes, "seed": seed, "rows": rows}


def run_r2_r3_presets(*, episodes: int = 20, seed: int = 42) -> dict:
    """R2 cost-sensitive, R3 latency-sensitive presets."""
    presets = {
        "balanced_default": {
            "cost_alpha": 0.1,
            "latency_beta": 1.0,
            "queue_penalty_gamma": 0.05,
        },
        "cost_sensitive": {"cost_alpha": 0.2, "latency_beta": 0.5, "queue_penalty_gamma": 0.05},
        "latency_sensitive": {
            "cost_alpha": 0.05,
            "latency_beta": 2.0,
            "queue_penalty_gamma": 0.1,
        },
    }
    out = {}
    for name, cfg in presets.items():
        env_cfg = {"traffic_mode": "auto", **cfg}
        res = _eval_greedy(env_cfg, episodes=episodes, seed=seed)
        out[name] = {
            "config": cfg,
            "episode_return_mean": res["episode_return_mean"],
            "mean_cost_penalty": res["mean_cost_penalty"],
            "mean_latency_penalty": res["mean_latency_penalty"],
        }
    return {"experiment": "R2_R3", "episodes": episodes, "seed": seed, "presets": out}


def run_t2_csv_eval(*, episodes: int = 20, seed: int = 42) -> dict:
    """T2: CSV-only eval for all policies."""
    return run_benchmark_suite(
        env_config={"traffic_mode": "csv"},
        episodes=episodes,
        seed=seed,
        mlflow_db=_REPO / "mlflow.db",
    )


def _eval_policy_at_cold_start(
    policy_name: str, env_cfg: dict, *, episodes: int, seed: int
) -> dict | None:
    env = AutoscalerEnv(config=env_cfg)
    if policy_name == "greedy":
        return evaluate_policy(greedy_policy, env, episodes=episodes, seed=seed)
    ckpt_map = {
        "ppo": (_REPO / "checkpoints" / "ppo" / "final", "checkpoints/final"),
        "dqn": (_REPO / "checkpoints" / "dqn" / "final",),
    }
    if policy_name not in ckpt_map:
        return None
    paths = ckpt_map[policy_name]
    ckpt = resolve_checkpoint_path(*paths)
    if ckpt is None:
        return None
    return evaluate_rllib_checkpoint(ckpt, env, episodes=episodes, seed=seed)


def run_t7_cold_start_ablation(*, episodes: int = 10, seed: int = 42) -> dict:
    """T7: cold_start_steps — greedy, PPO, DQN."""
    steps = [0, 3, 5, 10]
    rows = []
    for cs in steps:
        env_cfg = {"traffic_mode": "auto", "cold_start_steps": cs}
        row: dict = {"cold_start_steps": cs, "policies": {}}
        for pname in ("greedy", "ppo", "dqn"):
            res = _eval_policy_at_cold_start(pname, env_cfg, episodes=episodes, seed=seed)
            if res:
                row["policies"][pname] = {
                    "episode_return_mean": res["episode_return_mean"],
                    "mean_latency_penalty": res["mean_latency_penalty"],
                }
        rows.append(row)
    return {"experiment": "T7", "episodes": episodes, "seed": seed, "rows": rows}


def run_t3_held_out(*, episodes: int = 20, seed: int = 42) -> dict:
    """T3: train synthetic → eval CSV and train CSV checkpoint on synthetic."""
    out: dict = {"experiment": "T3", "evals": {}}
    synth_ckpt = resolve_checkpoint_path(_REPO / "checkpoints" / "ppo_synth" / "final")
    if synth_ckpt:
        out["evals"]["ppo_synth_train_csv_eval"] = evaluate_rllib_checkpoint(
            synth_ckpt,
            AutoscalerEnv(config={"traffic_mode": "csv"}),
            episodes=episodes,
            seed=seed,
        )
        out["evals"]["ppo_synth_train_csv_eval"] = {
            k: out["evals"]["ppo_synth_train_csv_eval"][k]
            for k in ("episode_return_mean", "mean_cost_penalty", "mean_latency_penalty")
        }
    csv_ckpt = resolve_checkpoint_path(_REPO / "checkpoints" / "ppo_csv" / "final")
    if csv_ckpt:
        r = evaluate_rllib_checkpoint(
            csv_ckpt,
            AutoscalerEnv(config={"traffic_mode": "synthetic"}),
            episodes=episodes,
            seed=seed,
        )
        out["evals"]["ppo_csv_train_synth_eval"] = {
            k: r[k]
            for k in ("episode_return_mean", "mean_cost_penalty", "mean_latency_penalty")
        }
    return out


def run_t5_spike_stress(*, episodes: int = 15, seed: int = 42) -> dict:
    """T5: high spike magnitude / probability on synthetic."""
    configs = {
        "default": {"traffic_mode": "synthetic"},
        "stress": {
            "traffic_mode": "synthetic",
            "spike_magnitude": 300.0,
            "spike_probability": 0.15,
        },
    }
    rows = {}
    for name, cfg in configs.items():
        res = _eval_greedy(cfg, episodes=episodes, seed=seed)
        rows[name] = {
            "episode_return_mean": res["episode_return_mean"],
            "mean_latency_penalty": res["mean_latency_penalty"],
        }
    return {"experiment": "T5", "rows": rows}


def run_t2_multi_seed(*, episodes: int = 15) -> dict:
    seeds = [0, 42, 123]
    return {
        "experiment": "T2_seeds",
        "traffic_mode": "csv",
        "seeds": {
            str(s): {
                k: v.get("episode_return_mean")
                for k, v in run_benchmark_suite(
                    env_config={"traffic_mode": "csv"},
                    episodes=episodes,
                    seed=s,
                    mlflow_db=_REPO / "mlflow.db",
                )["policies"].items()
                if isinstance(v, dict) and "error" not in v
            }
            for s in seeds
        },
    }


def run_a10_rl_seeds(*, episodes: int = 15) -> dict:
    seeds = [0, 42, 123]
    rows = []
    for s in seeds:
        bench = run_benchmark_suite(
            episodes=episodes, seed=s, mlflow_db=_REPO / "mlflow.db"
        )
        rows.append(
            {
                "seed": s,
                "ppo": bench["policies"].get("ppo", {}).get("episode_return_mean"),
                "dqn": bench["policies"].get("dqn", {}).get("episode_return_mean"),
                "greedy": bench["policies"].get("greedy", {}).get("episode_return_mean"),
            }
        )
    return {"experiment": "A10_rl", "episodes": episodes, "rows": rows}


def run_r4_churn_eval(*, episodes: int = 15, seed: int = 42) -> dict:
    configs = [
        {"traffic_mode": "auto"},
        {"traffic_mode": "auto", "churn_penalty_delta": 0.05},
    ]
    rows = []
    for cfg in configs:
        res = _eval_greedy(cfg, episodes=episodes, seed=seed)
        rows.append({"config": cfg, "episode_return_mean": res["episode_return_mean"]})
    return {"experiment": "R4_eval", "rows": rows}


def run_r6_util_band(*, episodes: int = 15, seed: int = 42) -> dict:
    cfg = {
        "traffic_mode": "auto",
        "util_band_penalty_zeta": 0.5,
        "util_band_low": 0.6,
        "util_band_high": 0.8,
    }
    res = _eval_greedy(cfg, episodes=episodes, seed=seed)
    return {
        "experiment": "R6",
        "config": cfg,
        "episode_return_mean": res["episode_return_mean"],
    }


def run_t4_production_trace_note() -> dict:
    return {
        "experiment": "T4",
        "status": "complete",
        "note": "data/traffic_trace.csv is the production-style trace used via traffic_mode=auto/csv",
    }


def run_b3_oracle_gap(*, seed: int = 42) -> dict:
    """B3: |N_active - N_ideal| for PPO, DQN, greedy."""
    env_cfg = {"traffic_mode": "auto"}
    policies: dict[str, dict] = {}

    greedy_res = evaluate_policy(
        greedy_policy,
        AutoscalerEnv(config=env_cfg),
        episodes=1,
        seed=seed,
        record_trajectory_episode=0,
    )
    policies["greedy"] = _oracle_gap(greedy_res["trajectory"])

    ppo_ckpt = resolve_checkpoint_path(_REPO / "checkpoints" / "ppo" / "final", "checkpoints/final")
    if ppo_ckpt:
        ppo_res = evaluate_rllib_checkpoint(
            ppo_ckpt,
            AutoscalerEnv(config=env_cfg),
            episodes=1,
            seed=seed,
            record_trajectory_episode=0,
        )
        if "trajectory" in ppo_res:
            policies["ppo"] = _oracle_gap(ppo_res["trajectory"])

    dqn_ckpt = resolve_checkpoint_path(_REPO / "checkpoints" / "dqn" / "final")
    if dqn_ckpt:
        dqn_res = evaluate_rllib_checkpoint(
            dqn_ckpt,
            AutoscalerEnv(config=env_cfg),
            episodes=1,
            seed=seed,
            record_trajectory_episode=0,
        )
        if "trajectory" in dqn_res:
            policies["dqn"] = _oracle_gap(dqn_res["trajectory"])

    return {"experiment": "B3", "seed": seed, "policies": policies}


def _action_histogram(env: AutoscalerEnv, seed: int, choose_action) -> dict[str, int]:
    counts = {"scale_down": 0, "hold": 0, "scale_up": 0}
    obs, _ = env.reset(seed=seed)
    truncated = False
    while not truncated:
        a = int(choose_action(obs, env))
        key = ("scale_down", "hold", "scale_up")[a]
        counts[key] += 1
        obs, _, _, truncated, _ = env.step(a)
    return counts


def run_b4_action_distribution(*, seed: int = 42) -> dict:
    """B4: action histogram {0,1,2} per policy on one rollout."""
    from rl_inference_autoscaler.baselines import do_nothing_policy
    from rl_inference_autoscaler.evaluation import (
        _checkpoint_version,
        _load_new_stack_action_fn,
        _load_old_stack_action_fn,
    )

    env_cfg = {"traffic_mode": "auto"}
    out: dict[str, dict] = {
        "greedy": _action_histogram(
            AutoscalerEnv(config=env_cfg), seed, greedy_policy
        ),
        "do_nothing": _action_histogram(
            AutoscalerEnv(config=env_cfg), seed, do_nothing_policy
        ),
    }

    ppo_ckpt = resolve_checkpoint_path(
        _REPO / "checkpoints" / "ppo" / "final", "checkpoints/final"
    )
    if ppo_ckpt:
        version, _ = _checkpoint_version(ppo_ckpt)
        fn = (
            _load_new_stack_action_fn(ppo_ckpt)
            if version >= 2.0
            else _load_old_stack_action_fn(ppo_ckpt)
        )
        out["ppo"] = _action_histogram(
            AutoscalerEnv(config=env_cfg),
            seed,
            lambda obs, _env: fn(obs),
        )

    dqn_ckpt = resolve_checkpoint_path(_REPO / "checkpoints" / "dqn" / "final")
    if dqn_ckpt:
        fn = _load_new_stack_action_fn(dqn_ckpt)
        out["dqn"] = _action_histogram(
            AutoscalerEnv(config=env_cfg),
            seed,
            lambda obs, _env: fn(obs),
        )

    return {"experiment": "B4", "seed": seed, "action_counts": out}


def run_b5_extended_eval(*, episodes: int = 50, seed: int = 42) -> dict:
    """B5: 50-episode benchmark (main table)."""
    bench = run_benchmark_suite(
        episodes=episodes,
        seed=seed,
        mlflow_db=_REPO / "mlflow.db",
    )
    return {
        "experiment": "B5",
        "episodes": episodes,
        "seed": seed,
        "policies": {
            k: {
                "episode_return_mean": v.get("episode_return_mean"),
                "episode_return_std": v.get("episode_return_std"),
                "mean_cost_penalty": v.get("mean_cost_penalty"),
                "mean_latency_penalty": v.get("mean_latency_penalty"),
                "error": v.get("error"),
            }
            for k, v in bench["policies"].items()
        },
    }


def run_a10_seed_sensitivity(*, episodes: int = 15) -> dict:
    """A10: seeds {0, 42, 123} on greedy."""
    seeds = [0, 42, 123]
    rows = []
    for s in seeds:
        res = _eval_greedy({"traffic_mode": "auto"}, episodes=episodes, seed=s)
        rows.append(
            {
                "seed": s,
                "episode_return_mean": res["episode_return_mean"],
                "episode_return_std": res["episode_return_std"],
            }
        )
    return {"experiment": "A10", "episodes": episodes, "rows": rows}


def run_t6_multi_trajectory(*, seed: int = 42) -> dict:
    """T6: record greedy trajectories for episodes 0, 5, 10."""
    episodes_out = []
    for ep in [0, 5, 10]:
        env = AutoscalerEnv(config={"traffic_mode": "auto"})
        res = evaluate_policy(
            greedy_policy,
            env,
            episodes=ep + 1,
            seed=seed,
            record_trajectory_episode=ep,
        )
        if "trajectory" in res:
            episodes_out.append({"episode_index": ep, "oracle_gap": _oracle_gap(res["trajectory"])})
    return {"experiment": "T6", "seed": seed, "episodes": episodes_out}


def _serialize(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    return obj


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runners = [
        ("r1_reward_sweep.json", run_r1_reward_sweep),
        ("r2_r3_presets.json", run_r2_r3_presets),
        ("r4_churn_eval.json", run_r4_churn_eval),
        ("r6_util_band.json", run_r6_util_band),
        ("t2_csv_benchmark.json", run_t2_csv_eval),
        ("t2_multi_seed.json", run_t2_multi_seed),
        ("t3_held_out.json", run_t3_held_out),
        ("t4_production_trace.json", run_t4_production_trace_note),
        ("t5_spike_stress.json", run_t5_spike_stress),
        ("t7_cold_start.json", run_t7_cold_start_ablation),
        ("b3_oracle_gap.json", run_b3_oracle_gap),
        ("b4_action_distribution.json", run_b4_action_distribution),
        ("b5_extended_benchmark.json", run_b5_extended_eval),
        ("a10_seed_sensitivity.json", run_a10_seed_sensitivity),
        ("a10_rl_seeds.json", run_a10_rl_seeds),
        ("t6_multi_trajectory.json", run_t6_multi_trajectory),
    ]
    index = {"experiments": []}
    for filename, fn in runners:
        print(f"Running {fn.__name__}...")
        data = _serialize(fn())
        path = OUT_DIR / filename
        path.write_text(json.dumps(data, indent=2))
        print(f"  Wrote {path}")
        index["experiments"].append({"file": filename, "experiment": data.get("experiment")})
    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Done. Index: {OUT_DIR / 'index.json'}")


if __name__ == "__main__":
    main()
