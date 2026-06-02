#!/usr/bin/env python3
"""
Generate comparison and training plots under results/.

Usage:
    uv sync --extra train --no-editable
    .venv/bin/python scripts/generate_results_plots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import matplotlib.pyplot as plt
import numpy as np

from rl_inference_autoscaler.evaluation import run_benchmark_suite

RESULTS_DIR = _REPO / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

POLICY_LABELS = {
    "ppo": "PPO (trained)",
    "dqn": "DQN (trained)",
    "greedy": "Greedy",
    "fixed_replica": "Fixed replica",
    "do_nothing": "Do nothing",
}
POLICY_COLORS = {
    "ppo": "#2563eb",
    "dqn": "#9333ea",
    "greedy": "#16a34a",
    "fixed_replica": "#ca8a04",
    "do_nothing": "#dc2626",
}
PARETO_ORDER = ["ppo", "dqn", "greedy", "fixed_replica", "do_nothing"]
SCALING_COMPARE_POLICIES = ["ppo", "dqn", "greedy"]


def _plot_policy_comparison(benchmark: dict) -> None:
    policies = benchmark["policies"]
    fixed_n = benchmark.get("fixed_replicas", 4)
    labels = {**POLICY_LABELS, "fixed_replica": f"Fixed replica (n={fixed_n})"}

    names: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    bar_colors: list[str] = []

    for key in PARETO_ORDER:
        if key not in policies or "error" in policies[key]:
            continue
        names.append(labels[key])
        means.append(policies[key]["episode_return_mean"])
        stds.append(policies[key]["episode_return_std"])
        bar_colors.append(POLICY_COLORS[key])

    if not names:
        print("No policy results available; skipping policy comparison plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=6, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Episode return (mean ± std)")
    ax.set_title("Inference autoscaler: PPO & DQN vs baselines")
    ax.axhline(0, color="#666", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(
        0.02,
        0.02,
        "Higher is better (closer to 0).\n"
        f"Episodes={benchmark['episodes']}, seed={benchmark['seed']}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        color="#444",
    )
    fig.tight_layout()
    out = FIGURES_DIR / "policy_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def _pareto_nondominated(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Minimize both axes: keep non-dominated (cost, latency) points."""
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    frontier: list[tuple[float, float]] = []
    best_latency = float("inf")
    for cost, latency in sorted_pts:
        if latency < best_latency:
            frontier.append((cost, latency))
            best_latency = latency
    return frontier


def _plot_pareto_frontier(benchmark: dict) -> None:
    policies = benchmark["policies"]
    fixed_n = benchmark.get("fixed_replicas", 4)
    labels = {**POLICY_LABELS, "fixed_replica": f"Fixed replica (n={fixed_n})"}

    fig, ax = plt.subplots(figsize=(8, 6))
    points: list[tuple[float, float]] = []

    for key in PARETO_ORDER:
        data = policies.get(key)
        if not data or "error" in data:
            continue
        cost = data["mean_cost_penalty"]
        latency = data["mean_latency_penalty"]
        ret = data["episode_return_mean"]
        points.append((cost, latency))
        ax.scatter(
            cost,
            latency,
            s=120,
            color=POLICY_COLORS[key],
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
            label=labels[key],
        )
        ax.annotate(
            f"{labels[key]}\nreturn={ret:.0f}",
            (cost, latency),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
            color="#333",
        )

    if len(points) >= 2:
        frontier = _pareto_nondominated(points)
        if len(frontier) >= 2:
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, "--", color="#64748b", linewidth=1.5, label="Pareto frontier", zorder=2)

    ax.set_xlabel("Mean episode cost penalty (Σ α·N_t)")
    ax.set_ylabel("Mean episode latency penalty (Σ β·overload/drops/queue)")
    ax.set_title("Cost–latency tradeoff (Pareto view)")
    ax.text(
        0.02,
        0.98,
        "Lower-left is better.\nReturn = -(cost + latency) per episode.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        color="#444",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "pareto_frontier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_scaling_vs_ground_truth(
    benchmark: dict,
    policy_key: str,
    outfile: str,
    title_suffix: str,
) -> None:
    policy = benchmark["policies"].get(policy_key, {})
    traj = policy.get("trajectory")
    if not traj:
        print(f"No {policy_key.upper()} trajectory recorded; skipping {outfile}.")
        return

    steps = np.arange(len(traj["active_replicas"]))
    active = np.array(traj["active_replicas"])
    ideal = np.array(traj["ideal_replicas"])
    rps = np.array(traj["rps"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    axes[0].step(steps, ideal, where="post", color="#16a34a", linewidth=2, label="Ground truth (ideal replicas)")
    axes[0].step(
        steps,
        active,
        where="post",
        color=POLICY_COLORS[policy_key],
        linewidth=2,
        linestyle="--",
        label=f"{POLICY_LABELS[policy_key]} (active replicas)",
    )
    axes[0].set_ylabel("Replica count (discrete)")
    axes[0].set_title(
        f"{title_suffix} scaling vs ground truth "
        f"(episode seed={benchmark['seed'] + benchmark.get('trajectory_episode', 0)})"
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0.5)

    axes[1].plot(steps, rps, color="#7c3aed", linewidth=1.2, alpha=0.9)
    axes[1].set_ylabel("RPS")
    axes[1].set_xlabel("Timestep")
    axes[1].set_title("Traffic (λ_t)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / outfile
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_combined_scaling_vs_ground_truth(benchmark: dict) -> None:
    """Side-by-side panels: one policy per column, ideal vs active only."""
    policies = benchmark["policies"]
    trajectories: dict[str, dict] = {}
    for key in SCALING_COMPARE_POLICIES:
        data = policies.get(key, {})
        if data.get("trajectory"):
            trajectories[key] = data["trajectory"]

    available = [k for k in SCALING_COMPARE_POLICIES if k in trajectories]
    if not available:
        print("No trajectories for combined scaling plot; skipping.")
        return

    ref_key = available[0]
    ideal = np.array(trajectories[ref_key]["ideal_replicas"])
    rps = np.array(trajectories[ref_key]["rps"])
    steps = np.arange(len(ideal))
    episode_seed = benchmark["seed"] + benchmark.get("trajectory_episode", 0)

    n_cols = len(available)
    fig = plt.figure(figsize=(4.2 * n_cols, 7.5))
    gs = fig.add_gridspec(
        2,
        n_cols,
        height_ratios=[1, 2.4],
        hspace=0.38,
        wspace=0.22,
    )

    ax_traffic = fig.add_subplot(gs[0, :])
    ax_traffic.plot(steps, rps, color="#7c3aed", linewidth=1.4)
    ax_traffic.fill_between(steps, rps, alpha=0.12, color="#7c3aed", step=None)
    ax_traffic.set_ylabel("RPS")
    ax_traffic.set_title(f"Traffic (λ_t) — episode seed={episode_seed}")
    ax_traffic.grid(True, alpha=0.25)
    ax_traffic.set_xlim(0, len(steps) - 1)

    y_max = float(np.max(ideal)) + 1.5
    for traj in trajectories.values():
        y_max = max(y_max, float(np.max(traj["active_replicas"])) + 1.5)

    policy_axes: list = []

    for col, key in enumerate(available):
        ax = fig.add_subplot(gs[1, col], sharex=ax_traffic)
        policy_axes.append(ax)
        if col > 0:
            plt.setp(ax.get_yticklabels(), visible=False)

        active = np.array(trajectories[key]["active_replicas"])
        n = min(len(active), len(ideal))
        gap = np.abs(active[:n] - ideal[:n])
        mae = float(np.mean(gap))

        ax.step(
            steps[:n],
            ideal[:n],
            where="post",
            color="#64748b",
            linewidth=1.8,
            linestyle="--",
            label="Ground truth",
            zorder=2,
        )
        ax.step(
            steps[:n],
            active[:n],
            where="post",
            color=POLICY_COLORS[key],
            linewidth=2.8,
            label="Active",
            zorder=3,
        )
        ax.fill_between(
            steps[:n],
            ideal[:n],
            active[:n],
            step="post",
            alpha=0.22,
            color=POLICY_COLORS[key],
            label="Gap",
        )

        ax.set_title(
            f"{POLICY_LABELS[key]}\nmean |active − ideal| = {mae:.2f}",
            fontsize=10,
        )
        ax.set_xlabel("Timestep")
        ax.set_ylim(0.5, y_max)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    policy_axes[0].set_ylabel("Replica count")
    fig.suptitle(
        "Scaling vs ground truth — one panel per policy (same episode)",
        fontsize=12,
        y=0.98,
    )

    out = FIGURES_DIR / "scaling_vs_ground_truth_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_training_metrics(
    mlflow_data: dict,
    *,
    algorithm: str,
    outfile: str,
    primary_color: str,
    entropy_key: str | None = "policy_entropy",
    secondary_key: str | None = None,
) -> None:
    metrics = mlflow_data.get("metrics") or {}
    if not metrics.get("episode_return_mean"):
        print(f"No MLflow episode_return_mean for {algorithm}; skipping {outfile}.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    steps = [s for s, _ in metrics["episode_return_mean"]]
    returns = [v for _, v in metrics["episode_return_mean"]]
    axes[0].plot(steps, returns, color=primary_color, linewidth=2, marker="o", markersize=3)
    axes[0].set_ylabel("episode_return_mean")
    axes[0].set_title(f"{algorithm} training metrics (MLflow)")
    axes[0].axhline(0, color="#999", linestyle="--", linewidth=0.8)
    axes[0].grid(True, alpha=0.3)

    if metrics.get("episode_len_mean"):
        lens = [v for _, v in metrics["episode_len_mean"]]
        axes[1].plot(steps[: len(lens)], lens, color="#7c3aed", linewidth=2)
        axes[1].set_ylabel("episode_len_mean")

    secondary = secondary_key and metrics.get(secondary_key)
    if secondary:
        axes[1].plot(
            [s for s, _ in secondary],
            [v for _, v in secondary],
            color="#ea580c",
            linewidth=1.5,
            alpha=0.8,
        )
        axes[1].set_ylabel(secondary_key.replace("_", " "))

    ent = entropy_key and metrics.get(entropy_key)
    if ent:
        ax2 = axes[0].twinx()
        ax2.plot([s for s, _ in ent], [v for _, v in ent], color="#ea580c", linewidth=1.5, alpha=0.8)
        ax2.set_ylabel(entropy_key, color="#ea580c")

    axes[1].set_xlabel("Training iteration")
    fig.tight_layout()
    out = FIGURES_DIR / outfile
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_ppo_training_metrics(mlflow_data: dict) -> None:
    _plot_training_metrics(
        mlflow_data,
        algorithm="PPO",
        outfile="ppo_training_metrics.png",
        primary_color="#2563eb",
        entropy_key="policy_entropy",
    )


def _plot_dqn_training_metrics(mlflow_data: dict) -> None:
    metrics = mlflow_data.get("metrics") or {}
    secondary = "td_error" if metrics.get("td_error") else "num_env_steps_sampled"
    _plot_training_metrics(
        mlflow_data,
        algorithm="DQN",
        outfile="dqn_training_metrics.png",
        primary_color="#9333ea",
        entropy_key="exploration_epsilon",
        secondary_key=secondary,
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Running benchmark rollouts...")
    benchmark = run_benchmark_suite(
        ppo_checkpoint_path=_REPO / "checkpoints" / "ppo" / "final",
        dqn_checkpoint_path=_REPO / "checkpoints" / "dqn" / "final",
        episodes=20,
        seed=42,
        fixed_replicas=4,
        mlflow_db=_REPO / "mlflow.db",
        trajectory_episode=0,
    )
    benchmark["fixed_replicas"] = 4

    summary = {
        "episodes": benchmark["episodes"],
        "seed": benchmark["seed"],
        "env_config": benchmark["env_config"],
        "fixed_replicas": benchmark["fixed_replicas"],
        "policies": {
            k: {
                "episode_return_mean": v.get("episode_return_mean"),
                "episode_return_std": v.get("episode_return_std"),
                "mean_cost_penalty": v.get("mean_cost_penalty"),
                "mean_latency_penalty": v.get("mean_latency_penalty"),
                "error": v.get("error"),
            }
            for k, v in benchmark["policies"].items()
        },
        "mlflow_run_ids": {
            "ppo": benchmark["mlflow"]["ppo"].get("run_id"),
            "dqn": benchmark["mlflow"]["dqn"].get("run_id"),
        },
    }
    json_path = RESULTS_DIR / "benchmark_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {json_path}")

    _plot_policy_comparison(benchmark)
    _plot_pareto_frontier(benchmark)
    _plot_scaling_vs_ground_truth(
        benchmark,
        "ppo",
        "ppo_scaling_vs_ground_truth.png",
        "PPO",
    )
    _plot_scaling_vs_ground_truth(
        benchmark,
        "dqn",
        "dqn_scaling_vs_ground_truth.png",
        "DQN",
    )
    _plot_combined_scaling_vs_ground_truth(benchmark)
    _plot_ppo_training_metrics(benchmark["mlflow"]["ppo"])
    _plot_dqn_training_metrics(benchmark["mlflow"]["dqn"])


if __name__ == "__main__":
    main()
