#!/usr/bin/env python3
"""
Generate comparison and training plots under results/.

Usage:
    uv sync --extra train --no-editable
    .venv/bin/python scripts/generate_results_plots.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from rl_inference_autoscaler.evaluation import run_benchmark_suite

RESULTS_DIR = _REPO / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIG_BENCHMARK = FIGURES_DIR / "benchmark"
FIG_TRAINING = FIGURES_DIR / "training"
FIG_SCALING = FIGURES_DIR / "scaling"
FIG_BASELINES = FIGURES_DIR / "baselines"

FIGURE_SUBDIRS = (FIG_BENCHMARK, FIG_TRAINING, FIG_SCALING, FIG_BASELINES)

POLICY_LABELS = {
    "ppo": "PPO (trained)",
    "dqn": "DQN (trained)",
    "greedy": "Greedy",
    "target_utilization": "Target util (HPA)",
    "fixed_replica": "Fixed replica",
    "do_nothing": "Do nothing",
}
POLICY_COLORS = {
    "ppo": "#2563eb",
    "dqn": "#9333ea",
    "greedy": "#16a34a",
    "target_utilization": "#0891b2",
    "fixed_replica": "#ca8a04",
    "do_nothing": "#dc2626",
}
PARETO_ORDER = [
    "ppo",
    "dqn",
    "greedy",
    "target_utilization",
    "fixed_replica",
    "do_nothing",
]
SCALING_COMPARE_POLICIES = ["ppo", "dqn", "greedy"]
SCALING_LINE_STYLES = {
    "ground_truth": {"color": "#111827", "linestyle": "-", "linewidth": 2.5, "alpha": 1.0},
    "ppo": {"color": "#2563eb", "linestyle": "--", "linewidth": 2.0, "alpha": 0.95},
    "dqn": {"color": "#9333ea", "linestyle": "-.", "linewidth": 2.0, "alpha": 0.95},
    "greedy": {"color": "#16a34a", "linestyle": ":", "linewidth": 2.4, "alpha": 0.95},
}


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
    out = FIG_BENCHMARK / "policy_comparison.png"
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


# Bottom-left of text box sits at point + offset → label reads above-right of dot.
_PARETO_ANNOTATE_OFFSET = (10, 8)


def _plot_pareto_frontier(benchmark: dict) -> None:
    policies = benchmark["policies"]
    fixed_n = benchmark.get("fixed_replicas", 4)
    labels = {
        **POLICY_LABELS,
        "ppo": "PPO",
        "dqn": "DQN",
        "fixed_replica": f"Fixed replica (n={fixed_n})",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    points: list[tuple[float, float]] = []

    for key in PARETO_ORDER:
        data = policies.get(key)
        if not data or "error" in data:
            continue
        cost = data["mean_cost_penalty"]
        latency = data["mean_latency_penalty"]
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
            labels[key],
            (cost, latency),
            textcoords="offset points",
            xytext=_PARETO_ANNOTATE_OFFSET,
            ha="left",
            va="bottom",
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
        0.02,
        "Lower-left is better.\nReturn = -(cost + latency) per episode.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        color="#444",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_BENCHMARK / "pareto_frontier.png"
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
    out = FIG_SCALING / outfile
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def _style_scaling_axes(ax, *, show_xlabel: bool = False) -> None:
    ax.grid(True, color="#e2e8f0", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11, colors="#475569")
    ax.yaxis.label.set_size(12)
    ax.yaxis.label.set_color("#1e293b")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if show_xlabel:
        ax.set_xlabel("Timestep", fontsize=12, color="#1e293b")
    else:
        ax.tick_params(labelbottom=False)


def _traffic_spike_intervals(
    rps: np.ndarray, *, percentile: float = 72.0
) -> list[tuple[int, int]]:
    """Contiguous timestep ranges where traffic is in the upper tail."""
    threshold = float(np.percentile(rps, percentile))
    high = rps >= threshold
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(high):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            intervals.append((start, i))
            start = None
    if start is not None:
        intervals.append((start, len(rps)))
    return intervals


def _annotate_scaling_story(
    ax_traffic,
    ax_replicas,
    steps: np.ndarray,
    rps: np.ndarray,
    ideal: np.ndarray,
    trajectories: dict[str, dict],
) -> None:
    """At most two annotations that explain the reliability vs cost tradeoff."""
    arrow = dict(arrowstyle="->", color="#64748b", lw=0.85, shrinkA=3, shrinkB=3)

    spike_t = int(np.argmax(rps))
    if rps[spike_t] > np.percentile(rps, 70):
        ax_traffic.annotate(
            "Traffic spike",
            xy=(spike_t, rps[spike_t]),
            xytext=(min(spike_t + 16, len(steps) - 1), rps[spike_t] * 0.88),
            fontsize=9,
            color="#6d28d9",
            arrowprops=arrow,
        )

    ppo_traj = trajectories.get("ppo")
    if ppo_traj is None:
        return
    active = np.array(ppo_traj["active_replicas"])
    ideal_falling = np.concatenate([[False], np.diff(ideal) < -0.5])
    delayed = (ideal < 4) & (active > ideal + 6) & ideal_falling
    if not np.any(delayed):
        return
    t0 = int(np.where(delayed)[0][-1])
    ax_replicas.annotate(
        "Delayed scale-down",
        xy=(t0, active[t0]),
        xytext=(max(t0 - 24, 0), active[t0] + 1.2),
        fontsize=9,
        color="#475569",
        arrowprops=arrow,
    )


def _plot_combined_scaling_vs_ground_truth(benchmark: dict) -> None:
    """Two-panel blog figure: traffic context and replicas vs ideal."""
    policies = benchmark["policies"]
    trajectories: dict[str, dict] = {}
    for key in SCALING_COMPARE_POLICIES:
        data = policies.get(key, {})
        if data.get("trajectory"):
            trajectories[key] = data["trajectory"]

    if not trajectories:
        print("No trajectories for combined scaling plot; skipping.")
        return

    ref_key = next(iter(trajectories))
    ideal = np.array(trajectories[ref_key]["ideal_replicas"])
    rps = np.array(trajectories[ref_key]["rps"])
    steps = np.arange(len(ideal))

    legend_labels = {"ppo": "PPO", "dqn": "DQN", "greedy": "Greedy"}
    replica_colors = {k: POLICY_COLORS[k] for k in SCALING_COMPARE_POLICIES}

    plot_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
    }

    with plt.rc_context(plot_rc):
        fig = plt.figure(figsize=(12, 8.5), facecolor="white")
        gs = fig.add_gridspec(
            2,
            1,
            height_ratios=[0.55, 3.2],
            hspace=0.14,
        )
        ax_traffic = fig.add_subplot(gs[0])
        ax_replicas = fig.add_subplot(gs[1], sharex=ax_traffic)

        fig.suptitle(
            "RL Autoscaling: Reliability vs Cost vs Ideal",
            fontsize=17,
            fontweight="bold",
            color="#0f172a",
            x=0.09,
            ha="left",
            y=0.97,
        )

        # (1) Traffic context with spike shading
        for start, end in _traffic_spike_intervals(rps):
            ax_traffic.axvspan(
                start,
                end,
                color="#ede9fe",
                alpha=0.55,
                zorder=0,
                linewidth=0,
            )
        ax_traffic.plot(steps, rps, color="#a78bfa", linewidth=0.85, alpha=0.85, zorder=2)
        ax_traffic.set_ylabel("Traffic (RPS)", fontsize=12)
        _style_scaling_axes(ax_traffic)

        spike_intervals = _traffic_spike_intervals(rps)

        # (2) Replica trajectories
        for start, end in spike_intervals:
            ax_replicas.axvspan(
                start,
                end,
                color="#ede9fe",
                alpha=0.35,
                zorder=0,
                linewidth=0,
            )
        ax_replicas.step(
            steps,
            ideal,
            where="post",
            label="Ideal policy",
            color="#111827",
            linewidth=3.4,
            zorder=6,
        )
        for key in SCALING_COMPARE_POLICIES:
            traj = trajectories.get(key)
            if not traj:
                continue
            active = np.array(traj["active_replicas"])
            ax_replicas.step(
                steps[: len(active)],
                active,
                where="post",
                label=legend_labels[key],
                color=replica_colors[key],
                linewidth=1.75,
                alpha=0.9,
                zorder=5 if key == "greedy" else 4,
            )

        ax_replicas.set_ylabel("Active GPU replicas", fontsize=12, fontweight="medium")
        replica_max = max(
            float(ideal.max()),
            *(
                float(np.max(traj["active_replicas"]))
                for traj in trajectories.values()
            ),
        )
        ax_replicas.set_ylim(0, int(np.ceil(replica_max)) + 1)
        ax_replicas.yaxis.set_major_locator(MaxNLocator(integer=True))
        _style_scaling_axes(ax_replicas, show_xlabel=True)

        handles, labels = ax_replicas.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.905),
            ncol=4,
            fontsize=11,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.8,
        )

        _annotate_scaling_story(
            ax_traffic, ax_replicas, steps, rps, ideal, trajectories
        )

        fig.text(
            0.09,
            0.02,
            "Ideal replicas use instantaneous RPS (no cold-start).",
            fontsize=9,
            color="#64748b",
            ha="left",
        )

        fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.08, hspace=0.16)
        out = FIG_SCALING / "scaling_vs_ground_truth_combined.png"
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
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
    out = FIG_TRAINING / outfile
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


def _plot_reward_components(benchmark: dict) -> None:
    """P6: stacked cost vs latency over episode (greedy trajectory)."""
    traj = benchmark["policies"].get("greedy", {}).get("trajectory")
    if not traj or "cost_penalty" not in traj:
        print("No component trajectory; skipping reward component plot.")
        return
    steps = np.arange(len(traj["cost_penalty"]))
    cost = np.array(traj["cost_penalty"])
    lat = np.array(traj["latency_penalty"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(steps, cost, lat, labels=["Cost", "Latency"], colors=["#2563eb", "#dc2626"], alpha=0.75)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Per-step penalty")
    ax.set_title("Greedy policy — reward components (episode 0)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = FIG_BASELINES / "reward_components_greedy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_training_overlay(mlflow_ppo: dict, mlflow_dqn: dict) -> None:
    """P5: PPO + DQN episode_return_mean on one chart."""
    ppo = (mlflow_ppo.get("metrics") or {}).get("episode_return_mean")
    dqn = (mlflow_dqn.get("metrics") or {}).get("episode_return_mean")
    if not ppo and not dqn:
        print("No MLflow curves for overlay; skipping.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    if ppo:
        ax.plot([s for s, _ in ppo], [v for _, v in ppo], label="PPO", color="#2563eb", marker="o", ms=3)
    if dqn:
        ax.plot([s for s, _ in dqn], [v for _, v in dqn], label="DQN", color="#9333ea", marker="o", ms=3)
    ax.axhline(0, color="#999", linestyle="--", lw=0.8)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("episode_return_mean")
    ax.set_title("PPO vs DQN training (MLflow)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_TRAINING / "training_curves_overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_b6_scatter(benchmark: dict) -> None:
    """B6: episode return vs cost/latency sums."""
    policies = benchmark["policies"]
    costs, lats, rets, names = [], [], [], []
    for key in PARETO_ORDER:
        p = policies.get(key)
        if not p or p.get("error"):
            continue
        costs.append(p["mean_cost_penalty"])
        lats.append(p["mean_latency_penalty"])
        rets.append(p["episode_return_mean"])
        names.append(POLICY_LABELS[key])
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(costs, rets, c=range(len(names)), cmap="tab10", s=80)
    axes[0].set_xlabel("Mean cost penalty")
    axes[0].set_ylabel("Episode return")
    axes[0].set_title("Return vs cost")
    axes[1].scatter(lats, rets, c=range(len(names)), cmap="tab10", s=80)
    axes[1].set_xlabel("Mean latency penalty")
    axes[1].set_ylabel("Episode return")
    axes[1].set_title("Return vs latency")
    for i, n in enumerate(names):
        axes[0].annotate(n, (costs[i], rets[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    fig.tight_layout()
    out = FIG_BENCHMARK / "return_vs_penalties_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_faceted_scaling(benchmark: dict, trajectory_episodes: list[int]) -> None:
    """P4: faceted scaling plots per policy."""
    n = len(SCALING_COMPARE_POLICIES)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, key in zip(axes, SCALING_COMPARE_POLICIES):
        traj = benchmark["policies"].get(key, {}).get("trajectory")
        if not traj:
            ax.set_title(f"{POLICY_LABELS[key]} (no data)")
            continue
        steps = np.arange(len(traj["ideal_replicas"]))
        ax.step(steps, traj["ideal_replicas"], where="post", color="#111827", label="Ground truth")
        ax.step(
            steps,
            traj["active_replicas"],
            where="post",
            color=POLICY_COLORS[key],
            linestyle="--",
            label="Active",
        )
        ax.set_ylabel("Replicas")
        ax.set_title(POLICY_LABELS[key])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Faceted scaling vs ground truth", y=1.01)
    fig.tight_layout()
    out = FIG_SCALING / "scaling_vs_ground_truth_faceted.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_fixed_replica_sweep(sweep: dict) -> None:
    """B2: fixed replica Pareto points."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for n, metrics in sorted(sweep.items(), key=lambda x: int(x[0])):
        ax.scatter(
            metrics["mean_cost_penalty"],
            metrics["mean_latency_penalty"],
            s=100,
            label=f"fixed n={n}",
        )
    ax.set_xlabel("Cost penalty")
    ax.set_ylabel("Latency penalty")
    ax.set_title("Fixed replica sweep (B2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_BASELINES / "fixed_replica_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--faceted", action="store_true", help="P4 faceted scaling plot")
    parser.add_argument("--trajectory-episodes", type=str, default="0,5,10", help="T6")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for sub in FIGURE_SUBDIRS:
        sub.mkdir(parents=True, exist_ok=True)
    traj_eps = [int(x) for x in args.trajectory_episodes.split(",") if x.strip()]

    print("Running benchmark rollouts...")
    benchmark = run_benchmark_suite(
        ppo_checkpoint_path=_REPO / "checkpoints" / "ppo" / "final",
        dqn_checkpoint_path=_REPO / "checkpoints" / "dqn" / "final",
        episodes=args.episodes,
        seed=args.seed,
        fixed_replicas=4,
        mlflow_db=_REPO / "mlflow.db",
        trajectory_episode=traj_eps[0] if traj_eps else 0,
    )
    benchmark["fixed_replicas"] = 4
    benchmark["trajectory_episodes"] = traj_eps

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
    _plot_training_overlay(benchmark["mlflow"]["ppo"], benchmark["mlflow"]["dqn"])
    _plot_reward_components(benchmark)
    _plot_b6_scatter(benchmark)
    if args.faceted:
        _plot_faceted_scaling(benchmark, traj_eps)

    # B2 fixed replica sweep
    from rl_inference_autoscaler.env import AutoscalerEnv
    from rl_inference_autoscaler.policies.baselines import evaluate_policy, fixed_replica_policy

    sweep: dict[str, dict] = {}
    for n in (2, 4, 8, 12):
        m = evaluate_policy(
            fixed_replica_policy,
            AutoscalerEnv(config={"traffic_mode": "auto", "initial_replicas": float(n)}),
            episodes=min(args.episodes, 30),
            seed=args.seed,
        )
        sweep[str(n)] = {
            "episode_return_mean": m["episode_return_mean"],
            "mean_cost_penalty": m["mean_cost_penalty"],
            "mean_latency_penalty": m["mean_latency_penalty"],
        }
    (_REPO / "results" / "experiments" / "b2_fixed_replica_sweep.json").write_text(
        json.dumps({"experiment": "B2", "sweep": sweep}, indent=2)
    )
    _plot_fixed_replica_sweep(sweep)

    # T6 extra trajectories stored in benchmark meta
    t6 = {"experiment": "T6", "episodes": []}
    for ep in traj_eps[1:]:
        b = run_benchmark_suite(
            episodes=ep + 1,
            seed=args.seed,
            trajectory_episode=ep,
            mlflow_db=_REPO / "mlflow.db",
        )
        for key in ("ppo", "dqn", "greedy"):
            traj = b["policies"].get(key, {}).get("trajectory")
            if traj:
                t6["episodes"].append({"episode_index": ep, "policy": key, "steps": len(traj["rps"])})
    (_REPO / "results" / "experiments" / "t6_plot_trajectories.json").write_text(
        json.dumps(t6, indent=2)
    )


if __name__ == "__main__":
    main()
