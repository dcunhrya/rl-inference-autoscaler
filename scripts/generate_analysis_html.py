#!/usr/bin/env python3
"""Build results/analysis.html from benchmark + experiment JSON + figures."""

from __future__ import annotations

import json
import html
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
RESULTS = _REPO / "results"
FIGURES = RESULTS / "figures"
EXPERIMENTS = RESULTS / "experiments"


def _load_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _table_from_policies(policies: dict) -> str:
    rows = []
    order = [
        "ppo",
        "dqn",
        "greedy",
        "target_utilization",
        "fixed_replica",
        "do_nothing",
    ]
    labels = {
        "ppo": "PPO (trained)",
        "dqn": "DQN (trained)",
        "greedy": "Greedy (oracle-style)",
        "target_utilization": "Target utilization (HPA)",
        "fixed_replica": "Fixed replica (baseline)",
        "do_nothing": "Do nothing",
    }
    for key in order:
        if key not in policies:
            continue
        p = policies[key]
        if p.get("error"):
            rows.append(
                f"<tr><td>{labels[key]}</td><td colspan='4' class='err'>"
                f"{html.escape(str(p['error']))}</td></tr>"
            )
            continue
        rows.append(
            "<tr>"
            f"<td>{labels[key]}</td>"
            f"<td>{p.get('episode_return_mean', '—'):.2f}</td>"
            f"<td>{p.get('episode_return_std', 0):.2f}</td>"
            f"<td>{p.get('mean_cost_penalty', '—'):.2f}</td>"
            f"<td>{p.get('mean_latency_penalty', '—'):.2f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _img(rel: str, alt: str) -> str:
    path = RESULTS / rel
    if not path.is_file():
        return f"<p class='muted'>Missing: {html.escape(rel)}</p>"
    return f'<img src="{html.escape(rel)}" alt="{html.escape(alt)}" loading="lazy">'


def _section_experiments() -> str:
    index = _load_json(EXPERIMENTS / "index.json")
    if not index:
        return "<p class='muted'>No experiment outputs in results/experiments/.</p>"

    parts = ["<h2>Backlog experiments (pre–Phase 3)</h2>"]
    for entry in index.get("experiments", []):
        fname = entry.get("file", "")
        data = _load_json(EXPERIMENTS / fname) if fname else None
        if not data:
            continue
        exp_id = data.get("experiment", fname)
        parts.append(f"<h3>{html.escape(str(exp_id))}</h3>")
        parts.append(
            f"<pre class='json-block'>{html.escape(json.dumps(data, indent=2)[:8000])}</pre>"
        )
    return "\n".join(parts)


def main() -> None:
    benchmark = _load_json(RESULTS / "benchmark_summary.json") or {}
    b3 = _load_json(EXPERIMENTS / "b3_oracle_gap.json")
    t2 = _load_json(EXPERIMENTS / "t2_csv_benchmark.json")

    env_cfg = benchmark.get("env_config", {})
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    oracle_section = ""
    if b3 and b3.get("policies"):
        oracle_rows = []
        for name, metrics in b3["policies"].items():
            oracle_rows.append(
                "<tr>"
                f"<td>{html.escape(name.upper())}</td>"
                f"<td>{metrics.get('mean_abs_gap', 0):.3f}</td>"
                f"<td>{metrics.get('max_abs_gap', 0):.3f}</td>"
                f"<td>{metrics.get('rmse', 0):.3f}</td>"
                "</tr>"
            )
        oracle_section = f"""
        <h2>Ground-truth gap (B3)</h2>
        <p>Mean |active replicas − ideal replicas| on a single held-out trajectory (seed 42).
        Greedy uses perfect next-step RPS; ideal is instantaneous capacity.</p>
        <table>
          <thead><tr><th>Policy</th><th>Mean |ΔN|</th><th>Max |ΔN|</th><th>RMSE</th></tr></thead>
          <tbody>{''.join(oracle_rows)}</tbody>
        </table>
        """

    csv_note = ""
    if t2:
        csv_policies = {
            k: {
                "episode_return_mean": v.get("episode_return_mean"),
                "mean_latency_penalty": v.get("mean_latency_penalty"),
            }
            for k, v in (t2.get("policies") or {}).items()
            if isinstance(v, dict)
        }
        csv_note = f"""
        <h2>CSV-only evaluation (T2)</h2>
        <pre class='json-block'>{html.escape(json.dumps(csv_policies, indent=2))}</pre>
        """

    policies = benchmark.get("policies", {})
    rl = [
        (k, policies[k])
        for k in ("ppo", "dqn")
        if k in policies and not policies[k].get("error")
    ]
    best_rl = (
        min(rl, key=lambda x: x[1]["episode_return_mean"])[0].upper()
        if rl
        else "N/A"
    )
    greedy_ret = policies.get("greedy", {}).get("episode_return_mean")
    ppo_ret = policies.get("ppo", {}).get("episode_return_mean")
    dqn_ret = policies.get("dqn", {}).get("episode_return_mean")
    rl_vs_greedy = ""
    if greedy_ret is not None and ppo_ret is not None:
        rl_vs_greedy = (
            f"PPO return is {abs(ppo_ret / greedy_ret):.1%} of greedy magnitude "
            f"({ppo_ret:.0f} vs {greedy_ret:.0f}); "
        )
        if dqn_ret is not None:
            rl_vs_greedy += (
                f"DQN is {abs(dqn_ret / greedy_ret):.1%} ({dqn_ret:.0f}). "
            )

    analysis_text = f"""
    <h2>Analysis</h2>
    <ul>
      <li><strong>Best RL policy (this run):</strong> {best_rl} by episode return on
      <code>traffic_mode={html.escape(str(env_cfg.get('traffic_mode', 'auto')))}</code>.
      {rl_vs_greedy}</li>
      <li><strong>PPO</strong> ({ppo_ret:.0f} mean return): balances replica cost
      ({policies.get('ppo', {}).get('mean_cost_penalty', 0):.0f}) vs latency
      ({policies.get('ppo', {}).get('mean_latency_penalty', 0):.0f}) on the updated trace.</li>
      <li><strong>DQN</strong> ({dqn_ret:.0f} mean return): discrete Q-learning; cost
      {policies.get('dqn', {}).get('mean_cost_penalty', 0):.0f}, latency
      {policies.get('dqn', {}).get('mean_latency_penalty', 0):.0f}.</li>
      <li><strong>Greedy</strong> ({greedy_ret:.0f}): perfect next-step RPS but cold-start
      limits effective capacity — large latency penalty on high-RPS CSV replay.</li>
      <li><strong>Fixed replica (n={benchmark.get('fixed_replicas', 4)})</strong>:
      {policies.get('fixed_replica', {}).get('episode_return_mean', 0):.0f} — static capacity
      cannot follow spikes in the new data.</li>
      <li><strong>Do nothing</strong>:
      {policies.get('do_nothing', {}).get('episode_return_mean', 0):.0f} — worst overload.</li>
      <li><strong>Ground truth</strong> in scaling plots: ideal replica count from instantaneous
      RPS (no boot delay). RL policies still show replica gap vs ideal (see B3 table).</li>
    </ul>
    """

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RL Inference Autoscaler — Results Analysis</title>
  <style>
    :root {{
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #94a3b8;
      --accent: #3b82f6;
      --border: #334155;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
      margin: 0;
      padding: 2rem max(1rem, 5vw);
      max-width: 1100px;
      margin-inline: auto;
    }}
    h1 {{ font-size: 1.75rem; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1.25rem; margin-top: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 0.35rem; }}
    h3 {{ font-size: 1.05rem; color: var(--muted); }}
    .meta {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.95rem; }}
    th, td {{ border: 1px solid var(--border); padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: var(--card); }}
    tr:nth-child(even) {{ background: rgba(255,255,255,0.03); }}
    .err {{ color: #f87171; }}
    img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 1rem 0; background: #fff; }}
    .grid {{ display: grid; gap: 1.5rem; }}
    @media (min-width: 720px) {{ .grid-2 {{ grid-template-columns: 1fr 1fr; }} }}
    pre.json-block {{
      background: var(--card);
      padding: 1rem;
      overflow-x: auto;
      font-size: 0.8rem;
      border-radius: 8px;
      max-height: 320px;
    }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
  <h1>RL Inference Autoscaler — Results</h1>
  <p class="meta">Generated {generated} · traffic_mode={html.escape(str(env_cfg.get('traffic_mode', 'auto')))}
  · episodes={benchmark.get('episodes', '—')} · seed={benchmark.get('seed', '—')}</p>

  <h2>Benchmark summary</h2>
  <p>Higher episode return is better (closer to zero). Cost and latency penalties are per-episode means.</p>
  <table>
    <thead>
      <tr><th>Policy</th><th>Return (mean)</th><th>Return (std)</th><th>Cost penalty</th><th>Latency penalty</th></tr>
    </thead>
    <tbody>
      {_table_from_policies(benchmark.get('policies', {}))}
    </tbody>
  </table>

  {analysis_text}
  {oracle_section}
  {csv_note}

  <h2>Figures</h2>
  <div class="grid">
    <figure>{_img('figures/policy_comparison.png', 'Policy comparison')}</figure>
    <figure>{_img('figures/pareto_frontier.png', 'Pareto frontier')}</figure>
  </div>
  <figure>{_img('figures/scaling_vs_ground_truth_combined.png', 'Scaling vs ground truth')}</figure>
  <div class="grid grid-2">
    <figure>{_img('figures/ppo_scaling_vs_ground_truth.png', 'PPO scaling')}</figure>
    <figure>{_img('figures/dqn_scaling_vs_ground_truth.png', 'DQN scaling')}</figure>
  </div>
  <div class="grid grid-2">
    <figure>{_img('figures/ppo_training_metrics.png', 'PPO training')}</figure>
    <figure>{_img('figures/dqn_training_metrics.png', 'DQN training')}</figure>
  </div>
  <figure>{_img('figures/training_curves_overlay.png', 'PPO vs DQN training overlay')}</figure>
  <figure>{_img('figures/scaling_vs_ground_truth_faceted.png', 'Faceted scaling')}</figure>
  <figure>{_img('figures/fixed_replica_sweep.png', 'Fixed replica sweep')}</figure>
  <figure>{_img('figures/return_vs_penalties_scatter.png', 'Return vs penalties')}</figure>
  <figure>{_img('figures/reward_components_greedy.png', 'Reward components')}</figure>

  {_section_experiments()}
</body>
</html>
"""

    out = RESULTS / "analysis.html"
    out.write_text(doc)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
