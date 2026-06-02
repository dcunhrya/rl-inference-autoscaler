#!/usr/bin/env python3
"""
Complete pre-Phase-3 experiments backlog (light local Ray).

Usage:
  .venv/bin/python scripts/complete_backlog.py --skip-train
  .venv/bin/python scripts/complete_backlog.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
PY = _REPO / ".venv" / "bin" / "python"
ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "1",
    "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
}


def run(cmd: list[str], *, desc: str) -> None:
    print(f"\n=== {desc} ===")
    subprocess.run(cmd, cwd=_REPO, env=ENV, check=True)


def train_jobs(skip_train: bool) -> None:
    if skip_train:
        return
    base = ["--local", "--mlflow-tracking-uri", "sqlite:///mlflow.db"]
    plans: list[tuple[str, list[str], str]] = [
        ("train.py", ["--iterations", "100"], "A1 PPO 100 iter"),
        ("train_dqn.py", ["--iterations", "100"], "A2 DQN 100 iter"),
        (
            "train.py",
            [
                "--iterations",
                "80",
                "--traffic-mode",
                "csv",
                "--checkpoint-dir",
                "checkpoints/ppo_csv",
                "--experiment-name",
                "autoscaler-ppo-csv",
            ],
            "T1 PPO CSV",
        ),
        (
            "train_dqn.py",
            [
                "--iterations",
                "80",
                "--traffic-mode",
                "csv",
                "--checkpoint-dir",
                "checkpoints/dqn_csv",
                "--experiment-name",
                "autoscaler-dqn-csv",
            ],
            "T1 DQN CSV",
        ),
        (
            "train.py",
            [
                "--iterations",
                "80",
                "--traffic-mode",
                "synthetic",
                "--checkpoint-dir",
                "checkpoints/ppo_synth",
                "--experiment-name",
                "autoscaler-ppo-synth",
            ],
            "T3 train synthetic",
        ),
        (
            "train.py",
            [
                "--iterations",
                "50",
                "--churn-penalty-delta",
                "0.05",
                "--pending-penalty-eta",
                "0.02",
                "--checkpoint-dir",
                "checkpoints/ppo_mdp",
                "--experiment-name",
                "autoscaler-ppo-mdp",
            ],
            "R4/R5 MDP penalties",
        ),
        (
            "train.py",
            ["--iterations", "40", "--lr", "1e-4", "--checkpoint-dir", "checkpoints/ppo_lr1e4", "--experiment-name", "autoscaler-ppo-lr1e4"],
            "A3 PPO lr=1e-4",
        ),
        (
            "train_dqn.py",
            ["--iterations", "40", "--lr", "1e-3", "--checkpoint-dir", "checkpoints/dqn_lr1e3", "--experiment-name", "autoscaler-dqn-lr1e3"],
            "A4 DQN lr=1e-3",
        ),
    ]
    for script, extra, desc in plans:
        run([str(PY), str(_REPO / script), *base, *extra], desc=desc)

    for steps, tag in [(50, "50"), (100, "100"), (200, "200")]:
        run(
            [
                str(PY),
                str(_REPO / "train.py"),
                *base,
                "--iterations",
                "40",
                "--max-steps-per-episode",
                str(steps),
                "--checkpoint-dir",
                f"checkpoints/ppo_curriculum_{tag}",
                "--experiment-name",
                f"autoscaler-ppo-curriculum-{tag}",
            ],
            desc=f"A7 curriculum max_steps={steps}",
        )


def export_mlflow_summary() -> None:
    import sqlite3

    db = _REPO / "mlflow.db"
    out: dict = {"experiments": []}
    if not db.is_file():
        return
    conn = sqlite3.connect(db)
    try:
        cur = conn.execute("SELECT name FROM experiments")
        for (name,) in cur.fetchall():
            cur2 = conn.execute(
                "SELECT experiment_id FROM experiments WHERE name = ?", (name,)
            )
            exp_id = cur2.fetchone()[0]
            runs = conn.execute(
                """
                SELECT run_uuid FROM runs
                WHERE experiment_id = ? AND status = 'FINISHED'
                ORDER BY start_time DESC LIMIT 5
                """,
                (exp_id,),
            ).fetchall()
            out["experiments"].append({"name": name, "run_ids": [r[0] for r in runs]})
    finally:
        conn.close()
    path = _REPO / "results" / "experiments" / "p2_mlflow_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


def main() -> None:
    skip_train = "--skip-train" in sys.argv
    train_jobs(skip_train)

    run([str(PY), "scripts/run_prephase3_experiments.py"], desc="Eval experiments")
    run(
        [str(PY), "scripts/generate_results_plots.py", "--episodes", "50", "--faceted"],
        desc="Plots P1",
    )
    run([str(PY), "scripts/run_eval.py"], desc="E2 eval")
    run([str(PY), "scripts/generate_analysis_html.py"], desc="analysis.html")

    exp_dir = _REPO / "results" / "experiments"
    rows = []
    for f in sorted(exp_dir.glob("*.json")):
        if f.name == "index.json":
            continue
        data = json.loads(f.read_text())
        rows.append({"file": f.name, "experiment": data.get("experiment")})
    (_REPO / "results" / "sweep_results_table.json").write_text(
        json.dumps({"experiments": rows}, indent=2)
    )
    export_mlflow_summary()

    modal_status = "dry_run_ok"
    try:
        subprocess.run(
            [
                "uv",
                "run",
                "--extra",
                "train",
                "--extra",
                "modal",
                "modal",
                "run",
                "src/rl_inference_autoscaler/modal_train.py",
            ],
            cwd=_REPO,
            env=ENV,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        modal_status = f"failed: {exc}"

    (_REPO / "results" / "experiments" / "a8_a9_modal.json").write_text(
        json.dumps(
            {
                "experiment": "A8_A9",
                "status": modal_status,
                "note": "Modal image build may fail on dependency pins; use local checkpoints",
            },
            indent=2,
        )
    )

    from rl_inference_autoscaler.training.config import validate_dqn_config, validate_ppo_config

    (_REPO / "results" / "experiments" / "a5_ppo_new_stack.json").write_text(
        json.dumps(
            {
                "experiment": "A5",
                "note": "PPO old API stack; DQN new stack (single eval path for DQN)",
                "ppo_validate": validate_ppo_config(),
                "dqn_validate": validate_dqn_config(),
            },
            indent=2,
            default=str,
        )
    )
    (_REPO / "results" / "experiments" / "a6_alternatives.json").write_text(
        json.dumps(
            {
                "experiment": "A6",
                "status": "deferred",
                "note": "SAC/A2C/Rainbow omitted; PPO/DQN are primary algorithms",
            },
            indent=2,
        )
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
