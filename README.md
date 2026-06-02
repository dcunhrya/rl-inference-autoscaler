# RL Inference Autoscaler

Reinforcement-learning autoscaler for ML inference workloads: a **Gymnasium** cluster simulator (Phase 1), **Ray RLlib PPO** training (Phase 2), and (planned) **Ray Serve** deployment (Phase 3).

## Project layout

```text
src/rl_inference_autoscaler/
  autoscaler_env.py   # Phase 1 MDP simulator
  traffic.py          # Synthetic + CSV request-rate data
  baselines.py        # Heuristic policy for comparison
  train_config.py     # Shared PPO / DQN RLlib settings
  train_common.py     # Shared Ray init, MLflow, metrics
  train_ray.py        # PPO training (local or Modal worker)
  train_dqn_ray.py    # DQN training (local or Modal worker)
  modal_train.py      # Modal App — keep separate from train_ray
data/
  traffic_trace.csv   # Optional reproducible RPS trace
train.py              # Thin CLI for local PPO training
train_dqn.py          # Thin CLI for local DQN training
project_info/
  project_goal.md
  phase1.md
  phase2.md
tests/
```

## Quick start

```bash
# Core simulator
uv sync
uv run pytest tests/test_autoscaler_env.py -v
uv run python main.py

# Training (use --no-editable on macOS — editable .pth files are skipped as "hidden")
uv sync --extra train --no-editable
uv sync --extra train --extra modal --extra dev --no-editable   # full stack

uv run pytest -v
.venv/bin/python train.py --dry-run
```

## Phase 1 — Environment

- **State:** `[RPS, utilization, active_replicas, rps_delta]`
- **Actions:** `0` scale down, `1` hold, `2` scale up
- **Dynamics:** cold-start delay on scale-up, queue buildup, overload penalties

**Traffic data** (no external download required for training):

| Mode | Source |
|------|--------|
| `synthetic` | Random walk + noise + 5% spikes |
| `csv` / `auto` | `data/traffic_trace.csv` if present, else synthetic |

Replace the CSV with exported production RPS for realistic replay. Details: [project_info/phase1.md](project_info/phase1.md).

## Phase 2 — Training

### Local Ray (RLlib PPO)

```bash
uv sync --extra train
# Must pass --extra train with uv run so Ray workers see ray/rllib
uv run --extra train python train.py --iterations 50 --num-env-runners 4
# Mac-friendly debug mode (no separate env-runner processes):
uv run --extra train python train.py --local --iterations 10
uv run --extra train python train.py --mlflow-tracking-uri file:./mlruns
```

If you see `ModuleNotFoundError: No module named 'rl_inference_autoscaler'`, run `uv sync --extra train --no-editable` (macOS skips hidden `__editable__*.pth` files) or use `train.py` / `main.py` from the repo root (they add `src/` to the path).

If you see `ModuleNotFoundError: No module named 'ray'` in `(raylet)` logs, Ray packaged the project without train deps — use `--extra train` or `.venv/bin/python train.py` after `uv sync --extra train --no-editable`.

Checkpoints: `checkpoints/ppo/final/` (legacy fallback: `checkpoints/final/`)

### Local Ray (RLlib DQN)

```bash
uv run --extra train python train_dqn.py --dry-run
uv run --extra train python train_dqn.py --iterations 50 --num-env-runners 4
uv run --extra train python train_dqn.py --local --iterations 10
uv run --extra train python train_dqn.py --mlflow-tracking-uri sqlite:///mlflow.db
```

Checkpoints: `checkpoints/dqn/final/`

### Modal (when you choose to run)

Modal code is **only** in `modal_train.py`. Default CLI is a dry-run:

```bash
uv sync --extra train --extra modal
modal setup
uv run modal run src/rl_inference_autoscaler/modal_train.py
# actual job:
uv run modal run src/rl_inference_autoscaler/modal_train.py --no-dry-run --iterations 100
# DQN on Modal:
uv run modal run src/rl_inference_autoscaler/modal_train.py --no-dry-run --algorithm dqn --iterations 100
```

Full walkthrough: [project_info/phase2.md](project_info/phase2.md).

## Phase 3 — Not yet implemented

- Export policy from Ray checkpoint
- Ray Serve + FastAPI endpoint for live metrics → scaling action

See [project_info/project_goal.md](project_info/project_goal.md).

## Results plots

```bash
uv sync --extra train --no-editable
.venv/bin/python scripts/generate_results_plots.py
```

Outputs: `results/figures/policy_comparison.png`, `pareto_frontier.png`, `ppo_scaling_vs_ground_truth.png`, `dqn_scaling_vs_ground_truth.png`, `ppo_training_metrics.png`, `dqn_training_metrics.png`, `results/benchmark_summary.json`.

## Development

```bash
uv run pytest -v
uv run python -m rl_inference_autoscaler.train_ray --dry-run
```

**Note:** `gymnasium` and `pandas` are pinned for Ray RLlib / MLflow compatibility (`gymnasium>=1.2.2,<1.3`, `pandas>=2.2,<3`). Raise these when upstream supports newer releases.

## Follow-ups you may want

- **W&B** instead of MLflow — add `wandb` to `[project.optional-dependencies.train]` and log in `train_ray._log_result`
- **Stronger traces** — replace `data/traffic_trace.csv` with cluster exports
- **Phase 3** — Serve deployment and policy eval script
