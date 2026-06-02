# Phase 2: Distributed Training Walkthrough

Phase 2 trains a **PPO** policy on `AutoscalerEnv` using **Ray RLlib**, with optional execution on **Modal**. Training logic is split so Modal stays isolated from the core Ray path.

**Status: infrastructure complete** — run training when ready; CI/tests validate config without launching Modal jobs.

## 1. Architecture

```text
┌─────────────────┐     parallel rollouts      ┌──────────────┐
│  AutoscalerEnv  │ ◄──────────────────────────│ Env runners  │
│  (Gymnasium)    │                            │ (Ray RLlib)  │
└────────┬────────┘                            └──────┬───────┘
         │                                              │
         │  observations / rewards                        ▼
         │                                     ┌──────────────┐
         └────────────────────────────────────►│ PPO learner  │
                                               │ 2×256 MLP    │
                                               └──────┬───────┘
                                                      │
                      ┌───────────────────────────────┼───────────────────────────────┐
                      ▼                               ▼                               ▼
               checkpoints/                    MLflow (optional)              Modal Volume
               (local Ray)                     episode_return_mean,            /checkpoints
                                              policy_entropy
```

| Module | Role |
|--------|------|
| `train_config.py` | Shared `TrainingSettings` / `DQNSettings`, `build_ppo_config`, `build_dqn_config` |
| `train_common.py` | Shared Ray init, MLflow setup, metric extraction |
| `train_ray.py` | `ray.init()`, PPO `algo.train()`, checkpoints, MLflow |
| `train_dqn_ray.py` | Same for DQN |
| `modal_train.py` | **Only** Modal `App`, image, volume, `train_on_modal(algorithm=...)` |
| `train.py` / `train_dqn.py` | Thin local CLIs |

## 2. Data During Training

Rollout data is **generated inside the simulator**, not downloaded:

- **Training (default):** `traffic_mode=auto` → synthetic random walk + spikes (exploration-friendly).
- **Evaluation / Modal optional:** mount `data/traffic_trace.csv` for reproducible traces.

Ray env runners each instantiate `AutoscalerEnv` with the `env_config` passed through `PPOConfig().environment(env_config=...)`.

## 3. Ray Setup (Local)

### Install

```bash
uv sync --extra train --extra dev --no-editable
```

On macOS, use `--no-editable` so the package is installed into `.venv` (editable `__editable__*.pth` files are hidden and skipped by Python).

### Dry-run (no `ray.init`, no training loop)

Validates RLlib config builds:

```bash
uv run python -m rl_inference_autoscaler.train_ray --dry-run
```

### Full local training

**Important (uv + Ray):** Install train deps and pass the extra when using `uv run`:

```bash
uv sync --extra train --no-editable
uv run --extra train python train.py --iterations 50 --num-env-runners 4
# or
uv run --extra train train-ray --iterations 50 --num-env-runners 4
```

On macOS, if workers crash with `ModuleNotFoundError: No module named 'ray'`, either use the command above or the single-process flag (training code also sets `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` automatically):

```bash
uv run --extra train python train.py --local --iterations 10
```

Alternative without `uv run` (uses project `.venv` directly):

```bash
uv sync --extra train
.venv/bin/python train.py --iterations 50 --num-env-runners 2
```

Checkpoints: `checkpoints/final/` (Ray RLlib checkpoint format).

### MLflow (optional)

```bash
uv run python train.py --iterations 20 --mlflow-tracking-uri file:./mlruns
uv run mlflow ui --backend-store-uri ./mlruns
```

Logged metrics per iteration: `episode_return_mean`, `episode_len_mean`, `policy_entropy`.

### Ray cluster behavior

`train_ray.run_training()`:

1. `register_env()` — registers `Autoscaler-v0` with Gymnasium  
2. `ray.init(ignore_reinit_error=True)` — local head node; for multi-node, start Ray cluster first and set `RAY_ADDRESS`  
3. `PPOConfig().environment(AutoscalerEnv).env_runners(num_env_runners=…).build()`  
4. Loop `algo.train()` → log → `algo.save(checkpoints/final)`  
5. `ray.shutdown()`  

### PPO hyperparameters (defaults in `TrainingSettings`)

| Parameter | Default |
|-----------|---------|
| `num_env_runners` | 4 (8 on Modal) |
| `train_batch_size` | 4000 |
| `lr` | 3e-4 |
| `gamma` | 0.99 |
| `clip_param` | 0.2 |
| `entropy_coeff` | 0.01 |
| Actor/Critic MLP | `[256, 256]` ReLU |

## 4. Modal Setup (Cloud)

All Modal code is in **`src/rl_inference_autoscaler/modal_train.py`** — do not add Modal imports to `train_ray.py`.

### One-time

```bash
uv sync --extra train --extra modal
modal setup
```

### Dry-run (default — no job submitted)

```bash
uv run modal run src/rl_inference_autoscaler/modal_train.py
```

Prints the exact command to run real training.

### Launch training (when you are ready)

```bash
uv run modal run src/rl_inference_autoscaler/modal_train.py --no-dry-run --iterations 100
```

- **Image:** Debian slim + Ray RLlib, torch, mlflow, project source  
- **Volume:** `autoscaler-rl-checkpoints` → `/checkpoints`  
- **Worker:** calls `run_training()` from `train_ray.py` with `checkpoint_dir=/checkpoints`  

Download checkpoints:

```bash
modal volume get autoscaler-rl-checkpoints final ./checkpoints/from_modal
```

## 5. Testing Infrastructure (No Modal Job)

```bash
uv sync --extra train --extra modal --extra dev
uv run pytest -v
```

| Test file | What it verifies |
|-----------|------------------|
| `tests/test_autoscaler_env.py` | Cold start, queue, CSV/synthetic traffic, baselines |
| `tests/test_train_infra.py` | Gym registration, PPO/DQN config build, Ray dry-run, Modal module shape |

Ray training tests use `--dry-run` / `validate_ppo_config()` so CI does not need GPUs or long runs.

## 6. Baseline Comparison

Before trusting a policy, compare against the heuristic in `baselines.py`:

```python
from rl_inference_autoscaler.baselines import evaluate_baseline
print(evaluate_baseline(episodes=5, seed=0))
```

After training, evaluate the checkpoint with RLlib’s eval API or a custom rollout script (Phase 3).

## 7. Ray vs Modal

| | **Local Ray** | **Modal** |
|---|---------------|-----------|
| Entry | `train.py` / `train-ray` | `modal run …/modal_train.py` |
| `ray.init` | On your machine | Inside Modal container |
| Parallelism | `--num-env-runners` (CPU cores) | `--num-env-runners 8` default |
| Checkpoints | `./checkpoints` | Volume `autoscaler-rl-checkpoints` |
| Cost | Your hardware | Modal usage billing |

Same `run_training()` implements both paths.

## 8. DQN Training

DQN mirrors the PPO stack with parallel modules. **Note:** DQN uses RLlib's **new API stack** (episode replay buffer); PPO remains on the old stack. Ray 2.54+ rejects the old-stack DQN replay buffer combo.

| Module | Role |
|--------|------|
| `train_config.py` | `DQNSettings` + `build_dqn_config()` |
| `train_common.py` | Shared Ray init, MLflow, metric extraction |
| `train_dqn_ray.py` | `run_dqn_training()`, local CLI |
| `train_dqn.py` | Thin local CLI wrapper |
| `modal_train.py` | `--algorithm dqn` → `run_dqn_training()` |

### Local DQN

```bash
uv sync --extra train --no-editable
uv run --extra train python train_dqn.py --dry-run
uv run --extra train python train_dqn.py --iterations 50 --num-env-runners 4
uv run --extra train python train_dqn.py --local --iterations 10
uv run --extra train python train_dqn.py --mlflow-tracking-uri sqlite:///mlflow.db
```

Checkpoints: `checkpoints/dqn/final/`

### Modal DQN

```bash
uv run modal run src/rl_inference_autoscaler/modal_train.py --no-dry-run --algorithm dqn --iterations 100
modal volume get autoscaler-rl-checkpoints dqn/final ./checkpoints/dqn/from_modal
```

### DQN hyperparameters (defaults in `DQNSettings`)

| Parameter | Default |
|-----------|---------|
| `num_env_runners` | 4 (8 on Modal) |
| `train_batch_size` | 32 |
| `lr` | 5e-4 |
| `gamma` | 0.99 |
| `replay_buffer_capacity` | 50_000 |
| `learning_starts` | 1_000 |
| `target_update_freq` | 500 |
| `double_q` / `dueling` | True / True |
| Q-network MLP | `[256, 256]` ReLU |

MLflow experiment: `autoscaler-dqn`. Logged metrics: `episode_return_mean`, `episode_len_mean`, `td_error`, `num_env_steps_sampled`, `exploration_epsilon`.

### Results plots

After training PPO and/or DQN, regenerate comparison plots:

```bash
.venv/bin/python scripts/generate_results_plots.py
```

Produces `results/figures/dqn_training_metrics.png`, `dqn_scaling_vs_ground_truth.png`, and includes DQN in policy comparison / Pareto plots when a checkpoint exists.

## 9. Next: Phase 3

1. Load checkpoint via `Algorithm.from_checkpoint` / `get_module()`  
2. Ray Serve + FastAPI: metrics in → scale action out  
3. Compare served policy vs `target_utilization_policy` on live or replayed metrics  

See `project_info/project_goal.md`.
