# Best run (local, updated trace)

Generated after completing the pre–Phase 3 experiments backlog.

## Training

| Algorithm | Iterations | Traffic | Checkpoint | MLflow experiment |
|-----------|------------|---------|------------|-------------------|
| PPO | 100 | auto (CSV trace) | `checkpoints/ppo/final` | `autoscaler-ppo` |
| DQN | 100 | auto | `checkpoints/dqn/final` | `autoscaler-dqn` |
| PPO (CSV train) | 80 | csv | `checkpoints/ppo_csv/final` | `autoscaler-ppo-csv` |
| DQN (CSV train) | 80 | csv | `checkpoints/dqn_csv/final` | `autoscaler-dqn-csv` |
| PPO (synthetic) | 80 | synthetic | `checkpoints/ppo_synth/final` | `autoscaler-ppo-synth` |

Settings: `--local`, `num_env_runners=0`, `OMP_NUM_THREADS=4`, `reward_mode=balanced` (default).

## Benchmark (50 episodes, seed=42, `traffic_mode=auto`)

| Policy | Return (mean) | Cost | Latency |
|--------|---------------|------|---------|
| **PPO** | **−692.2** | 299.4 | 392.8 |
| DQN | −901.9 | 263.2 | 638.7 |
| Target utilization | −9,519.0 | 170.9 | 9,348.1 |
| Greedy | −17,566.2 | 98.3 | 17,467.9 |
| Fixed (n=4) | −46,820.6 | 80.0 | 46,740.6 |
| Do nothing | −82,099.0 | 20.0 | 82,079.0 |

**Recommendation:** deploy evaluation uses `checkpoints/ppo/final` on the bundled production-style CSV trace.

## Reproduce

```bash
uv sync --extra train --no-editable
.venv/bin/python train.py --local --iterations 100 --mlflow-tracking-uri sqlite:///mlflow.db
.venv/bin/python train_dqn.py --local --iterations 100 --mlflow-tracking-uri sqlite:///mlflow.db
.venv/bin/python scripts/generate_results_plots.py --episodes 50 --faceted
.venv/bin/python scripts/run_eval.py
open results/analysis.html
```
