# Experiments Backlog

Living checklist for the **RL inference autoscaler** project.  
**Phase 3 (deployment) is out of scope here.** Pre–Phase 3 items below are marked done after the full local backlog run (Jun 2026).

**Related docs:** [phase1.md](phase1.md) · [phase2.md](phase2.md) · [project_goal.md](project_goal.md) · [best_run.md](best_run.md) · [results/README.md](../results/README.md)

---

## Done (baseline)

- [x] Phase 1 simulator: cold start, queue, synthetic/CSV traffic
- [x] PPO / DQN local training (`--local`, light CPU)
- [x] Modal wrappers (dry-run OK; full Modal image build may fail on dep pins)
- [x] Benchmark suite + plots + `results/analysis.html`
- [x] Comparison: PPO, DQN, greedy, target utilization, fixed replica, do nothing

---

## 1. Reward & environment

| ID | Status | Notes |
|----|--------|-------|
| R1 | [x] | `results/experiments/r1_reward_sweep.json` |
| R2 | [x] | Presets eval `r2_r3_presets.json` |
| R3 | [x] | Same |
| R4 | [x] | Env `churn_penalty_delta`; eval + `checkpoints/ppo_mdp` train |
| R5 | [x] | Env `pending_penalty_eta`; combined with R4 train |
| R6 | [x] | Env util band; `r6_util_band.json` |
| R7 | [x] | Per-step `cost_penalty` / `latency_penalty` in `info`; MLflow env params |
| R8 | [x] | `--reward-mode` on `train.py` / `train_dqn.py` |

---

## 2. Traffic & evaluation

| ID | Status | Notes |
|----|--------|-------|
| T1 | [x] | `checkpoints/ppo_csv`, `checkpoints/dqn_csv` |
| T2 | [x] | `t2_csv_benchmark.json`, `t2_multi_seed.json` |
| T3 | [x] | `t3_held_out.json` (synth train → CSV eval) |
| T4 | [x] | Updated `data/traffic_trace.csv` via `auto`/`csv` |
| T5 | [x] | `t5_spike_stress.json` |
| T6 | [x] | Plot script + `t6_multi_trajectory.json` |
| T7 | [x] | PPO/DQN/greedy `t7_cold_start.json` |

---

## 3. Algorithms & training

| ID | Status | Notes |
|----|--------|-------|
| A1 | [x] | PPO 100 iter |
| A2 | [x] | DQN 100 iter |
| A3 | [x] | `checkpoints/ppo_lr1e4` (40 iter) |
| A4 | [x] | `checkpoints/dqn_lr1e3` (40 iter) |
| A5 | [x] | Documented: PPO old stack, DQN new — `a5_ppo_new_stack.json` |
| A6 | [x] | Deferred (SAC/A2C) — `a6_alternatives.json` |
| A7 | [x] | Curriculum runs `checkpoints/ppo_curriculum_{50,100,200}` |
| A8 | [~] | Modal dry-run attempted; image build failed locally |
| A9 | [~] | Same as A8 |
| A10 | [x] | `a10_seed_sensitivity.json`, `a10_rl_seeds.json` |

---

## 4. Baselines & analysis

| ID | Status | Notes |
|----|--------|-------|
| B1 | [x] | `target_utilization_policy` in benchmark + fixed test |
| B2 | [x] | `b2_fixed_replica_sweep.json` + figure |
| B3 | [x] | `b3_oracle_gap.json` |
| B4 | [x] | `b4_action_distribution.json` |
| B5 | [x] | 50-episode `b5_extended_benchmark.json` |
| B6 | [x] | `return_vs_penalties_scatter.png` |

---

## 5. Results, plots & tracking

| ID | Status | Notes |
|----|--------|-------|
| P1 | [x] | `scripts/generate_results_plots.py` |
| P2 | [x] | `results/experiments/p2_mlflow_summary.json` |
| P3 | [x] | `results/sweep_results_table.json` |
| P4 | [x] | `--faceted` → `scaling_vs_ground_truth_faceted.png` |
| P5 | [x] | `training_curves_overlay.png` |
| P6 | [x] | `reward_components_greedy.png` |

---

## 6. Phase 3 — deployment (not started)

| ID | Experiment |
|----|------------|
| D1–D5 | Checkpoint export, Ray Serve, replay, shadow, A/B |

---

## 7. Engineering

| ID | Status | Notes |
|----|--------|-------|
| E1 | [x] | Parametrized reward tests in `test_autoscaler_env.py` |
| E2 | [x] | `scripts/run_eval.py` + `eval.json` |
| E3 | [x] | `checkpoints/{algo}/final` |
| E4 | [x] | [best_run.md](best_run.md) |

---

## Quick commands

```bash
# Full backlog (training + eval + plots) — ~1h+ on M1 Max
.venv/bin/python scripts/complete_backlog.py

# Eval/plots only
.venv/bin/python scripts/complete_backlog.py --skip-train

# Report
open results/analysis.html
```

---

## Latest benchmark (50 ep, seed 42, auto)

| Policy | Return | Cost | Latency |
|--------|--------|------|---------|
| PPO | −692.2 | 299.4 | 392.8 |
| DQN | −901.9 | 263.2 | 638.7 |
| Greedy | −17,566.2 | 98.3 | 17,467.9 |
| Fixed (n=4) | −46,820.6 | 80.0 | 46,740.6 |
| Do nothing | −82,099.0 | 20.0 | 82,079.0 |

See `results/benchmark_summary.json` and `results/analysis.html` for full analysis.
