# Phase 1: Environment Engineering Walkthrough

Phase 1 delivers a **Gymnasium** simulator for a Kubernetes-style inference autoscaler. The agent observes cluster metrics, chooses discrete scale actions, and receives a reward that trades off **replica cost** vs **latency / overload**.

**Status: complete** — implementation lives in `src/rl_inference_autoscaler/autoscaler_env.py` with traffic generation in `traffic.py`.

## 1. MDP Summary

| Component | Definition |
|-----------|------------|
| **State** \(S_t\) | `[λ_t, μ_t, N_t, δ_λ_t]` — RPS, utilization, **active** replicas, RPS delta |
| **Action** \(A_t\) | Discrete `{0, 1, 2}` → scale down, hold, scale up (`{-1, 0, +1}` replicas) |
| **Reward** \(R_t\) | `- α·N_t - β·(overload + drops + γ·queue_depth)` |

- **α** (`cost_alpha`): cost per active replica  
- **β** (`latency_beta`): penalty for overload and dropped work  
- **γ** (`queue_penalty_gamma`): extra penalty for queue backlog (wait proxy)  
- **μ** (`throughput_per_replica`): max RPS per replica  

## 2. Where the Data Comes From

The environment does **not** require external APIs for training. Request rates come from `TrafficGenerator` (`traffic.py`):

| Mode | Source | When to use |
|------|--------|-------------|
| **`auto`** (default) | Uses `data/traffic_trace.csv` if present; else **synthetic** | Training & CI |
| **`synthetic`** | Drifting baseline + noise + multi-step bursts (8% per step) | Training when no CSV; bursty GPU-like load |
| **`csv`** | `data/traffic_trace.csv` (200 steps, ~10–16 bursts) | Reproducible eval or real traces |

To use **production metrics**, export time-series RPS to CSV and set:

```python
AutoscalerEnv(config={"traffic_mode": "csv", "traffic_csv_path": "/path/to/trace.csv"})
```

The bundled `data/traffic_trace.csv` is a **200-step** GPU-style trace (drifting baseline, jitter, many multi-step bursts). Regenerate with:

```bash
python scripts/generate_traffic_trace.py
```

## 3. Realism: Cold Start & Queue

Unlike the initial sketch, scale-up is **not** instantaneous:

1. **Cold start** — `+1` enqueues a pod boot for `cold_start_steps` (default 3). Until ready, capacity does not increase. Scale-down removes pending boots first, then active replicas (min 1).
2. **Queue** — Each step, arrivals (`new_rps`) plus backlog are served up to `active_replicas × throughput_per_replica`. Remaining work stays in `queue_depth`. Overflow beyond `max_queue` is dropped and penalized.

`info` on each step includes `queue_depth`, `pending_replicas`, `dropped_requests`, `overload_rps`.

## 4. Python API

```python
from rl_inference_autoscaler import AutoscalerEnv, register_env

register_env()  # optional: gym.make("Autoscaler-v0")

env = AutoscalerEnv(config={
    "cold_start_steps": 3,
    "traffic_mode": "auto",
    "max_steps_per_episode": 200,
})
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)  # action in {0,1,2}
```

### Configuration keys

| Key | Default | Meaning |
|-----|---------|---------|
| `max_replicas` | 20 | Upper bound on active + pending capacity |
| `throughput_per_replica` | 50 | RPS capacity per replica |
| `cost_alpha` | 0.1 | Replica cost weight |
| `latency_beta` | 1.0 | Overload / drop penalty |
| `queue_penalty_gamma` | 0.05 | Queue depth penalty |
| `cold_start_steps` | 3 | Timesteps until new replica is active |
| `max_queue` | 500 | Max backlog before drops |
| `traffic_mode` | `auto` | `auto` / `synthetic` / `csv` |
| `traffic_csv_path` | `data/traffic_trace.csv` | Override CSV location |

## 5. Validation

```bash
uv sync
uv run pytest tests/test_autoscaler_env.py -v
uv run python main.py   # one-episode baseline smoke run
```

## 6. Reference Implementation

The canonical code is in the repository (not duplicated inline here):

- `src/rl_inference_autoscaler/autoscaler_env.py` — `AutoscalerEnv`
- `src/rl_inference_autoscaler/traffic.py` — traffic sources
- `src/rl_inference_autoscaler/baselines.py` — target-utilization heuristic for comparison

Next: **[Phase 2 — Distributed Training](phase2.md)**.
