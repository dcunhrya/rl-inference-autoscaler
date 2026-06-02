"""Gymnasium environment for the inference autoscaler MDP."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_inference_autoscaler.env.reward import apply_reward_mode
from rl_inference_autoscaler.env.traffic import TrafficGenerator, TrafficMode


class AutoscalerEnv(gym.Env):
    """
    Dynamic inference autoscaler simulator.

    Action space (Discrete 3): 0 scale down, 1 hold, 2 scale up.
    Observation (Box 4): [RPS, utilization, active_replicas, rps_delta].

    Scale-up requests enter a cold-start pipeline; scale-down removes capacity
    immediately. Excess load accumulates in a queue before drops are counted.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        cfg = apply_reward_mode(dict(config or {}))

        self.max_replicas = int(cfg.get("max_replicas", 20))
        self.max_rps = float(cfg.get("max_rps", 1000.0))
        self.throughput_per_replica = float(cfg.get("throughput_per_replica", 50.0))
        self.cost_alpha = float(cfg.get("cost_alpha", 0.1))
        self.latency_beta = float(cfg.get("latency_beta", 1.0))
        self.queue_penalty_gamma = float(cfg.get("queue_penalty_gamma", 0.05))
        self.cold_start_steps = int(cfg.get("cold_start_steps", 3))
        self.max_queue = float(cfg.get("max_queue", 500.0))
        self.max_steps_per_episode = int(cfg.get("max_steps_per_episode", 200))
        self.initial_replicas = float(cfg.get("initial_replicas", 1.0))
        # R4–R6 optional reward terms (default off for backward compatibility)
        self.churn_penalty_delta = float(cfg.get("churn_penalty_delta", 0.0))
        self.pending_penalty_eta = float(cfg.get("pending_penalty_eta", 0.0))
        self.util_band_low = float(cfg.get("util_band_low", 0.6))
        self.util_band_high = float(cfg.get("util_band_high", 0.8))
        self.util_band_penalty_zeta = float(cfg.get("util_band_penalty_zeta", 0.0))
        self.reward_mode = cfg.get("reward_mode")

        traffic_mode: TrafficMode = cfg.get("traffic_mode", "auto")
        csv_path = cfg.get("traffic_csv_path")
        self.traffic = TrafficGenerator(
            mode=traffic_mode,
            max_rps=self.max_rps,
            csv_path=csv_path,
            spike_magnitude=float(cfg.get("spike_magnitude", 100.0)),
            spike_probability=float(cfg.get("spike_probability", 0.05)),
            noise_std=float(cfg.get("noise_std", 5.0)),
        )

        self.action_space = spaces.Discrete(3)
        low = np.array([0.0, 0.0, 1.0, -self.max_rps], dtype=np.float32)
        high = np.array(
            [self.max_rps, 1.0, float(self.max_replicas), self.max_rps],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.current_step = 0
        self.state: np.ndarray | None = None
        self._rng = np.random.default_rng()
        self._active_replicas = 1.0
        self._pending_boots: list[int] = []
        self._queue_depth = 0.0
        self._prev_active_replicas = 1.0

    def _compute_reward(
        self,
        *,
        action: int,
        overload: float,
        dropped_requests: float,
        queue_depth: float,
        utilization: float,
        pending_count: int,
    ) -> tuple[float, dict[str, float]]:
        cost_penalty = self.cost_alpha * self._active_replicas
        latency_penalty = self.latency_beta * (
            overload + dropped_requests + self.queue_penalty_gamma * queue_depth
        )
        churn_penalty = 0.0
        if self.churn_penalty_delta > 0.0 and int(action) != 1:
            churn_penalty = self.churn_penalty_delta
        pending_penalty = self.pending_penalty_eta * float(pending_count)
        util_penalty = 0.0
        if self.util_band_penalty_zeta > 0.0:
            if utilization < self.util_band_low:
                util_penalty = self.util_band_penalty_zeta * (
                    self.util_band_low - utilization
                )
            elif utilization > self.util_band_high:
                util_penalty = self.util_band_penalty_zeta * (
                    utilization - self.util_band_high
                )
        total = cost_penalty + latency_penalty + churn_penalty + pending_penalty + util_penalty
        reward = -total
        components = {
            "cost_penalty": cost_penalty,
            "latency_penalty": latency_penalty,
            "churn_penalty": churn_penalty,
            "pending_penalty": pending_penalty,
            "util_penalty": util_penalty,
            "reward": reward,
        }
        return reward, components

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.current_step = 0
        self._active_replicas = float(
            np.clip(self.initial_replicas, 1.0, float(self.max_replicas))
        )
        self._pending_boots = []
        self._queue_depth = 0.0
        self._prev_active_replicas = self._active_replicas

        initial_rps = self.traffic.reset(self._rng)
        capacity = self._active_replicas * self.throughput_per_replica
        utilization = float(np.clip(initial_rps / max(capacity, 1e-6), 0.0, 1.0))
        self.state = np.array(
            [initial_rps, utilization, self._active_replicas, 0.0],
            dtype=np.float32,
        )
        info = {
            "traffic_mode": self.traffic.resolved_mode,
            "queue_depth": self._queue_depth,
            "pending_replicas": len(self._pending_boots),
        }
        return self.state, info

    def _apply_scaling_action(self, action: int) -> None:
        replica_change = int(action) - 1
        if replica_change > 0:
            for _ in range(replica_change):
                if (
                    self._active_replicas + len(self._pending_boots)
                    < self.max_replicas
                ):
                    self._pending_boots.append(self.cold_start_steps)
        elif replica_change < 0:
            for _ in range(-replica_change):
                if self._pending_boots:
                    self._pending_boots.pop()
                elif self._active_replicas > 1.0:
                    self._active_replicas -= 1.0

    def _advance_cold_starts(self) -> None:
        ready = 0
        remaining: list[int] = []
        for steps_left in self._pending_boots:
            if steps_left <= 1:
                ready += 1
            else:
                remaining.append(steps_left - 1)
        self._pending_boots = remaining
        self._active_replicas = float(
            np.clip(
                self._active_replicas + ready,
                1.0,
                float(self.max_replicas),
            )
        )

    def _process_queue(self, arrivals: float, capacity: float) -> tuple[float, float]:
        """Serve arrivals + backlog; return (dropped, post-step queue depth)."""
        backlog = self._queue_depth + arrivals
        served = min(backlog, capacity)
        self._queue_depth = max(0.0, backlog - served)
        overflow = max(0.0, self._queue_depth - self.max_queue)
        if overflow > 0.0:
            self._queue_depth -= overflow
        return overflow, self._queue_depth

    def step(self, action: int):
        self.current_step += 1
        current_rps, _, _, _ = self.state  # type: ignore[misc]

        self._apply_scaling_action(int(action))
        self._advance_cold_starts()

        new_rps = self.traffic.next_rps(self._rng, float(current_rps))
        new_rps_delta = new_rps - float(current_rps)

        capacity = self._active_replicas * self.throughput_per_replica
        dropped_requests, queue_depth = self._process_queue(new_rps, capacity)

        overload = max(0.0, new_rps - capacity)
        utilization = float(np.clip(new_rps / max(capacity, 1e-6), 0.0, 1.0))
        pending_count = len(self._pending_boots)
        reward, components = self._compute_reward(
            action=int(action),
            overload=overload,
            dropped_requests=dropped_requests,
            queue_depth=queue_depth,
            utilization=utilization,
            pending_count=pending_count,
        )
        self._prev_active_replicas = self._active_replicas
        self.state = np.array(
            [
                new_rps,
                utilization,
                self._active_replicas,
                new_rps_delta,
            ],
            dtype=np.float32,
        )

        terminated = False
        truncated = self.current_step >= self.max_steps_per_episode
        info = {
            "dropped_requests": dropped_requests,
            "overload_rps": overload,
            "active_replicas": self._active_replicas,
            "pending_replicas": pending_count,
            "queue_depth": queue_depth,
            "traffic_mode": self.traffic.resolved_mode,
            "utilization": utilization,
            **components,
        }
        return self.state, float(reward), terminated, truncated, info
