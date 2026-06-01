# Phase 1: Environment Engineering Walkthrough

To build the simulator, we will use the `gymnasium` API. This environment needs to track the state of the cluster, apply the actions chosen by the RL agent, and return the calculated reward and next state.

Implementation for Phase 1.

## 1. The Python Implementation

Create a file named `autoscaler_env.py` and implement the following class. This script sets up the observation space, action space, and the core simulation loop.

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AutoscalerEnv(gym.Env):
    """
    Custom Environment for a Dynamic Inference Autoscaler.
    Action Space: Discrete(3) -> 0: Scale Down, 1: Do Nothing, 2: Scale Up
    Observation Space: Box(4) -> [RPS, Utilization, Active Replicas, RPS_Delta]
    """
    
    def __init__(self, config=None):
        super(AutoscalerEnv, self).__init__()
        
        # Hardware & Penalty Config
        self.max_replicas = 20
        self.max_rps = 1000.0
        self.throughput_per_replica = 50.0  # requests per second a single node can handle
        self.cost_alpha = 0.1               # cost penalty per active node
        self.latency_beta = 1.0             # severe penalty for dropped/delayed requests
        
        # 0: remove node (-1), 1: do nothing (0), 2: add node (+1)
        self.action_space = spaces.Discrete(3)
        
        # State definition: [current_rps, avg_utilization, num_replicas, rps_delta]
        # Low and High bounds for the observation space
        low = np.array([0.0, 0.0, 1.0, -self.max_rps], dtype=np.float32)
        high = np.array([self.max_rps, 1.0, self.max_replicas, self.max_rps], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        self.current_step = 0
        self.max_steps_per_episode = 200
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize cluster with random baseline traffic and 1 node
        initial_rps = np.random.uniform(10, 50)
        initial_utilization = initial_rps / self.throughput_per_replica
        num_replicas = 1.0
        rps_delta = 0.0
        
        self.state = np.array([initial_rps, initial_utilization, num_replicas, rps_delta], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        
        current_rps, avg_util, num_replicas, _ = self.state
        
        # 1. Apply Action (Scale up, down, or do nothing)
        # Map Discrete(3) [0, 1, 2] to changes [-1, 0, 1]
        replica_change = action - 1 
        new_replicas = np.clip(num_replicas + replica_change, 1.0, self.max_replicas)
        
        # 2. Simulate Traffic Fluctuation (The Environment Dynamics)
        # In a real setup, you would load historical traffic data here.
        # For simulation, we inject a random walk with occasional spikes.
        spike = np.random.choice([0, 100], p=[0.95, 0.05]) 
        noise = np.random.normal(0, 5)
        new_rps = np.clip(current_rps + noise + spike, 0.0, self.max_rps)
        new_rps_delta = new_rps - current_rps
        
        # 3. Calculate New Hardware Utilization
        total_capacity = new_replicas * self.throughput_per_replica
        new_avg_util = np.clip(new_rps / total_capacity, 0.0, 1.0)
        
        # 4. Calculate Reward
        # Penalty for active hardware
        cost_penalty = self.cost_alpha * new_replicas
        
        # Penalty for dropped/queued requests (when RPS exceeds total capacity)
        dropped_requests = max(0.0, new_rps - total_capacity)
        latency_penalty = self.latency_beta * dropped_requests
        
        reward = -(cost_penalty + latency_penalty)
        
        # 5. Update State
        self.state = np.array([new_rps, new_avg_util, new_replicas, new_rps_delta], dtype=np.float32)
        
        # 6. Check Termination
        terminated = False
        truncated = self.current_step >= self.max_steps_per_episode
        
        # Info dictionary for tracking metrics in Ray
        info = {
            "dropped_requests": dropped_requests,
            "active_replicas": new_replicas
        }
        
        return self.state, float(reward), terminated, truncated, info