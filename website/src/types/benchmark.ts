export interface PolicyMetrics {
  episode_return_mean: number;
  episode_return_std: number;
  mean_cost_penalty: number;
  mean_latency_penalty: number;
  error: string | null;
}

export interface BenchmarkSummary {
  episodes: number;
  seed: number;
  env_config: { traffic_mode: string };
  policies: Record<string, PolicyMetrics>;
}

export interface Trajectory {
  rps: number[];
  active_replicas: number[];
  ideal_replicas: number[];
  actions: number[];
  cost_penalty?: number[];
  latency_penalty?: number[];
}

export interface TrajectoriesData {
  seed: number;
  trajectory_episode: number;
  policies: Record<string, { trajectory: Trajectory }>;
}
