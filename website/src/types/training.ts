export interface TrainingCurvesData {
  ppo: { run_id: string | null; episode_return_mean: [number, number][] };
  dqn: { run_id: string | null; episode_return_mean: [number, number][] };
}
