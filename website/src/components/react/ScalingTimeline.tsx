import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { dataUrl } from '../../lib/dataUrl';
import { chartTooltipStyle, colors } from '../../lib/theme';
import type { TrajectoriesData } from '../../types/benchmark';

const POLICY_OPTIONS = [
  { key: 'ppo', label: 'PPO' },
  { key: 'dqn', label: 'DQN' },
  { key: 'greedy', label: 'Greedy' },
] as const;

const CHART_MARGIN = { top: 8, right: 16, left: 4, bottom: 4 };

export default function ScalingTimeline() {
  const [raw, setRaw] = useState<TrajectoriesData | null>(null);
  const [policy, setPolicy] = useState<string>('ppo');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(dataUrl('data/trajectories.json'))
      .then((r) => r.json())
      .then(setRaw)
      .finally(() => setLoading(false));
  }, []);

  const chartData = useMemo(() => {
    const traj = raw?.policies[policy]?.trajectory;
    if (!traj) return [];
    return traj.rps.map((rps, i) => ({
      step: i,
      rps,
      active: traj.active_replicas[i],
      ideal: traj.ideal_replicas[i],
      action: traj.actions[i],
    }));
  }, [raw, policy]);

  const replicaMax = useMemo(() => {
    if (chartData.length === 0) return 20;
    const peak = Math.max(
      ...chartData.map((d) => Math.max(d.active, d.ideal)),
    );
    return Math.ceil(peak) + 1;
  }, [chartData]);

  if (loading) {
    return <p className="text-ink-light text-sm py-8 text-center">Loading trajectories…</p>;
  }

  if (!raw || chartData.length === 0) {
    return (
      <p className="text-ink-light text-sm py-8 text-center">
        No trajectory data. Run <code className="text-accent">npm run sync</code> after generating
        benchmarks.
      </p>
    );
  }

  const axisTick = { fill: colors.inkMuted, fontSize: 11 };

  return (
    <div className="chart-card not-prose">
      <div className="flex flex-wrap items-center gap-4 mb-4">
        <label className="text-sm text-ink-muted">
          Policy:{' '}
          <select
            value={policy}
            onChange={(e) => setPolicy(e.target.value)}
            className="ml-2 rounded-md border border-border bg-cream-card px-2 py-1 text-ink"
            aria-label="Select policy for scaling timeline"
          >
            {POLICY_OPTIONS.map((o) => (
              <option key={o.key} value={o.key}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <p className="text-sm text-ink-light">
          Episode {raw.trajectory_episode}, seed {raw.seed} · ideal = instantaneous capacity from
          RPS
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <ResponsiveContainer width="100%" height={168}>
          <LineChart data={chartData} margin={CHART_MARGIN}>
            <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
            <XAxis dataKey="step" tick={false} axisLine={false} height={8} />
            <YAxis
              tick={axisTick}
              width={48}
              label={{
                value: 'RPS',
                angle: -90,
                position: 'insideLeft',
                fill: colors.inkMuted,
                fontSize: 11,
                dx: 4,
              }}
            />
            <Tooltip contentStyle={chartTooltipStyle} />
            <Line
              type="monotone"
              dataKey="rps"
              name="RPS"
              stroke={colors.inkLight}
              dot={false}
              strokeWidth={1.5}
            />
          </LineChart>
        </ResponsiveContainer>

        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} margin={{ ...CHART_MARGIN, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
            <XAxis
              dataKey="step"
              tick={axisTick}
              label={{
                value: 'Timestep',
                position: 'insideBottom',
                offset: -2,
                fill: colors.inkMuted,
                fontSize: 11,
              }}
            />
            <YAxis
              tick={axisTick}
              width={48}
              allowDecimals={false}
              domain={[0, replicaMax]}
              label={{
                value: 'Replicas',
                angle: -90,
                position: 'insideLeft',
                fill: colors.inkMuted,
                fontSize: 11,
                dx: 4,
              }}
            />
            <Tooltip contentStyle={chartTooltipStyle} />
            <Legend wrapperStyle={{ color: colors.inkMuted, fontSize: 12 }} />
            <Line
              type="stepAfter"
              dataKey="active"
              name="Active replicas"
              stroke={colors.cardinal}
              dot={false}
              strokeWidth={2}
            />
            <Line
              type="stepAfter"
              dataKey="ideal"
              name="Ideal replicas"
              stroke={colors.cardinalLight}
              strokeDasharray="4 4"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
