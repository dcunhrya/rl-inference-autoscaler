import { useEffect, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { dataUrl } from '../../lib/dataUrl';
import { chartTooltipStyle, colors } from '../../lib/theme';
import type { BenchmarkSummary } from '../../types/benchmark';

const LABELS: Record<string, string> = {
  ppo: 'PPO',
  dqn: 'DQN',
  greedy: 'Greedy',
  target_utilization: 'Target util',
  fixed_replica: 'Fixed (n=4)',
  do_nothing: 'Do nothing',
};

const ORDER = [
  'ppo',
  'dqn',
  'greedy',
  'target_utilization',
  'fixed_replica',
  'do_nothing',
];

export default function PolicyComparisonChart() {
  const [data, setData] = useState<{ name: string; return: number }[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(dataUrl('data/benchmark_summary.json'))
      .then((r) => r.json())
      .then((bench: BenchmarkSummary) => {
        const rows = ORDER.filter((k) => bench.policies[k]).map((key) => ({
          name: LABELS[key] ?? key,
          return: bench.policies[key].episode_return_mean,
        }));
        setData(rows);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p className="text-ink-light text-sm py-8 text-center">Loading benchmark data…</p>;
  }

  return (
    <div className="chart-card not-prose">
      <p className="text-sm text-ink-muted mb-4">
        Mean episode return (50 episodes). Higher (closer to zero) is better.
      </p>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 8, bottom: 64 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
          <XAxis
            dataKey="name"
            tick={{ fill: colors.inkMuted, fontSize: 11 }}
            angle={-35}
            textAnchor="end"
            height={70}
          />
          <YAxis tick={{ fill: colors.inkMuted, fontSize: 11 }} />
          <Tooltip
            contentStyle={chartTooltipStyle}
            formatter={(value: number) => [value.toFixed(1), 'Return']}
          />
          <Bar dataKey="return" fill={colors.cardinal} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
