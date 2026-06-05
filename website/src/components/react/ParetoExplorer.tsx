import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import { dataUrl } from '../../lib/dataUrl';
import { chartTooltipStyle, colors, policyColors } from '../../lib/theme';
import type { BenchmarkSummary } from '../../types/benchmark';

const LABELS: Record<string, string> = {
  ppo: 'PPO',
  dqn: 'DQN',
  greedy: 'Greedy',
  target_utilization: 'Target util',
  fixed_replica: 'Fixed (n=4)',
  do_nothing: 'Do nothing',
};

interface Point {
  key: string;
  name: string;
  cost: number;
  latency: number;
  return: number;
}

export default function ParetoExplorer() {
  const [points, setPoints] = useState<Point[]>([]);
  const [highlightPareto, setHighlightPareto] = useState(true);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(dataUrl('data/benchmark_summary.json'))
      .then((r) => r.json())
      .then((bench: BenchmarkSummary) => {
        const pts: Point[] = Object.entries(bench.policies).map(([key, m]) => ({
          key,
          name: LABELS[key] ?? key,
          cost: m.mean_cost_penalty,
          latency: m.mean_latency_penalty,
          return: m.episode_return_mean,
        }));
        setPoints(pts);
      })
      .finally(() => setLoading(false));
  }, []);

  const paretoKeys = useMemo(() => {
    const sorted = [...points].sort((a, b) => a.cost - b.cost);
    const frontier = new Set<string>();
    let bestLatency = Infinity;
    for (const p of sorted) {
      if (p.latency < bestLatency) {
        frontier.add(p.key);
        bestLatency = p.latency;
      }
    }
    return frontier;
  }, [points]);

  if (loading) {
    return <p className="text-ink-light text-sm py-8 text-center">Loading…</p>;
  }

  return (
    <div className="chart-card not-prose">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <p className="text-sm text-ink-muted">
          Cost vs latency trade-off (mean penalties per episode)
        </p>
        <label className="flex items-center gap-2 text-sm text-ink cursor-pointer">
          <input
            type="checkbox"
            checked={highlightPareto}
            onChange={(e) => setHighlightPareto(e.target.checked)}
            className="rounded border-border text-accent focus:ring-accent"
          />
          Highlight Pareto-efficient policies
        </label>
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <ScatterChart margin={{ top: 12, right: 24, bottom: 48, left: 16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
          <XAxis
            type="number"
            dataKey="cost"
            name="Cost"
            tick={{ fill: colors.inkMuted, fontSize: 11 }}
            label={{ value: 'Cost penalty', position: 'bottom', fill: colors.inkMuted, offset: 24 }}
          />
          <YAxis
            type="number"
            dataKey="latency"
            name="Latency"
            tick={{ fill: colors.inkMuted, fontSize: 11 }}
            label={{
              value: 'Latency penalty',
              angle: -90,
              position: 'insideLeft',
              fill: colors.inkMuted,
            }}
          />
          <ZAxis range={[80, 400]} />
          <Tooltip
            cursor={{ strokeDasharray: '3 3', stroke: colors.chartGrid }}
            contentStyle={chartTooltipStyle}
            formatter={(value: number, name: string) => [
              value.toFixed(1),
              name === 'latency' ? 'Latency' : name === 'cost' ? 'Cost' : name,
            ]}
            labelFormatter={(_, payload) => {
              const p = payload?.[0]?.payload as Point | undefined;
              return p ? `${p.name} (return: ${p.return.toFixed(0)})` : '';
            }}
          />
          {points.map((p) => (
            <Scatter
              key={p.key}
              name={p.name}
              data={[p]}
              fill={
                highlightPareto && paretoKeys.has(p.key)
                  ? policyColors[p.key] ?? colors.cardinal
                  : highlightPareto
                    ? colors.inkLight
                    : policyColors[p.key] ?? colors.cardinal
              }
              stroke={highlightPareto && paretoKeys.has(p.key) ? colors.cardinalDark : 'none'}
              strokeWidth={2}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
      <ul className="flex flex-wrap gap-3 mt-4 text-xs text-ink-muted">
        {points.map((p) => (
          <li key={p.key} className="flex items-center gap-1.5">
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ background: policyColors[p.key] ?? colors.cardinal }}
            />
            {p.name}
            {highlightPareto && paretoKeys.has(p.key) ? (
              <span className="text-accent font-medium">(Pareto)</span>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}
