import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { dataUrl } from '../../lib/dataUrl';
import { chartTooltipStyle, colors } from '../../lib/theme';
import type { TrainingCurvesData } from '../../types/training';

const PPO_COLOR = colors.cardinal;
const DQN_COLOR = '#9333ea';

interface ChartRow {
  iteration: number;
  ppo: number | null;
  dqn: number | null;
}

function mergeCurves(raw: TrainingCurvesData): ChartRow[] {
  const byIter = new Map<number, ChartRow>();

  for (const [step, value] of raw.ppo.episode_return_mean) {
    byIter.set(step, { iteration: step, ppo: value, dqn: null });
  }
  for (const [step, value] of raw.dqn.episode_return_mean) {
    const row = byIter.get(step) ?? { iteration: step, ppo: null, dqn: null };
    row.dqn = value;
    byIter.set(step, row);
  }

  return [...byIter.values()].sort((a, b) => a.iteration - b.iteration);
}

function TrainingTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { dataKey: string; value: number | null; color: string }[];
  label?: number;
}) {
  if (!active || payload == null || label == null) return null;

  const ppo = payload.find((p) => p.dataKey === 'ppo')?.value;
  const dqn = payload.find((p) => p.dataKey === 'dqn')?.value;

  return (
    <div style={chartTooltipStyle} className="text-sm px-3 py-2 shadow-sm">
      <p className="font-medium text-ink mb-1">Iteration {label}</p>
      {ppo != null && (
        <p style={{ color: PPO_COLOR }}>
          PPO: {ppo.toFixed(1)}
        </p>
      )}
      {dqn != null && (
        <p style={{ color: DQN_COLOR }}>
          DQN: {dqn.toFixed(1)}
        </p>
      )}
    </div>
  );
}

export default function TrainingCurvesChart() {
  const [raw, setRaw] = useState<TrainingCurvesData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(dataUrl('data/training_curves.json'))
      .then((r) => r.json())
      .then(setRaw)
      .finally(() => setLoading(false));
  }, []);

  const chartData = useMemo(() => (raw ? mergeCurves(raw) : []), [raw]);

  if (loading) {
    return (
      <p className="text-ink-light text-sm py-8 text-center">Loading training curves…</p>
    );
  }

  if (!raw || chartData.length === 0) {
    return (
      <p className="text-ink-light text-sm py-8 text-center">
        No training curve data. Run <code className="text-accent">npm run sync</code> after
        generating benchmarks.
      </p>
    );
  }

  const axisTick = { fill: colors.inkMuted, fontSize: 11 };

  return (
    <figure className="chart-card not-prose my-6">
      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={chartData} margin={{ top: 8, right: 16, left: 12, bottom: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
          <XAxis
            dataKey="iteration"
            tick={axisTick}
            label={{
              value: 'Training iteration',
              position: 'insideBottom',
              offset: -2,
              fill: colors.inkMuted,
              fontSize: 11,
            }}
          />
          <YAxis
            tick={axisTick}
            width={52}
            label={{
              value: 'Episode return (mean)',
              angle: -90,
              position: 'left',
              style: { textAnchor: 'middle' },
              fill: colors.inkMuted,
              fontSize: 11,
            }}
          />
          <Tooltip content={<TrainingTooltip />} />
          <Legend wrapperStyle={{ color: colors.inkMuted, fontSize: 12 }} />
          <ReferenceLine y={0} stroke={colors.inkLight} strokeDasharray="4 4" />
          <Line
            type="monotone"
            dataKey="ppo"
            name="PPO"
            stroke={PPO_COLOR}
            strokeWidth={2}
            dot={{ r: 3, strokeWidth: 0 }}
            activeDot={{ r: 5 }}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="dqn"
            name="DQN"
            stroke={DQN_COLOR}
            strokeWidth={2}
            dot={{ r: 3, strokeWidth: 0 }}
            activeDot={{ r: 5 }}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
      <figcaption className="text-sm text-ink-muted text-center mt-3">
        Episode return during training (hover a point to compare PPO and DQN at the same
        iteration)
      </figcaption>
    </figure>
  );
}
