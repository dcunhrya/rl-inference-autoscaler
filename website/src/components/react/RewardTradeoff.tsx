import { useMemo, useState } from 'react';
import { colors } from '../../lib/theme';

/** Toy scenario: fixed load vs capacity for educational reward decomposition. */
export default function RewardTradeoff() {
  const [alpha, setAlpha] = useState(0.1);
  const [beta, setBeta] = useState(1.0);
  const [gamma, setGamma] = useState(0.05);
  const [replicas, setReplicas] = useState(6);
  const [rps, setRps] = useState(280);
  const throughputPerReplica = 50;
  const queueDepth = 40;

  const { cost, overload, queue, total } = useMemo(() => {
    const capacity = replicas * throughputPerReplica;
    const overloadRps = Math.max(0, rps - capacity);
    const costPenalty = alpha * replicas;
    const latencyPenalty = beta * (overloadRps + gamma * queueDepth);
    return {
      cost: costPenalty,
      overload: beta * overloadRps,
      queue: beta * gamma * queueDepth,
      total: -(costPenalty + latencyPenalty),
    };
  }, [alpha, beta, gamma, replicas, rps, queueDepth]);

  const maxBar = Math.max(cost, overload + queue, 1);

  return (
    <div className="chart-card not-prose space-y-6">
      <p className="text-sm text-ink-muted">
        Adjust reward weights and a toy load scenario. Total reward{' '}
        <span className="text-accent font-mono">R = −(α·N + β·overload + β·γ·queue)</span>.
      </p>

      <div className="grid sm:grid-cols-2 gap-6">
        <Slider
          label="α (cost per replica)"
          value={alpha}
          min={0.01}
          max={0.5}
          step={0.01}
          onChange={setAlpha}
        />
        <Slider
          label="β (latency weight)"
          value={beta}
          min={0.1}
          max={3}
          step={0.1}
          onChange={setBeta}
        />
        <Slider
          label="γ (queue weight)"
          value={gamma}
          min={0}
          max={0.2}
          step={0.01}
          onChange={setGamma}
        />
        <Slider
          label="Active replicas N"
          value={replicas}
          min={1}
          max={20}
          step={1}
          onChange={setReplicas}
        />
        <Slider label="Request rate λ (RPS)" value={rps} min={50} max={500} step={10} onChange={setRps} />
      </div>

      <div className="rounded-lg bg-cream-muted/80 border border-border p-4 text-sm space-y-2">
        <p className="text-ink-muted">
          Capacity: <strong className="text-ink">{capacity(replicas, throughputPerReplica)}</strong>{' '}
          RPS · Overload:{' '}
          <strong className="text-ink">
            {Math.max(0, rps - replicas * throughputPerReplica).toFixed(0)}
          </strong>{' '}
          RPS
        </p>
        <p className="text-accent font-mono text-lg">Episode-step reward ≈ {total.toFixed(2)}</p>
      </div>

      <div className="space-y-3">
        <Bar label="Cost (α·N)" value={cost} max={maxBar} color={colors.cardinal} />
        <Bar label="Overload (β·max(0, λ−μN))" value={overload} max={maxBar} color={colors.cardinalLight} />
        <Bar label="Queue (β·γ·queue)" value={queue} max={maxBar} color="#C4A574" />
      </div>
    </div>
  );
}

function capacity(n: number, mu: number) {
  return n * mu;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block text-sm text-ink-muted">
      <span className="flex justify-between mb-1">
        <span>{label}</span>
        <span className="font-mono text-accent">{value.toFixed(step < 1 ? 2 : 0)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[#8C1515]"
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
      />
    </label>
  );
}

function Bar({
  label,
  value,
  max,
  color,
}: {
  label: string;
  value: number;
  max: number;
  color: string;
}) {
  const pct = (value / max) * 100;
  return (
    <div>
      <div className="flex justify-between text-xs text-ink-light mb-1">
        <span>{label}</span>
        <span>{value.toFixed(2)}</span>
      </div>
      <div className="h-3 rounded-full bg-cream-muted overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-200"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}
