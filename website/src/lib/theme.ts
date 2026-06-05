/** Stanford-inspired palette (cardinal red + cream). */
export const colors = {
  cream: '#FAF7F2',
  creamCard: '#FFFDF9',
  creamMuted: '#F0EBE3',
  cardinal: '#8C1515',
  cardinalDark: '#6B0F0F',
  cardinalLight: '#B83A3A',
  ink: '#2E2D29',
  inkMuted: '#5F5C57',
  inkLight: '#8A8680',
  border: '#E8E2D8',
  chartGrid: '#E0D8CC',
} as const;

export const chartTooltipStyle = {
  background: colors.creamCard,
  border: `1px solid ${colors.border}`,
  borderRadius: 8,
  color: colors.ink,
};

export const policyColors: Record<string, string> = {
  ppo: colors.cardinal,
  dqn: colors.cardinalLight,
  greedy: '#C4A574',
  target_utilization: '#6B7280',
  fixed_replica: '#9CA3AF',
  do_nothing: '#D1D5DB',
};
