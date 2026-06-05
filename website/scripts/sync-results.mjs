#!/usr/bin/env node
/**
 * Copy benchmark JSON and figures from repo results/ into website/public/.
 * Run before dev/build (see package.json prebuild).
 */
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const websiteRoot = join(__dirname, '..');
const repoRoot = join(websiteRoot, '..');
const publicDir = join(websiteRoot, 'public');

const FIGURE_COPIES = [
  ['results/figures/benchmark/policy_comparison.png', 'figures/benchmark/policy_comparison.png'],
  ['results/figures/benchmark/pareto_frontier.png', 'figures/benchmark/pareto_frontier.png'],
  [
    'results/figures/scaling/scaling_vs_ground_truth_combined.png',
    'figures/scaling/scaling_vs_ground_truth_combined.png',
  ],
  [
    'results/figures/training/training_curves_overlay.png',
    'figures/training/training_curves_overlay.png',
  ],
];

const TRAJECTORY_KEYS = ['ppo', 'dqn', 'greedy'];

async function ensureDir(path) {
  await mkdir(path, { recursive: true });
}

async function copyIfExists(srcRel, destRel) {
  const src = join(repoRoot, srcRel);
  const dest = join(publicDir, destRel);
  await ensureDir(dirname(dest));
  try {
    await copyFile(src, dest);
    console.log(`  copied ${srcRel}`);
  } catch (err) {
    if (err.code === 'ENOENT') {
      console.warn(`  skip (missing): ${srcRel}`);
    } else {
      throw err;
    }
  }
}

async function syncBenchmarkSummary() {
  const src = join(repoRoot, 'results/benchmark_summary.json');
  const dest = join(publicDir, 'data/benchmark_summary.json');
  await ensureDir(dirname(dest));
  await copyFile(src, dest);
  console.log('  copied results/benchmark_summary.json');
}

async function syncTrajectories() {
  const fullPath = join(repoRoot, 'results/benchmark_full.json');
  let raw;
  try {
    raw = await readFile(fullPath, 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') {
      console.warn('  skip trajectories (missing benchmark_full.json)');
      return;
    }
    throw err;
  }

  const full = JSON.parse(raw);
  const out = {
    seed: full.seed,
    trajectory_episode: full.trajectory_episode ?? 0,
    policies: {},
  };

  for (const key of TRAJECTORY_KEYS) {
    const traj = full.policies?.[key]?.trajectory;
    if (traj) {
      out.policies[key] = { trajectory: traj };
    }
  }

  const dest = join(publicDir, 'data/trajectories.json');
  await ensureDir(dirname(dest));
  await writeFile(dest, JSON.stringify(out, null, 2));
  console.log('  wrote public/data/trajectories.json');
}

async function main() {
  console.log('Syncing results into website/public/...');
  await syncBenchmarkSummary();
  await syncTrajectories();
  for (const [src, dest] of FIGURE_COPIES) {
    await copyIfExists(src, dest);
  }
  console.log('Done.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
