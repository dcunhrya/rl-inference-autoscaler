# RL Inference Autoscaler — Interactive Blog

Long-form interactive article for the [RL inference autoscaler](https://github.com/dcunhrya/rl-inference-autoscaler) project.

**Live site:** [https://dcunhrya.github.io/rl-inference-autoscaler/](https://dcunhrya.github.io/rl-inference-autoscaler/)

## Stack

- [Astro](https://astro.build/) 4 + MDX
- [React](https://react.dev/) (interactive charts)
- [Recharts](https://recharts.org/)
- [Tailwind CSS](https://tailwindcss.com/)

## Local development

```bash
# From repo root: refresh plots/data (optional)
.venv/bin/python scripts/generate_results_plots.py

cd website
npm install
npm run dev
```

Open [http://localhost:4321](http://localhost:4321).

`npm run sync` copies `results/benchmark_summary.json`, trajectory slices, and figure PNGs into `public/`. This runs automatically before `dev` and `build`.

## Build

```bash
npm run build
npm run preview
```

## Content

- Article sections: `src/content/sections/*.mdx`
- React interactives: `src/components/react/`
- Entry page: `src/pages/index.astro`

## Deploy

Pushes to `main` that touch `website/` or `results/` trigger [`.github/workflows/deploy-website.yml`](../.github/workflows/deploy-website.yml).

One-time: **Settings → Pages → Source: GitHub Actions**.
