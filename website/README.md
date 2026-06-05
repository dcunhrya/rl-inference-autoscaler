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

## Deploy (GitHub Pages)

**Live URL:** [https://dcunhrya.github.io/rl-inference-autoscaler/](https://dcunhrya.github.io/rl-inference-autoscaler/)

1. The workflow builds Astro, pushes `website/dist` to **`gh-pages`**, and calls the GitHub API to enable Pages on **`gh-pages` / (root)**.
2. Pushes to `main` that touch `website/`, `results/`, or the workflow file run [`.github/workflows/deploy-website.yml`](../.github/workflows/deploy-website.yml).
3. Or trigger manually: **Actions → Deploy website to GitHub Pages → Run workflow**.
4. **If the URL still 404s:** open **Settings → Pages** and confirm **Deploy from a branch → gh-pages → / (root)**. Some org accounts require an admin to allow Pages on the repo first.

The workflow sets `SITE_URL` and `BASE_PATH=/rl-inference-autoscaler/` for Astro. `public/.nojekyll` ensures GitHub Pages serves Astro’s `_astro/` assets (Jekyll would ignore them otherwise).

Local production check:

```bash
cd website
SITE_URL=https://dcunhrya.github.io BASE_PATH=/rl-inference-autoscaler/ npm run build
npm run preview -- --base /rl-inference-autoscaler/
```
