# GitHub Pages setup

The deploy workflow uses **GitHub Actions → Pages artifact** (not the `gh-pages` branch).

## Fix: `Get Pages site failed` / `configure-pages` / `HttpError: Not Found`

That error means **Pages is not enabled for GitHub Actions** on this repo yet. The workflow does **not** use `actions/configure-pages` (which fails until Pages exists).

### One-time setup (required)

1. Open **[Settings → Pages](https://github.com/dcunhrya/rl-inference-autoscaler/settings/pages)**.
2. Under **Build and deployment**, set **Source** to **`GitHub Actions`** (not “Deploy from a branch”).
3. If GitHub shows workflow suggestions, **ignore them** — this repo already has `.github/workflows/deploy-website.yml`.
4. Go to **Actions → Deploy website to GitHub Pages → Run workflow** (or push to `main`).

After 1–3 minutes the site should be live at:

**https://dcunhrya.github.io/rl-inference-autoscaler/**

### If Source is set to `gh-pages` branch instead

Either:

- Change Source to **GitHub Actions** (matches the current workflow), **or**
- Use an older workflow that pushes to `gh-pages` via `peaceiris/actions-gh-pages` — the current workflow does **not** do that.

**Do not mix** “Deploy from branch `gh-pages`” with a workflow that uses `configure-pages` / `deploy-pages`.

## How the workflow works

1. **build** — `npm ci && npm run build` in `website/` (with `BASE_PATH=/rl-inference-autoscaler/`)
2. **upload** — `website/dist` → Pages artifact
3. **deploy** — `actions/deploy-pages` publishes to the `github-pages` environment

## Redeploy

Push changes under `website/` or `results/` to `main`, or run the workflow manually from the Actions tab.

## Local production check

```bash
cd website
SITE_URL=https://dcunhrya.github.io BASE_PATH=/rl-inference-autoscaler/ npm run build
npm run preview -- --base /rl-inference-autoscaler/
```
