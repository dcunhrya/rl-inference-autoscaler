# GitHub Pages setup (required once)

The deploy workflow builds the Astro site and pushes it to the **`gh-pages`** branch automatically.  
GitHub will **not** serve that branch until you turn on Pages for this repository.

## Enable the site (2 minutes)

1. Open **[Settings → Pages](https://github.com/dcunhrya/rl-inference-autoscaler/settings/pages)** for this repo.
2. Under **Build and deployment**, set **Source** to **Deploy from a branch** (not “GitHub Actions” unless you switch workflows).
3. Set **Branch** to **`gh-pages`** and folder **`/ (root)`**, then click **Save**.
4. Wait 1–3 minutes, then open: **https://dcunhrya.github.io/rl-inference-autoscaler/**

You should see a green banner on the Pages settings screen like “Your site is live at …”.

## Verify the deploy artifact

The **`gh-pages`** branch should contain `index.html`, `.nojekyll`, and `_astro/` at the root:

https://github.com/dcunhrya/rl-inference-autoscaler/tree/gh-pages

If that branch looks correct but the URL still 404s, Pages is not enabled or the wrong branch is selected in Settings.

## Redeploy

- Push changes under `website/` or `results/` to `main`, or  
- **Actions → Deploy website to GitHub Pages → Run workflow**

## Local production build

```bash
cd website
SITE_URL=https://dcunhrya.github.io BASE_PATH=/rl-inference-autoscaler/ npm run build
npm run preview -- --base /rl-inference-autoscaler/
```
