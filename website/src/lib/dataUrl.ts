/** Resolve a public data path respecting Astro base URL (GitHub Pages subpath). */
export function dataUrl(path: string): string {
  const base = import.meta.env.BASE_URL ?? '/';
  const normalized = path.startsWith('/') ? path.slice(1) : path;
  return `${base}${normalized}`;
}
