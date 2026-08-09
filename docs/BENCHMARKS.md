# Benchmarks

| Concern | Tool | Notes |
| --- | --- | --- |
| Build output size | `npm run build` output | Next.js prints per-route bundle sizes; watch for regressions on large content additions |
| Page performance | [Lighthouse](https://developer.chrome.com/docs/lighthouse/) (`npx lighthouse <url>`) | Run via [`.github/workflows/benchmark.yml`](../.github/workflows/benchmark.yml) against the static export, or manually against the deployed site |

> **TODO:** Track Lighthouse scores over time (store/compare reports) if this becomes worth tracking beyond a point-in-time check.
