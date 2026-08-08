# Website performance benchmark

This directory contains a dependency-light benchmark for the statically exported site. It measures the assets that a visitor can request from `out/`, without requiring a hosted server, API, telemetry, or a third-party account.

## Goals

1. Catch accidental growth in the default route.
2. Record the transfer cost of representative content routes.
3. Make benchmark output reviewable in pull requests.
4. Leave room for browser-level Lighthouse/Cypress measurements later.

## Quick start

```bash
npm run benchmark:build
npm run benchmark
```

`benchmark:build` creates the production export. `benchmark` starts a temporary local static server, requests the configured routes, and writes JSON (`latest.json`) plus a reviewable Markdown summary (`latest.md`) under `benchmark/results/`.

## What is measured

- HTTP status and response time for `/github-pages/` and representative content routes.
- Response bytes and compressed bytes when the server provides them.
- Total export bytes and file count.
- JavaScript and CSS payload totals.
- Largest individual assets.
- A pass/warn/fail assessment against the roadmap budgets.

The script intentionally does not claim to measure real-user LCP, INP, CLS, GPU frame time, memory, or accessibility. Those require a browser matrix and are tracked as follow-up work in the infrastructure roadmap.

## Budgets

The initial budgets are deliberately visible and editable in `measure.mjs`:

| Budget | Default | Reason |
| --- | ---: | --- |
| Initial JavaScript | 200 kB | Protect the first interaction path. |
| Initial CSS | 80 kB | Keep typography and layout inexpensive. |
| Homepage transfer | 2 MB | Leave room for optional visual islands. |
| Route response | 3 MB | Keep content routes usable on slow connections. |
| Largest asset | 1 MB | Encourage responsive, licensed media. |

Budgets are not a substitute for profiling. A route may exceed one budget for a justified media story, but the roadmap entry must record the reason, fallback, and measured alternative.

## Repeatable procedure

1. Use a clean production build.
2. Run the benchmark on the same machine and Node version used by CI.
3. Compare the generated summary with the previous commit.
4. Investigate changes greater than 10% before merging.
5. Record intentional changes in `docs/moon/CHANGELOG.md` and the relevant roadmap item.

## Browser follow-up

The next benchmark milestone will run Playwright or Cypress against Chromium, Firefox, and WebKit. It will collect navigation timing, Web Vitals, reduced-motion behavior, WebGL fallback behavior, and ten mount/unmount leak probes. Browser automation is kept separate so the static asset benchmark remains fast and works in constrained CI runners.

## Result hygiene

Generated results are ignored by Git. Keep one manually selected baseline in a release note or benchmark report when a roadmap gate is closed. Never commit private URLs, local file paths, cookies, user content, or machine identifiers.
