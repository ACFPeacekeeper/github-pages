# Infrastructure, Performance and Test Roadmap

Goal: make ambitious visuals safe to ship by treating accessibility, performance, browser resilience, and asset lifecycle as release gates.

| ID | Deliverable | Effort | Depends on | Status |
| --- | --- | --- | --- | --- |
| IT1 | Vitest unit suite organized under `test/unit` | S | — | ✅ |
| IT2 | RTL integration suite with MSW boundary harness | M | IT1 | ✅ |
| IT3 | Cypress e2e and smoke suites under `test/cypress` | S | — | ✅ |
| IT4 | Static export/base-path compatible local production server | S | — | ✅ |
| IT5 | CI jobs for lint, type/build, unit, integration and Cypress | M | IT1–IT4 | ✅ |
| IT6 | Automated Lighthouse CI, bundle budgets and route-level Web Vitals baselines | M | IT5 | 📋 |
| IT7 | Coverage reporting with meaningful thresholds for utilities/hooks rather than blanket snapshot targets | S | IT5 | 📋 |
| IT8 | 3D/media asset pipeline: validation, Draco/Meshopt/KTX2 options, thumbnails, provenance and manifest budgets | L | IF5 | 📋 |
| IT9 | Typed Web Worker protocol, Comlink evaluation, cancellation/progress/error semantics and transferable buffers | L | UI6 | 📋 |
| IT10 | Graphics test harness: deterministic canvas mocks, context-loss tests, frame/leak probes and Playwright/Cypress visual checks | L | IF2 | 📋 |
| IT11 | Browser/device matrix in CI for Chromium, Firefox and WebKit plus manual low-power/mobile profiling | L | IT6 | 📋 |
| IT12 | Dependency/security automation, CSP/static-header guidance, SBOM and third-party asset/license audit | M | IT5 | 📋 |
| IT13 | Fix existing navigation and “other” Cypress failures and enforce zero uncaught console errors | S | IT3 | 📋 |
| IT14 | Preview deployment with before/after Lighthouse and visual-diff artefacts on pull requests | M | IT6, IT10 | 📋 |

## Release gates

- Lint, strict TypeScript, unit/integration, static build, core Cypress smoke, and broken-link checks pass.
- No route exceeds its recorded JavaScript/image/3D budget without an approved ADR.
- Home p75 targets: LCP ≤ 2.5 s, INP ≤ 200 ms, CLS ≤ 0.1; no long task > 200 ms from an optional visualization.
- Interactive rendering uses adaptive quality and does not continuously consume resources in a background tab.
- Automated axe checks report no serious/critical findings; manual keyboard and screen-reader smoke results accompany major UI work.
- A failed worker, denied GPU adapter, context loss, missing asset, or offline revisit produces recoverable UI rather than a blank region.

## Worker protocol requirements

Messages are discriminated unions with version, request ID and payload. Jobs support `queued/running/succeeded/failed/cancelled`, monotonic progress, structured errors, timeout, termination, and stale-response rejection. Large numeric data uses transferable typed-array buffers; tests cover malformed input, cancellation, worker crash and component unmount.

## Profiling procedure

Capture production builds on a representative integrated-GPU laptop and a throttled mobile profile. Record route bundle size, LCP/INP/CLS, time-to-first-interaction, median/1% frame time, context count, heap before/after ten navigations, and asset transfer bytes. Store the baseline in `docs/BENCHMARKS.md` and compare it for every graphics milestone.
