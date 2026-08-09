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

## Multi-framework and WASM quality hooks

When [multi_framework_platform.md](multi_framework_platform.md) lands islands, GraphQL fixtures, or WASM:

- Extend IT6 budgets with **per-island chunk** measurements (Vue/Aurelia/Apollo/WASM) and fail CI only on host-route regressions unless an island budget is explicitly approved.
- Extend IT9/IT10 for worker/WASM: cancellation, stale request IDs, ten mount/unmount leak probes per framework adapter (MFP15).
- Keep GraphQL tests offline via MSW or static fixtures (MFP11)—do not require a live GraphQL server for green CI.
- Treat dual-runtime INP regressions as quality-gate failures equal to bundle regressions.

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

## R2 research integration

RR9 requires capability matrices for WebGPU/WebGL, secure-context messaging, adapter/device failure fixtures, and an SVG/canvas fallback benchmark. RR10 requires deterministic worker replay tests, typed-array transfer accounting, stale-response rejection, and leak checks after ten mount/unmount cycles. Research demos cannot regress the static export or default route budget.

## R3 benchmark implementation

The root `benchmark/` harness now measures the production export without a browser. It records representative route status/bytes, export file count, JavaScript and CSS totals, largest assets, and explicit budget checks in both `benchmark/results/latest.json` and a reviewable `latest.md`. Run `npm run benchmark:build && npm run benchmark`; set `BENCHMARK_STRICT=1` in CI when a budget breach should fail the job. The first baseline exposes existing oversized media and aggregate bundles; those are optimization inputs, not hidden failures.

### Implementation approach

1. Keep route fixtures deterministic and local to avoid network variance.
2. Measure aggregate payloads first, then add browser navigation timing and Web Vitals.
3. Compare the same route set before and after every graphics or content milestone.
4. Treat intentional exceptions as roadmap entries with a fallback and remediation owner.
