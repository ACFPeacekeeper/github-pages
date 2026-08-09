# Multi-Framework Platform Roadmap

**IDs:** MFP1–MFP16 · **Status:** 🔄 Partial foundation · **Evidence base:** [`Next.js Multi-Framework Architecture.md`](../research/Next.js%20Multi-Framework%20Architecture.md), [`React Hosting Vue Micro-Frontends.md`](../research/React%20Hosting%20Vue%20Micro-Frontends.md), [`Advanced Web Portfolio Architecture Research.md`](../research/Advanced%20Web%20Portfolio%20Architecture%20Research.md), [`Global Interactive Portfolio Website Architecture.md`](../research/Global%20Interactive%20Portfolio%20Website%20Architecture.md)

Goal: make the static portfolio a **polyglot island host** where React owns the Next.js shell, while Vue 3, Astro, and Aurelia 2 can present specialized UI islands that share **framework-neutral data contracts**, a **single GraphQL/Apollo cache**, and **optional WASM workers**—without breaking static export, accessibility, or route budgets.

## Current codebase baseline (2026-08-09)

| Layer | Path / state | Notes |
| --- | --- | --- |
| React host | `src/frameworks/react/` | Primary shell, domain components, visualization primitives |
| Astro | `src/frameworks/astro/`, `src/pages/*.astro`, `public/astro-island/` | Island pattern via wrapper + static HTML iframe; not yet first-class SFC compilation in Next webpack |
| Aurelia | `src/frameworks/aurelia/{mount.ts,convergence-app.ts,components/AureliaWrapper.tsx}` | Lazy mount/unmount for convergence demo; depends on `aurelia` package |
| Vue | — | **Not present** (research only) |
| GraphQL | `src/graphql/schema.graphql` (placeholder), `src/graphql/fragments/` empty | No Apollo Client, no operations, no build-time schema pipeline |
| WASM | — | **Not present**; research points to HiGHS/solver and ExecuTorch/embedding paths |
| Shared | `src/frameworks/shared/utils.ts`, `src/simulations/**`, `src/interfaces/**` | Framework-neutral simulations already prove the island pattern |
| Tests | `test/unit` + `test/integration` + Cypress | Cover React shell/simulations; no multi-framework matrix yet |
| Static constraint | Next.js `output: 'export'`, GitHub Pages | No runtime API; GraphQL must be build-time/fixture or explicit optional backend fork |

## Deliverable index

| ID | Deliverable | Effort | Depends on | Status |
| --- | --- | --- | --- | --- |
| MFP1 | Document multi-framework directory contract under `src/frameworks/{react,vue,astro,aurelia,shared}` and ADR for host vs island ownership | S | — | 🔄 |
| MFP2 | Next.js webpack (or dual-build) blueprint: `vue-loader`, Aurelia loader isolation, `output.uniqueName`, splitChunks for vue/aurelia/apollo; static-export compatibility note | L | MFP1, IT6 | ⬜ |
| MFP3 | React island host utilities: dynamic `ssr: false` boundary, layout reservation, mount error fallback, visibility/intent hydration | M | MFP1, UI6 | 🔄 |
| MFP4 | Vue 3 island path: SFC demo, client-only wrapper (Web Component *or* Veaury spike), props/events bridge, teardown + heap test | L | MFP2, MFP3 | ⬜ |
| MFP5 | Astro island promotion: replace iframe-only path with documented build of `astro-island` assets + typed React host, or Container API evaluation under static export | M | MFP1, MFP3 | 🔄 |
| MFP6 | Aurelia island hardening: DI registration of shared services, unmount guarantees, lazy chunk, same simulation contract as React (extends SIM4/SIM9) | M | MFP3, SIM2 | 🔄 |
| MFP7 | Cross-framework visual parity kit: one visitor question rendered in React + second framework with identical a11y summary/table | M | MFP4 or MFP6, IF3 | ⬜ |
| MFP8 | GraphQL schema v1 for portfolio content graph (posts, reports, projects, media nodes, edges) replacing the `_empty` placeholder | M | DOC/UI data model | ⬜ |
| MFP9 | Apollo Client singleton module (framework-agnostic): InMemoryCache policies, type policies, no React imports in core | M | MFP8 | ⬜ |
| MFP10 | Framework Apollo adapters: React (`@apollo/client` hooks), Vue (vue-apollo or composition helper), Aurelia DI registration, Astro client script consumer | L | MFP9, MFP4–MFP6 | ⬜ |
| MFP11 | Static-export GraphQL strategy: build-time fixture generation and/or MSW mock server for tests; document “no live endpoint” default and optional backend fork | M | MFP8, MFP9, IT2 | ⬜ |
| MFP12 | WASM toolchain: `wasm-pack`/`emscripten` build recipe, `src/wasm/` (or `src/libraries/wasm/`) loader, typed JS bindings, CI artifact check | L | IT5 | ⬜ |
| MFP13 | WASM worker for optimization/solver slice (HiGHS-class or thin stub): CSC/typed-array transfer, progress, cancel, fallback pure-TS generator | L | MFP12, SIM6, MO2 | ⬜ |
| MFP14 | WASM/WebGPU ML path decision: tiny embedding or model-card + recorded replay first; WebNN/ExecuTorch only as research spike with budgets | XL | MFP12, ML1–ML2, IF10 | 🔬 |
| MFP15 | Multi-framework test matrix: unit mount/unmount, integration cache broadcast, Cypress island smoke, bundle budgets per framework chunk | L | MFP4–MFP6, MFP10, IT6 | ⬜ |
| MFP16 | Authoring guide + catalogue: when to use React vs Vue vs Astro vs Aurelia vs WASM; failure modes; ADR links | M | MFP7, MFP11, MFP15, DOC6 | ⬜ |

## Architecture principles (from research + site constraints)

1. **React/Next is the host.** Routing, layout shell, content Markdown pipeline, and GitHub Pages export remain React/Next responsibilities.
2. **Islands are client-only.** Vue and Aurelia never run in the RSC/Node pre-render path; load with dynamic import and `ssr: false` (or equivalent static placeholders).
3. **Astro is content/static-first.** Prefer prebuilt HTML/islands or build-time compilation; do not assume a long-lived Next Node server (this site is `output: 'export'`).
4. **Module Federation is not the default.** Research notes `@module-federation/nextjs-mf` is a poor fit for App Router; prefer colocated islands, Web Components, or Multi-Zone only if we split deploys later.
5. **One data graph.** Apollo `InMemoryCache` is the shared source of truth for cross-island entity state; avoid dual Redux+Pinia ownership of the same entities.
6. **Simulations stay framework-neutral.** Numerical work lives in `src/simulations/**` (and later WASM); islands only adapt snapshots and controls (see [simulations_and_aurelia.md](simulations_and_aurelia.md)).
7. **WASM is opt-in enhancement.** Pure-TS fixtures remain the static/reduced path; WASM accelerates the full tier after capability and memory checks.
8. **Budgets are law.** Optional framework runtime chunk ≤ 300 kB gzip per island policy; initial route JS still ≤ 200 kB gzip excluding host framework; islands load after intent/visibility.

## Target directory layout

```text
src/frameworks/
  react/          # host UI (existing)
  vue/            # Vue SFCs + React bridge wrappers (MFP4)
  astro/          # Astro sources + host wrappers (existing → MFP5)
  aurelia/        # Aurelia apps + mount (existing → MFP6)
  shared/         # mount helpers, types, no framework imports
src/graphql/
  schema.graphql  # v1 content graph (MFP8)
  fragments/      # shared selections
  operations/     # queries/mutations (codegen later)
  fixtures/       # static JSON for export/tests (MFP11)
src/libraries/
  apollo/         # client singleton, cache, policies (MFP9–MFP10)
  wasm/           # loaders + JS bindings (MFP12–MFP14)
src/simulations/  # unchanged ownership; WASM worker is an adapter (MFP13)
```

## Work packages

### MFP1–MFP3 — Host contract and build readiness

**Problem.** Multi-framework research assumes Webpack loaders and federation patterns that can collide with Next App Router + static export if applied naively.

**Options**

| Option | Pros | Cons |
| --- | --- | --- |
| A. Colocated islands + dual runtime | Matches research; single deploy | Bundle cost; dual VDOM |
| B. Web Components only | Framework-agnostic DOM | Shadow DOM / Tailwind friction |
| C. Compile-time Vue→React | Zero Vue runtime | Not true multi-framework demo |
| D. Multi-Zone separate apps | Clean isolation | Operational complexity for GH Pages |

**Recommendation.** **A + selective B:** colocated client islands for Vue/Aurelia demos; Web Components where style isolation is required. Document that Module Federation is out of scope for the static host (ADR).

**Acceptance**

- ADR records host/island ownership, static-export constraints, and rejected federation default.
- `src/frameworks/` layout and import rules are documented; tests fail if simulations import a view framework.
- Island host helper reserves layout, loads after intent, and shows semantic fallback on error.

### MFP4–MFP7 — Framework islands

**Vue (MFP4).** Spike Veaury vs `defineCustomElement`; pick one for the first demo. Implement a small research widget (e.g. filterable citation list or constraint toggles) that mirrors a React control’s visitor question. Prove unmount does not leak Vue app instances (heap snapshot protocol in IT10 spirit).

**Astro (MFP5).** Today `AstroWrapper` iframes a prebuilt `public/astro-island` asset. Promote to a repeatable build step (`astro build` → `public/astro-island`), document hydration islands (`client:visible`), and ensure `basePath` `/github-pages` works.

**Aurelia (MFP6).** Extend existing mount boundary: inject shared Apollo/simulation services via DI, enforce lazy chunk, and align with SIM4/SIM9 acceptance (static fallback, stop on unmount).

**Parity (MFP7).** One dual-framework story on a tools or experiments route: same scenario seed, same accessible table, keyboard path in both islands.

### MFP8–MFP11 — GraphQL + Apollo

**Schema (MFP8).** Replace the placeholder:

```graphql
type Query { _empty: String }
```

with a **static portfolio content graph** (ContentItem, Publication, Project, MediaAsset, Topic, edges). Prefer build-time generation from Markdown front matter over a live server.

**Client (MFP9–MFP10).** `createApolloClient()` lives under `src/libraries/apollo` with **zero React imports**. React uses `ApolloProvider` only at island/host boundaries that need it. Vue/Aurelia adapt via thin wrappers that call the same cache.

**Static export (MFP11).** Default mode: fixture JSON + MSW for tests; optional “Apollo link to static files”. Live GraphQL HTTP is an explicit backend fork (violates G0 if required at runtime).

**Acceptance**

- Mutation/update in one island (or test double) updates a React subscriber via cache broadcast.
- Cypress/integration tests run offline with fixtures only.
- Bundle does not ship Apollo on routes that never open a GraphQL island.

### MFP12–MFP14 — WebAssembly

**Toolchain (MFP12).** Check in build docs + script; load WASM only after capability check; graceful failure when `WebAssembly` missing or memory low.

**Solver path (MFP13).** Align with MO/SIM: pure-TS convergence remains default; WASM worker accelerates larger fixtures with typed-array transfer, request IDs, cancel, and stale-result rejection (RR10).

**ML path (MFP14).** Research-only until ML model cards and budgets pass. Prefer recorded replay + tiny WASM embedding over shipping large models. WebNN/ExecuTorch remain spikes under IF/ML, not product defaults.

### MFP15–MFP16 — Quality and docs

- Per-island bundle reports in `benchmark/` or webpack analyzer notes.
- Mount/unmount tests for each framework adapter.
- Cypress smoke: host route loads; island falls back without console errors when force-disabled.
- Catalogue entry in component catalogue / DOC workstream: decision tree for framework choice.

## Testing plan

| Layer | What to prove |
| --- | --- |
| Unit | Mount/unmount helpers; cache policies; WASM loader error paths; pure-TS fallback equals WASM for fixture seeds |
| Integration | MSW GraphQL (or fixture link); React subscriber updates when cache writes; island failure preserves DOM fallback |
| E2E/smoke | Route with multi-framework demo; keyboard path; reduced-motion; no uncaught errors |
| Perf | Island chunk sizes; LCP of host route unchanged when island not opened; INP during dual-runtime interaction |
| Memory | Ten mount/unmount cycles per framework; no retained detached nodes |

## Performance budgets (multi-framework deltas)

| Budget | Target |
| --- | --- |
| Host route JS (no island open) | Unchanged vs current baseline (±5%) |
| First island open (gzip) | ≤ 300 kB additional for that framework runtime+demo |
| Second concurrent island | Prefer sequential exclusive mode if dual-runtime INP > 200 ms |
| WASM module download | Explicit consent/progress for > 1 MB; never block first paint |
| GraphQL fixture set | Prefer incremental route-level fixtures, not one megagraph on home |

## Risks and mitigations

| Risk | Detection | Mitigation |
| --- | --- | --- |
| Dual-runtime jank (React+Vue) | INP/long tasks | Load one secondary framework per route; capability gate |
| Next webpack loader conflicts | Build failures / SWC collisions | Isolate loaders by path; dual-package builds for islands |
| Apollo pulled into every route | Bundle analyzer | Dynamic import Apollo only with GraphQL islands |
| Static export vs “Container API” Astro | Runtime assumptions | Prefer prebuild Astro assets for GH Pages |
| WASM OOM | Crash / fail to instantiate | Fixture size caps; pure-TS fallback; memory gate |
| Overclaim multi-framework “architecture” | Marketing copy | Islands are demos of skill + isolation, not micro-org federation |
| Unmaintained bridges (Veaury) | Dependency audit | Prefer Web Components if maintenance stalls; ADR exit |

## R2 research mapping

| Research theme | Roadmap IDs |
| --- | --- |
| Vue in Next host (Web Components / Veaury / no App Router federation) | MFP2–MFP4, MFP7 |
| Astro Container / islands vs static export | MFP5 |
| Aurelia DI host element lifecycle | MFP6, SIM4/SIM9 |
| Apollo shared InMemoryCache across frameworks | MFP8–MFP11 |
| Webpack uniqueName + splitChunks | MFP2, MFP15 |
| WASM solvers (HiGHS-class) and edge ML | MFP12–MFP14, MO/ML |

## Exit gates

### Gate MF-G1 — Island host (partial → complete when MFP1–MFP3/MFP6 done)

React host can open/close Astro and Aurelia islands without leaking resources; static fallback survives.

### Gate MF-G2 — Second client framework

Vue island ships with parity demo, tests, and budget evidence (MFP4, MFP7, MFP15).

### Gate MF-G3 — Unified data graph

Schema v1 + Apollo singleton + static fixtures; cross-island cache update proven in tests (MFP8–MFP11).

### Gate MF-G4 — WASM acceleration

Solver or compute slice uses WASM worker with pure-TS fallback and memory gate (MFP12–MFP13).

## Document history

| Date | Revision | Change |
| --- | --- | --- |
| 2026-08-09 | R1 | Initial multi-framework platform roadmap from architecture research + codebase inventory. |
