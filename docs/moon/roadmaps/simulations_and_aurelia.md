# Interactive Simulations and Aurelia Roadmap

Goal: build reproducible interactive experiments once, then present them through lightweight React or isolated Aurelia 2 islands without coupling numerical logic to a view framework.

| ID | Deliverable | Effort | Depends on | Status |
| --- | --- | --- | --- | --- |
| SIM1 | Directory boundaries for `repository`, `scenarios`, `generator`, `context`, React adapters and lazy Aurelia islands | S | — | ✅ |
| SIM2 | Framework-neutral typed simulation contract with deterministic seeds, snapshots, validation and lifecycle controller | M | SIM1 | 🚧 |
| SIM3 | Optimization convergence demonstration with strategy presets, playback, metrics and accessible SVG chart | M | SIM2, IF3 | 🚧 |
| SIM4 | Aurelia 2 island mount/unmount boundary consuming the same simulation core | M | SIM2 | 🚧 |
| SIM5 | Generic simulation runner with play/pause/step/reset, speed, seed, progress, cancellation and serialized scenarios | L | SIM2, IT9 | 📋 |
| SIM6 | Worker adapter for CPU-heavy generators using the versioned transferable protocol | L | SIM5, IT9 | 📋 |
| SIM7 | Comparative experiment mode with synchronized timelines, small multiples and downloadable result data | L | SIM5, IF3 | 📋 |
| SIM8 | Aurelia component library for simulation controls, metrics, legends and error/loading states | L | SIM4, SIM5 | 📋 |
| SIM9 | Static-export island loader with visibility/intent hydration, failure fallback and per-island bundle budget | M | SIM4, IT6 | 📋 |
| SIM10 | Simulation authoring/testing guide and example templates for optimization, ML and stochastic systems | M | SIM5, DOC6 | 📋 |

## Architecture boundaries

- `repository/` owns serializable contracts and dataset lookup; `scenarios/` owns immutable presets. Neither imports a UI framework.
- `generator/` owns deterministic domain calculations. Generators accept explicit seeds/configuration and never read the DOM or clock.
- `repository/` owns serializable discriminated types. Snapshots must be safe to post to a worker and persist in a URL/file.
- `context/` owns orchestration and lifecycle transitions. Controllers expose progress, step/reset and later cancellation without rendering.
- React/Aurelia adapters translate user events and snapshots only. They do not reimplement algorithms.
- `src/aurelia/` is a lazy island boundary; React/Next.js continues to own routing and the shared shell.

## Acceptance criteria

- Identical scenario/seed inputs produce identical results in React, Aurelia, tests and workers.
- Invalid parameters fail before execution with a contextual, visitor-readable error; runaway jobs support cancellation and maximum limits.
- Every chart has a title/description, live but non-noisy metric text, keyboard controls, data-table/download equivalent and reduced-motion mode.
- Aurelia is absent from routes without an Aurelia island. Mount errors preserve server-rendered fallback content; unmount stops the Aurelia application and releases listeners/workers.
- Each simulation documents the concept being taught, assumptions, parameter units, validity limits and whether results are illustrative or scientifically computed.

## R2 research integration

RR2 and RR3 consume the deterministic controller and scenario repository for route playback, constraint toggles, comparison metrics, and Pareto explanations. RR10 adds versioned worker messages and replay export. React and Aurelia remain interchangeable: identical scenario/seed inputs produce identical snapshots, while an Aurelia mount failure preserves the static result.

### Exit gates

- Route replay has a table/download equivalent and URL-shareable scenario state.
- Solver labels distinguish illustrative, best-known, and proven-optimal results.
- Worker fixtures cover malformed messages, cancellation, stale responses, timeout, crash, and unmount.

## Relationship to multi-framework platform

Aurelia remains the **first non-React client island** for simulation controls. Platform-level work (shared Apollo client, Vue islands, WASM workers, webpack isolation) lives in [multi_framework_platform.md](multi_framework_platform.md) (MFP*). Prefer:

- **SIM\*** when changing numerical contracts, scenarios, workers, or Aurelia control widgets for experiments.
- **MFP\*** when changing host/island tooling, GraphQL/Apollo sharing, Vue/Astro build paths, or cross-framework test matrix.

SIM4/SIM6/SIM9 feed MFP6/MFP13/MFP15; do not duplicate acceptance criteria—link issues across both IDs when a PR touches both layers.

## Document history

| Date | Revision | Change |
|---|---|---|
| 2026-08-09 | R2.1 | Linked multi-framework platform roadmap (MFP) for Apollo/WASM/host concerns. |
| 2026-08-08 | R2 | Renamed simulation boundaries and linked RR2/RR3/RR10. |
