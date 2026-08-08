# github-pages Product Roadmap

> **Vision**: turn the site into an accessible, cinematic research portfolio where visitors can explore projects, algorithms, datasets, three-dimensional artefacts, and long-form writing without sacrificing the speed and resilience of a static website.

This roadmap translates the architecture research in [`research/`](research/) into an incremental delivery plan. Detailed, issue-ready work lives in [`roadmaps/`](roadmaps/); completed work is recorded in [`CHANGELOG.md`](CHANGELOG.md).

Status: ✅ complete · 🚧 in progress · 📋 planned · 🔬 discovery. Effort: S (days) · M (1–2 weeks) · L (multi-week).

## Product principles and non-negotiable budgets

1. **Content first** — navigation, project summaries, charts, and calls to action remain usable without WebGL or animation.
2. **Progressive immersion** — load interactive islands on visibility/intent; never make a 3D engine the price of reading an article.
3. **One graphics context** — if route-spanning graphics are adopted, own one renderer at the application shell and explicitly dispose route assets.
4. **Inclusive motion** — every effect has keyboard semantics, visible focus, a reduced-motion mode, and a static fallback.
5. **Measured fidelity** — target LCP ≤ 2.5 s, INP ≤ 200 ms, CLS ≤ 0.1 at the 75th percentile; target 60 fps on a representative laptop and 30 fps in constrained mode.
6. **Bounded payloads** — initial-route JavaScript ≤ 200 kB gzip excluding framework runtime; lazy interactive chunks ≤ 300 kB each; initial 3D assets ≤ 2 MB compressed with explicit exceptions documented.
7. **Static-export compatible** — features execute at build time or in the browser. Any backend is a separately approved architectural fork.

## Delivery sequence

| Milestone | Outcome | Exit criteria | Status |
| --- | --- | --- | --- |
| M0 Foundation | Next.js static portfolio, content routes, theme, baseline tests | Static export deploys and core routes pass unit/integration/e2e checks | ✅ |
| M1 Visual language | Tokenized aurora/glass system, responsive bento composition, motion preferences | WCAG 2.2 AA audit, no layout shift, baseline Lighthouse captured | 🚧 |
| M2 Interactive home | Interactive research constellation and capability-gated 3D model hero | Keyboard-operable visualization, static fallback, lazy 3D chunk, tests | 🚧 |
| M3 Project explorer | Filterable case-study grid, detail transitions, project metrics and timelines | URL-shareable filters, accessible data table, reduced-motion transition path | 📋 |
| M4 Spatial stories | Reusable model viewer, 360° panorama, annotations, audio-reactive report | Asset pipeline, context recovery, input alternatives, mobile quality tiers | 📋 |
| M5 Computational lab | PCVRP visual solver and small private client-side ML demonstration | Worker/WASM isolation, progress/cancel, reproducible examples, no runtime secrets | 📋 |
| M6 WebGPU frontier | Profile-driven WebGPU/TSL, large graphs or splats, optional XR | WebGL/static fallback and documented device matrix; budgets remain green | 🔬 |

## Workstreams

| Workstream | Scope | Detailed roadmap |
| --- | --- | --- |
| Visual design and UX | Design tokens, layout, typography, motion, accessibility | [`user_interface.md`](roadmaps/user_interface.md) |
| Interactive graphics | Data visualization, 3D, 360°, audio, effects | [`interactive_features.md`](roadmaps/interactive_features.md) |
| Simulations and Aurelia | Framework-neutral simulation engines and isolated Aurelia islands | [`simulations_and_aurelia.md`](roadmaps/simulations_and_aurelia.md) |
| Mathematical optimization | Interactive routing and browser solvers | [`mathematical_optimization.md`](roadmaps/mathematical_optimization.md) |
| Machine learning | Private, capability-aware browser inference | [`machine_learning.md`](roadmaps/machine_learning.md) |
| Infrastructure and quality | CI, budgets, observability, asset pipeline, browser matrix | [`infrastructure_and_testing.md`](roadmaps/infrastructure_and_testing.md) |
| Documentation and content | Architecture records, authoring guides, case studies | [`documentation.md`](roadmaps/documentation.md) |

## Cross-workstream dependency map

`UI6 capability policy` → `IF2 3D hero` → `IF5 reusable model viewer` → `IF6 panorama` → `IF10 WebGPU experiments`

`IT6 performance gates` → every M2+ interactive feature. `IF3 visualization primitives` → `MO2 route explorer` and `ML4 model metrics`. `IT9 worker protocol` → `MO3 solver` and `ML3 inference`. `SIM2 simulation contracts` → React/Aurelia adapters and computational labs. `DOC5 authoring schema` → project explorer and spatial annotations.

## Definition of done for every feature

- Acceptance criteria and analytics-free success metric are documented.
- Semantic static content and a no-WebGL/no-JavaScript fallback exist.
- Pointer, keyboard, touch, screen-reader, reduced-motion, light, and dark paths are covered as applicable.
- Unit tests cover public logic; component/integration tests cover interaction; a Cypress journey covers a new user-facing flow.
- `npm run lint`, `npm test`, and `npm run build` pass; performance and bundle deltas are recorded.
- New assets have provenance, compression settings, dimensions, and disposal/loading behavior documented.
- The roadmap status and [`CHANGELOG.md`](CHANGELOG.md) are updated in the same change.
