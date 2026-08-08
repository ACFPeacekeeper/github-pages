# github-pages — Immersive Research Portfolio Master Roadmap

**Last updated:** 2026-08-08 · **Roadmap session:** R2 · **Delivery model:** static-first progressive enhancement

This is the product-level index. Each workstream file is an issue-ready implementation document with options, acceptance criteria, tests, budgets, risks, and a history of scope changes. The new evidence base is [`Interactive Features and Visual Storytelling Research.md`](research/Interactive%20Features%20and%20Visual%20Storytelling%20Research.md).

## Table of contents

- [Vision and operating rules](#vision-and-operating-rules)
- [Implementation timeline](#implementation-timeline)
- [Current state](#current-state)
- [How to use this roadmap](#how-to-use-this-roadmap)
- [Phase gates](#phase-gates)
- [Workstreams](#workstreams)
- [Research-derived feature index](#research-derived-feature-index)
- [Dependency and risk register](#dependency-and-risk-register)
- [Effort × impact matrix](#effort--impact-matrix)
- [Anchor index](#anchor-index)
- [Document history](#document-history)

## Vision and operating rules

Turn the site into a beautiful, legible research observatory: visitors can understand waste-fleet routing, deep reinforcement learning, optimization trade-offs, game-development experiments, media references, and technical/political history through carefully staged interaction.

1. **Explain before embellishing.** The claim, data, units, source, and next action are visible before a canvas or model loads.
2. **One fact, multiple senses.** Every visualization has DOM text, keyboard controls, a list/table or download, and a reduced-motion path.
3. **Static export is a product constraint.** No runtime API, token, secret, or required server is assumed. Backend proposals are explicit forks.
4. **Capability tiers are policy, not guesses.** Static, reduced, and full tiers respond to preferences, hardware, browser support, visibility, and measured performance.
5. **Research honesty is visual.** Show assumptions, uncertainty, feasibility, incumbent/bound/gap, confidence, limitations, and provenance.
6. **Local by default.** Local audio, notes, annotations, simulation inputs, and future inference inputs stay on-device unless a visitor explicitly opts in.
7. **Every effect has a budget.** LCP ≤ 2.5 s, INP ≤ 200 ms, CLS ≤ 0.1 at p75; initial route JS ≤ 200 kB gzip excluding framework runtime; optional island ≤ 300 kB gzip; initial 3D/media ≤ 2 MB.

## Implementation timeline

> **Legend:** node fill = work type (blue feature, violet augmentation, cyan infrastructure, amber performance, green docs, slate research); border = status (green complete, amber in progress, slate planned, red blocked). `==>` is a blocking dependency, `-->` sequential, `---` complementary.

```mermaid
flowchart LR
  classDef feature fill:#2563eb,color:#fff
  classDef augment fill:#7c3aed,color:#fff
  classDef infra fill:#0891b2,color:#fff
  classDef perf fill:#ea580c,color:#fff
  classDef docs fill:#15803d,color:#fff
  classDef research fill:#475569,color:#fff
  classDef done stroke:#16a34a,stroke-width:4px
  classDef active stroke:#d97706,stroke-width:4px
  classDef planned stroke:#64748b,stroke-width:2px

  R0["R0 Foundation\nstatic site + tests"]:::infra
  R1["R1 Visual language\ntokens + motion"]:::augment
  R2["R2 Explainable\nvisual primitives"]:::feature
  R3["R3 Fleet observatory\nmap + solver replay"]:::feature
  R4["R4 ML lab\nmodel + audio"]:::feature
  R5["R5 Culture room\nmedia + books + game"]:::feature
  R6["R6 Spatial tier\n360 + WebGL"]:::augment
  R7["R7 WebGPU frontier\nworkers + optional XR"]:::research
  Q["Quality gates\nbudgets + a11y"]:::perf
  D["Docs + provenance\nresearch reports"]:::docs
  class R0 done
  class R1,R2,Q,D active
  class R3,R4,R5,R6,R7 planned
  R0 ==> R1 ==> R2 ==> R3
  R2 --> R4
  R2 --> R5
  R3 --> R6
  R4 --> R7
  Q ==> R2
  Q ==> R3
  D --- R2
  D --- R5
```

## Current state

### Shipped or partially shipped (R1)

- Next.js 14 static export, content routes, theme shell, Vitest/RTL/MSW, Cypress, and notebooks workspace.
- Tokenized observatory homepage with Three.js model fallback, graph/DOM constellation, deterministic optimization-convergence simulation, and field-note components.
- Typed contracts under `src/interfaces`, Redux experience state under `src/redux`, domain components under `src/components/{audio,books,canvas,games,graph,image,maps,models,routes,video}`.
- Framework-neutral simulations with `repository/` types, `scenarios/` presets, `generator/` computation, and `context/` lifecycle; optional Aurelia boundary.
- 51 tests passing at the last implementation gate; production static export succeeds. Existing lint warnings for legacy `<img>` usage and one ref cleanup remain tracked in infrastructure work.
- Root `benchmark/` harness now records static-export bytes, representative route responses, largest assets, and budget checks; the first baseline identifies existing media and aggregate-bundle optimization work.

### Open risks

- Route map and solver are currently illustrative SVG/Canvas/recorded traces; they do not claim live municipal data or optimality.
- Three.js model is a procedural object, not yet a reusable glTF/360 asset pipeline.
- Redux currently covers shell-level experience state; capability-tier selection and worker job state remain to be added.
- The current home imports the field-note gallery eagerly; domain-level lazy loading is a quality-gate task.

## How to use this roadmap

Each workstream section follows the Image-Toolkit convention:

1. Read the timeline and current-status table before starting.
2. Read the item’s problem statement, options/trade-offs, recommendation, acceptance criteria, test plan, performance budget, and risks.
3. Link implementation to a stable ID and GitHub issue. Mark `Partial` when only the minimum slice shipped; do not silently convert a discovery item to done.
4. Record measured results, deviations, and residual risks in the item and changelog. Update the dependency graph when scope changes.

Status vocabulary: ✅ Done · 🔄 Partial/in progress · ⬜ Planned · 🔬 Research · ⛔ Blocked. Effort: S (<2 days), M (2–7 days), L (1–3 weeks), XL (multi-week/architecture fork).

## Phase gates

### Gate G0 — Static contract (complete)

Core routes build on GitHub Pages, semantic content survives disabled JavaScript, and unit/integration/e2e tests run deterministically.

### Gate G1 — Visual system (in progress)

Tokens, responsive hierarchy, reduced motion, focus visibility, contrast, and a capability policy exist before additional effects are added. Exit: Lighthouse baseline and manual keyboard/screen-reader smoke.

### Gate G2 — Explainable interaction (in progress)

Shared scales, palettes, legends, selection, URL state, summaries, tables, and fixture data exist. Exit: a visitor can reproduce the same conclusion from the visual and DOM representations.

### Gate G3 — Computational storytelling (planned)

Fleet route playback, solver comparison, model replay, and audio are worker/fixture-safe. Exit: cancellation, stale-result protection, metrics, provenance, and performance evidence.

### Gate G4 — Spatial/media stories (planned)

3D model/panorama assets have manifests, licenses, thumbnails, quality tiers, and disposal tests. Exit: flat/static fallback and mobile profile remain within budgets.

### Gate G5 — GPU/experimental frontier (research)

WebGPU, splats, WebXR, and large graphs are only promoted after a device matrix proves a user-facing benefit. Exit: WebGL/SVG fallback, privacy review, and a documented support matrix.

## Workstreams

| Workstream | IDs | Detail | Current state |
| --- | --- | --- | --- |
| Visual design and UX | UI1–UI14 | [user_interface.md](roadmaps/user_interface.md) | 🔄 UI3–UI5/UI13–UI14 partial |
| Interactive graphics | IF1–IF13 | [interactive_features.md](roadmaps/interactive_features.md) | 🔄 IF2–IF4/IF7/IF13 partial |
| Simulations and Aurelia | SIM1–SIM10 | [simulations_and_aurelia.md](roadmaps/simulations_and_aurelia.md) | 🔄 SIM1–SIM4 partial |
| Mathematical optimization | MO1–MO8 | [mathematical_optimization.md](roadmaps/mathematical_optimization.md) | ✅ MO1; ⬜ MO2+ |
| Machine learning | ML1–ML8 | [machine_learning.md](roadmaps/machine_learning.md) | ✅ ML1; ⬜ ML2+ |
| Infrastructure and quality | IT1–IT14 | [infrastructure_and_testing.md](roadmaps/infrastructure_and_testing.md) | ✅ IT1–IT5; ⬜ IT6+ |
| Documentation/content | DOC1–DOC11 | [documentation.md](roadmaps/documentation.md) | 🔄 DOC3; ⬜ DOC4+ |
| Research-derived interaction | RR1–RR10 | [research report](research/Interactive%20Features%20and%20Visual%20Storytelling%20Research.md) | 🔬 research captured |

### R3 implementation slices

| Slice | Implementation approach | Evidence gate |
| --- | --- | --- |
| IT-B1 | Artifact-first benchmark with local server and deterministic routes | `benchmark/results/latest.json`, route status 200 |
| DOC-B1 | Root README launch/structure/runbook and architecture diagrams | README and architecture exceed maintenance-detail threshold |
| MO-B1 | Route playback remains static-first before solver worker integration | deterministic fixture and table equivalent |
| ML-B1 | Model-card contract precedes runtime/model download | CPU fixture, version, limitations, privacy note |
| IF-B1 | Domain islands remain independently importable | bundle and fallback checks |

## Detailed implementation playbook

This section turns the roadmap IDs into repeatable engineering decisions. It is intentionally explicit so that an issue can be implemented without rediscovering the architecture.

### 1. Prepare the visitor question

Every feature begins with a sentence of the form “A visitor should be able to understand or compare ___ by interacting with ___.” The first render contains that sentence, the units, the source/provenance link, and the next action. If the sentence cannot be stated without promising a scientific conclusion, the feature is a research prototype and must use illustrative language.

### 2. Define the serializable contract

Create or extend an interface under `src/interfaces` or `src/simulations/repository`. Prefer literal unions for status and quality tiers, readonly arrays for fixture data, explicit units in field names or documentation, and nullable fields for unavailable measurements. Avoid classes, functions, DOM nodes, browser handles, and implicit dates. Add a fixture that represents a valid, empty, invalid, and degraded state.

### 3. Build the non-visual path

Render semantic headings, paragraphs, lists, tables, links, labels, and buttons first. The fallback must answer the visitor question without CSS animation, WebGL, WebGPU, audio, a map tile, or a network request. Give it a stable test selector only when a semantic role or accessible name is insufficient.

### 4. Add local interaction

Use component state for hover, focus, drag, playback cursor, filters, and drafts. Use a controller for simulation lifecycle. Use Redux only for preferences or cross-route selections. Keyboard actions mirror pointer actions and have visible focus. URL state is added only when a visitor benefits from sharing or refreshing a view.

### 5. Add the enhancement behind a capability boundary

Load Three.js, audio analysis, map layers, or future GPU code after visibility/intent. Respect reduced motion and quality preference before allocating resources. Catch initialization and context-loss failures, return the fallback, and report a concise status message. Do not let a dynamic import change the document's layout.

### 6. Teardown and test

The owner of an animation frame cancels it. The owner of an event listener removes it. The owner of an audio context suspends/closes it. The owner of a worker terminates it. The owner of an object URL revokes it. Tests mount, interact, unmount, and verify no stale update occurs afterward.

### 7. Measure and document

Run the artifact benchmark, capture route bytes and largest assets, and record any intentional budget exception. For graphics, add a representative device profile and frame/heap observation. Update the relevant roadmap row, changelog, architecture note, and GitHub issue before calling the slice In review or Done.

## Workstream implementation details

### Visual UI (UI1–UI14)

- Establish tokens for surface, text, border, focus, accent, spacing, radius, type scale, and motion.
- Keep light/dark themes semantic; components consume tokens rather than hard-coded colors.
- Reserve media and canvas dimensions to prevent layout shift.
- Test 320, 768, and 1440 px widths, forced colors, 200% zoom, keyboard-only navigation, and reduced motion.
- Use View Transitions only as an enhancement; browser navigation and focus restoration remain correct without it.

### Interactive graphics (IF1–IF13)

- Separate data encodings from renderers so SVG, Canvas, and WebGL can share scales and legends.
- Keep selection represented in the DOM and announce it without frame-by-frame live-region noise.
- Prefer event-driven SVG/canvas for small fixtures; evaluate deck.gl/WebGPU only after thresholds are measured.
- Keep model annotations in DOM content and expose a sequential annotation index.
- Store asset manifests with license, dimensions, compression, quality tier, and fallback poster.

### Mathematical optimization (MO1–MO8)

- Start with seeded fixtures for depot, vehicle capacity, demand, time windows, and dropped visits.
- Provide baseline and heuristic traces before adding a solver runtime.
- Label feasible, infeasible, timed out, incumbent, bound, gap, best-known, and proven-optimal states distinctly.
- Move long computation to a worker with request IDs, cancellation, progress, and typed-array transfer only after profiling.
- Export scenario and result JSON so a reader can reproduce a chart and inspect assumptions.

### Machine learning (ML1–ML8)

- Define a model card before selecting a runtime: task, data, preprocessing, version, provider, latency, memory, limitations, and license.
- Ship a deterministic recorded replay and CPU/static fallback before WebGPU or WASM acceleration.
- Keep local inputs local and state clearly when a model is illustrative rather than validated.
- Visualize reward/cost, confidence, policy choice, and error cases with a table equivalent.
- Test unsupported operators, corrupt model cache, cancellation, and out-of-memory messaging.

### Simulations and Aurelia (SIM1–SIM10)

- Keep `repository`, `scenarios`, `generator`, and `context` free of view imports.
- Require React and Aurelia to consume the same snapshot contract and deterministic seed.
- Mount Aurelia only in an island with a static fallback and a strict unmount path.
- Keep worker protocol versioned and reject stale responses.
- Document assumptions and validity limits beside every scenario fixture.

### Infrastructure and testing (IT1–IT14)

- Keep lint, typecheck, unit, integration, browser smoke, build, and benchmark commands independently runnable.
- Treat benchmark output as evidence, not a score; explain regressions in issues.
- Add a browser matrix for WebGL unavailable/context lost, reduced motion, slow network, and coarse pointer.
- Run dependency/license checks before adding media, models, runtimes, or map providers.
- Record ten-navigation heap/context probes for persistent graphics owners.

### Documentation (DOC1–DOC11)

- Give each feature an overview, data contract, lifecycle, accessibility equivalent, browser support, fallback, test plan, benchmark, and license section.
- Link a stable roadmap ID and GitHub issue from implementation notes.
- Distinguish academic evidence, standards guidance, corporate examples, personal inspiration, and measured repository results.
- Preserve historical rationale in changelog and document history tables.

## Definition-of-done template

Copy this template into an issue before implementation:

```markdown
## Visitor question
## Roadmap ID and dependencies
## Data contract and provenance
## Static/reduced/full rendering paths
## Keyboard and screen-reader behavior
## Failure, cancellation, and teardown behavior
## Tests and deterministic fixtures
## Benchmark before/after
## Licensing and privacy review
## Changelog and documentation updates
```

## Review gates

| Gate | Reviewer asks | Evidence |
| --- | --- | --- |
| Content | Can the claim be read without JavaScript? | static route and text fallback |
| Contract | Are units, status, provenance, and failure states explicit? | interface and fixtures |
| Interaction | Can keyboard and reduced-motion visitors reach the same conclusion? | tests and manual path |
| Lifecycle | Are resources released after hidden/unmount/error? | teardown test/profile |
| Performance | Did route bytes or first interaction regress? | benchmark result |
| Honesty | Does wording distinguish illustrative from measured/optimal? | copy review and model card |
| Operations | Is the issue/status/changelog/roadmap synchronized? | project item and commit |

## Document history

| Date | Revision | Change |
| --- | --- | --- |
| 2026-08-08 | R3 | Added implementation playbook, workstream approaches, definition-of-done template, and review gates. |

## Research-derived feature index

| ID | Feature | Primary workstream | Evidence and next slice |
| --- | --- | --- | --- |
| RR1 | Cited research/source graph and reading room | DOC/UI | Narrative visualization + accessible graph; static first |
| RR2 | Waste-fleet route playback | MO/IF | deck.gl TripsLayer/Mapbox patterns + OR-Tools semantics |
| RR3 | Solver/heuristic/Pareto comparison | MO/IF | incumbent, bound, gap, feasibility and export |
| RR4 | ML training/policy replay and model card | ML/IF | interactive ML + Manifold model comparison |
| RR5 | Local/demo audio spectrum and spectrogram | ML/IF | MDN `AnalyserNode`, explicit gesture/teardown |
| RR6 | Media/reading timeline and argument graph | UI/DOC | narrative chapters, citations, uncertainty |
| RR7 | Playable game mechanic and devlog | UI/IF | small island, pause/restart, storyboard fallback |
| RR8 | Annotated 360° media room | IF | Three.js panorama, sequential hotspot alternative |
| RR9 | WebGPU route/graph aggregation experiment | IF/IT | capability gate, WebGL/SVG fallback |
| RR10 | Shared worker protocol and replay export | SIM/IT | versioned messages, cancellation, transferables |

## Dependency and risk register

| ID | Dependency/risk | Detection | Mitigation / decision |
| --- | --- | --- | --- |
| X1 | Shared visualization semantics drift | same data encoded differently | typed scales/palettes/legend contract under IF3 |
| X2 | Map/vendor token or tile outage | fixture route fails to render | SVG/Canvas fixture + adapter; no client secret |
| X3 | Solver overclaim | no proof/bound/timeout shown | feasibility/incumbent/bound/gap/status fields required |
| X4 | WebGL context/VRAM growth | ten-navigation heap/context probe | one owner, disposal, visibility suspension, reduced tier |
| X5 | Main-thread jank | long-task/frame profile | worker, event-driven SVG, progressive chunks |
| X6 | Accessibility gap | hover-only or chart-only insight | DOM summary/table, keyboard tree/list, user testing |
| X7 | ML privacy/energy | large download or upload | tiny opt-in model, local-only inputs, static trace |
| X8 | Scope inflation | effect added without claim | item must state visitor question and exit metric |

## Effort × impact matrix

| | High impact | Medium impact | Discovery |
| --- | --- | --- | --- |
| S | RR1 source cards; IT6 budget baseline | UI13 taxonomy cleanup | — |
| M | RR5 audio; RR6 timeline; SIM5 runner | UI8 search; DOC6 embed guide | RR9 WebGPU spike |
| L | RR2 fleet playback; RR3 comparison; RR4 ML replay | RR7 game island; RR8 panorama | RR10 worker protocol hardening |
| XL | — | — | WebGPU splats/XR; live backend solver |

## Anchor index

- [RR research report](research/Interactive%20Features%20and%20Visual%20Storytelling%20Research.md)
- [UI visual system](roadmaps/user_interface.md)
- [IF graphics](roadmaps/interactive_features.md)
- [SIM simulations/Aurelia](roadmaps/simulations_and_aurelia.md)
- [MO optimization](roadmaps/mathematical_optimization.md)
- [ML browser ML](roadmaps/machine_learning.md)
- [IT quality](roadmaps/infrastructure_and_testing.md)
- [DOC documentation](roadmaps/documentation.md)

## Document history

- 2026-08-08 — R2: added research-derived RR1–RR10 index, phase gates, risk register, Image-Toolkit-style timeline/status conventions, and explicit current-state accounting.
- 2026-08-08 — R1: established the immersive portfolio vision and initial feature workstreams.
