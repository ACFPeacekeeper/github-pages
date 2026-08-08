# Interactive Graphics, Data Visualization, 3D and Spatial Roadmap

Goal: make research tangible through direct manipulation. Every canvas is an enhancement over meaningful HTML/SVG content and follows an explicit quality tier.

| ID | Deliverable | Effort | Depends on | Status |
| --- | --- | --- | --- | --- |
| IF1 | Graphics feasibility spike: persistent singleton canvas versus isolated lazy islands; record ADR and memory profile | M | UI6, IT6 | 📋 |
| IF2 | Capability-gated 3D hero model with orbit/pointer response, pause/reset controls, static SVG fallback and context recovery | L | UI3–UI6 | 🚧 |
| IF3 | Shared typed visualization primitives: scales, palettes, legends, tooltips, selection, keyboard roving and accessible summaries | L | UI3, IT6 | 🚧 |
| IF4 | Interactive research constellation showing relationships among AI, optimization, projects and publications | M | IF3 | 🚧 |
| IF5 | Reusable glTF/GLB model viewer with camera presets, annotations, loading progress, quality tiers and asset disposal | L | IF1, IF2, IT8 | 📋 |
| IF6 | Equirectangular 360° panorama viewer with drag/keyboard look, hotspot DOM overlays, minimap and flat-image fallback | L | IF1, IF5 | 📋 |
| IF7 | Audio-reactive signal-processing exhibit using Web Audio FFT, play/pause, user-selected/local media and non-audio demo mode | L | IF2, IT9 | 📋 |
| IF8 | Cinematic but bounded effects: cursor spotlight, card tilt, particles, bloom/noise and page distortion with preference controls | M | UI6, IF2 | 📋 |
| IF9 | Large graph/geospatial renderer evaluation (SVG/Canvas first; Deck.gl only beyond measured thresholds) | L | IF3, IT6 | 📋 |
| IF10 | WebGPU/TSL renderer experiment with WebGL2 and static fallback, shader warm-up and device-limit inspection | L | IF1, IT6 | 🔬 |
| IF11 | Optional 3D Gaussian splat gallery with LOD streaming and strict memory/download gate | XL | IF10, IT8 | 🔬 |
| IF12 | Optional WebXR viewing mode with explicit consent, session lifecycle controls and equivalent desktop navigation | XL | IF5, UI11 | 🔬 |
| IF13 | Domain-specific interactive component catalogue for audio, canvas, graph, maps, models, routes, video, books, images and games | M | UI13 | 🚧 |

## Rendering contract

| Tier | Trigger | Behaviour |
| --- | --- | --- |
| Static | no WebGL, reduced data, crawler, failure | Poster/SVG, semantic annotations and ordinary links; no lost information |
| Reduced | coarse pointer, low memory/performance, reduced motion | Lower DPR/geometry, no post-processing, event-driven rendering, short transitions |
| Full | capable device and visitor preference | 60 fps target, adaptive DPR, enhanced lighting/effects, continuous render only while visible |

## Acceptance criteria

### IF1–IF2 — renderer and first 3D experience

- Compare memory, route persistence, context count, first interaction cost and chunk size before choosing the renderer lifecycle.
- The 3D chunk loads after intent/visibility, never blocks hero copy or CTAs, and reserves its layout space.
- Pointer drag and arrow keys offer equivalent model control; reset/pause buttons have accessible names and visible focus.
- Handle `webglcontextlost`/`webglcontextrestored`; dispose geometry, material, texture, listeners and animation frames on teardown.
- Clamp device pixel ratio and suspend rendering when hidden/offscreen. Record fps and heap/VRAM proxy measurements.

### IF3–IF4 — data interaction

- Data and visual encodings are separate typed modules; invalid/empty datasets return a descriptive state.
- Nodes/series are reachable by keyboard, selection is announced, color is never the only category encoding, and an equivalent textual list/table is present.
- Filtering/highlighting maintains 60 fps for the documented reference dataset and completes within 100 ms on the main thread.
- URLs can link to a selected entity and the visualization does not overwrite browser navigation semantics.

### IF5–IF7 — spatial and audio stories

- Assets use glTF/GLB with mesh/texture compression where supported; budgets and licenses are stored beside the asset manifest.
- Annotations remain DOM content with collision-aware positioning and a non-spatial index.
- Panorama controls constrain pitch, avoid motion sickness, and provide discrete “previous/next hotspot” navigation.
- Audio starts only after a user gesture, exposes playback state, releases `AudioContext` resources, and never uploads local media.

### IF8–IF12 — advanced effects

- Each effect is individually feature-flagged and removable without affecting content flow.
- WebGPU code checks adapter/device limits instead of assuming buffer sizes; compilation is warmed during idle time with visible progress.
- Experimental XR/splat features are opt-in and excluded from default route payloads.

### IF13 — component catalogue

- Every catalogue component documents its input contract, data source, accessibility equivalent, loading behavior, and whether it owns a canvas/audio/video resource.
- Components are independently importable so a future route can lazy-load only the required domain chunk; there is no growing `components/interactive` catch-all.

## R2 research-derived backlog

The research report turns the catalogue into evidence-led packages: RR2 fleet playback, RR3 solver/Pareto comparison, RR4 ML replay/model cards, RR5 audio explanation, RR6 media/reading timeline, RR7 game prototype, RR8 360 room, RR9 optional WebGPU aggregation, and RR10 worker replay protocol. Each package must ship a semantic/list fallback, provenance, reduced-motion mode, and performance measurements before it can be marked Done.

## Document history

| Date | Revision | Change |
|---|---|---|
| 2026-08-08 | R2 | Added research-derived interaction packages and evidence gates. |
