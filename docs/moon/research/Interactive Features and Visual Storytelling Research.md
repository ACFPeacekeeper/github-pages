# Interactive Features and Visual Storytelling for a Research Portfolio

**Research date:** 2026-08-08  
**Scope:** website features that make artificial intelligence, machine learning, mathematical optimization, fleet routing, game development, anime/film/television, and technical or political history legible through interaction.  
**Repository constraint:** Next.js 14 static export, GitHub Pages hosting, browser execution, no runtime secrets, and progressive enhancement.

## Executive synthesis

A beautiful portfolio is not a collection of effects. The strongest pattern across the academic literature, platform documentation, creative studios, and independent practitioners is a **layered explanation system**:

1. A clear editorial claim and ordinary HTML content establish meaning.
2. A small, direct manipulation exposes one relationship or process.
3. A richer canvas, map, audio, or 3D layer rewards curiosity without hiding the claim.
4. A textual, keyboard, reduced-motion, and low-bandwidth equivalent preserves the same conclusion.
5. A measured result—cost, route gap, confidence, latency, memory, or visual quality—makes the work credible.

This is consistent with Segel and Heer’s narrative-visualization design space: a story should balance author-directed narrative flow with reader-directed discovery ([IEEE/PubMed record](https://pubmed.ncbi.nlm.nih.gov/20975152/)). It is also consistent with accessibility research from MIT and Microsoft: charts should expose structure, navigation, description, and agency to screen-reader users, not merely ship an image with alt text ([MIT Rich Screen Reader Experiences](https://vis.csail.mit.edu/pubs/rich-screen-reader-vis-experiences/), [Microsoft Chart Reader](https://www.microsoft.com/en-us/research/publication/chart-reader-accessible-visualization-experiences-designed-with-screen-reader-users/)).

## Research method and source quality

Sources were sampled from four complementary groups. Academic sources establish interaction and evaluation principles; official platform sources establish implementable browser capabilities; corporate/creative examples establish art direction and production patterns; practitioner writing exposes implementation trade-offs and failure modes. A source is used for an architectural or design observation only when the linked page makes that capability or finding explicit.

| Group | Sources reviewed | What it contributes |
| --- | --- | --- |
| Academic and standards | Segel & Heer narrative visualization; Isenberg et al. survey of immersive analytics; Zong et al. rich screen-reader experiences; interactive ML framework; Manifold model debugging; W3C View Transitions | Narrative pacing, immersion, accessibility, human-in-the-loop explanation, model comparison, transition semantics |
| Browser and graphics platforms | [Three.js fundamentals/examples](https://threejs.org/manual/en/fundamentals.html), [Three.js animation example](https://threejs.org/examples/webgl_animation_skinning_morph.html), [WebGPU on MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API), [Web Audio AnalyserNode](https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode), [W3C View Transitions](https://www.w3.org/TR/css-view-transitions-2/) | WebGL/3D lifecycle, animation state, GPU compute limits, FFT visualization, progressive transitions |
| Geospatial and operations platforms | [deck.gl TripsLayer](https://deck.gl/docs/api-reference/geo-layers/trips-layer), [Mapbox examples](https://docs.mapbox.com/mapbox-gl-js/example/), [Mapbox route-animation case study](https://www.mapbox.com/blog/building-cinematic-route-animations-with-mapboxgl), [Google OR-Tools VRP](https://developers.google.com/optimization/routing/vrp) | Animated fleet paths, 3D route scenes, feature interaction, constrained routing and honest optimality language |
| Corporate and practitioner examples | [Bruno Simon](https://bruno-simon.com/), [Active Theory showcase](https://www.webgpu.com/showcase/active-theory-portfolio/), [Codrops](https://tympanus.net/codrops/), [three.js community gallery](https://discourse.threejs.org/t/mesh3d-gallery-a-curated-gallery-of-3d-web-experiences/91297), [personal portfolio retrospective](https://issamzk.com/blog/rebuilding-my-personal-website/) | Playful 3D navigation, shader/scroll craft, production inspiration, and the importance of a clear portfolio narrative |

### Important evidence boundaries

- WebGPU is powerful but remains limited-availability and secure-context-only in MDN’s compatibility guidance; it must be an optional tier, not the default experience ([MDN WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)).
- `AnalyserNode` is broadly available and exposes FFT/time-domain data suitable for browser audio visualizations ([MDN AnalyserNode](https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode)). Audio still requires an explicit user gesture and a teardown path.
- OR-Tools documents capacity, time-window, resource, and dropped-visit variants, while warning that VRP difficulty grows rapidly and a returned solution may be good rather than optimal ([Google OR-Tools](https://developers.google.com/optimization/routing/vrp)). A portfolio visualization must display feasibility and gap/bound language rather than label every result “optimal.”
- Mapbox and deck.gl demonstrate rich animated paths, 3D models, feature-level interaction, camera choreography, and data-driven styling. Their commercial tokens, map data, and accessibility limitations must be treated as separate product decisions rather than silently assumed dependencies.
- Creative references are inspiration, not evidence that a specific effect improves comprehension. Every effect proposed below needs a small usability and performance check.

## Feature catalogue mapped to this portfolio

### 1. Waste-fleet optimization observatory

**Core story:** “How does a fleet choose routes when capacity, time windows, periodic service, and uncertainty collide?”

- **Animated route playback:** a depot-to-stop map with a scrubber, play/pause, day selector, vehicle selector, route trails, and current load. Use a `TripsLayer`-like typed path model or SVG/Canvas for small examples. The side panel remains a semantic ordered stop list.
- **Scenario compare:** compare heuristic, learned policy, and solver incumbent in synchronized small multiples. Display distance/cost, number of vehicles, capacity utilization, late visits, feasibility, runtime, bound, and gap.
- **Constraint toggles:** turn capacity, time windows, periodic visits, road closure, and dropped-visit penalties on/off. Each toggle updates an explanation of why the feasible region changed.
- **Uncertainty ribbon:** show demand intervals and route robustness, not a false single “best” line. Let visitors inspect which stops cause sensitivity.
- **Pareto frontier:** plot distance, emissions proxy, service lateness, and fleet count; selecting a point updates the map. Include a table and CSV/JSON download.
- **Vehicle digital twin:** a lightweight 3D truck model follows the route in the full tier; a moving marker remains in the reduced/static tiers.
- **Solver timeline:** show incumbent improvements, lower bound, search strategy, and cancellation. This turns an opaque solve into a teachable process.

**Recommended first implementation:** SVG route map + ordered table + deterministic seeded presets. Add Mapbox/deck.gl only after real datasets exceed the documented SVG/Canvas threshold.

### 2. Machine-learning and deep-RL lab

**Core story:** “What did the model learn, how sure is it, and where does a human still need to intervene?”

- **Training replay:** epoch/step scrubber with loss, validation metric, learning-rate, and sample preview. Use a recorded deterministic trace before attempting live browser training.
- **Latent-space explorer:** a 2D/3D projection of project, route, or audio embeddings with class/cluster filters, nearest-neighbor links, and a textual selection summary. State that projection is a view, not proof of semantic truth.
- **Policy trajectory view:** show an RL agent’s state/action/reward path through a tiny routing environment. Step mode and “why this action?” metadata make it educational.
- **Model comparison dashboard:** confusion matrix, calibration/reliability view, threshold slider, error gallery, and dataset-slice filters. Borrow Manifold’s model-comparison framing ([USENIX Manifold](https://www.usenix.org/conference/opml19/presentation/li-lezhi)).
- **Human-in-the-loop correction:** allow a visitor to flag a bad route, misclassified audio segment, or confusing embedding; log locally as an example annotation, never silently upload it. Interactive ML research treats the user/model loop as a first-class design object ([interactive ML framework](https://arxiv.org/abs/1610.05463)).
- **Model card overlay:** show model/version, dataset scope, preprocessing, latency, memory, known failure cases, privacy, and whether the result is illustrative.
- **Audio spectrogram/spectrum:** use `AnalyserNode` for a waveform, FFT bars, onset markers, and an explanation of frequency bands. Provide a prerecorded demo and no-audio mode.

**Recommended first implementation:** recorded convergence and feature traces, then a tiny worker-hosted inference demo. Do not begin with a large generative model.

### 3. Game-development startup showcase

**Core story:** “We are building systems and worlds together.”

- **Playable micro-prototype:** one mechanic, one scene, and one clear goal in a small Canvas/WebGL island. Keyboard, touch, pause, restart, and a non-playable storyboard fallback are required.
- **Mechanics graph:** nodes for input, state, rule, feedback, and consequence; selecting a node reveals a short design note and an implementation artifact.
- **World-building timeline:** camera-scroll chapters for concept art, prototype, playtest insight, and next decision. This follows narrative visualization’s author-guided/discovery balance.
- **Interactive devlog:** video chapter markers, shader/animation toggles, before/after asset comparisons, and a “what changed after playtesting?” annotation.
- **Design-system sandbox:** typography, palette, UI sound, controller affordances, and motion presets shown as live tokens rather than a static screenshot.

**Recommended first implementation:** a deterministic mechanic card and storyboard; only add a full playable island after input, pause, and performance budgets are proven.

### 4. Anime, film, television, and image culture

- **Media constellation:** connect a work to themes, creators, genres, historical context, and personal notes; use the same graph primitives as the research map.
- **Poster/shot mosaic:** hover/focus reveals title, year, medium, visual motif, and source; preserve alt text and a list view.
- **Scene comparison slider:** compare a frame, color palette, storyboard, or animation pass with a draggable split and keyboard “show previous/show next” controls.
- **360° gallery:** place hotspots in a panorama or stylized 3D room, but expose a sequential hotspot list for keyboard/screen-reader visitors.
- **Watch/read timeline:** combine technical papers, political history books, anime, and films into a filtered timeline with “why it matters to my work” annotations.
- **Citation cards:** source, date, claim, confidence, and a link; separate recommendation from evidence.

### 5. Technical and political history reading room

- **Argument map:** claims, evidence, counterclaims, actors, dates, and uncertainty represented as a navigable graph.
- **Primary-source timeline:** document scans or excerpts with synchronized annotations and a plain-text transcription.
- **Concept compare:** side-by-side cards for two authors/periods with a shared vocabulary and a “where they disagree” view.
- **Bibliographic trails:** a citation graph that links portfolio research, papers, books, and resulting implementations; no opaque recommendation score.
- **Reading progress journal:** local-only notes, tags, and revisit dates stored in browser storage with export/delete controls.

## Interaction patterns worth adopting

| Pattern | Use here | Guardrail |
| --- | --- | --- |
| Scrollytelling chapters | PCVRP explanation, research reports, game devlog | Every chapter has a heading, deep link, and static sequence |
| Scrubber + playback | Route traces, training replay, video chapters | Step buttons, current-value announcement, pause on hidden tab |
| Linked views | Map ↔ stop table, latent plot ↔ sample, timeline ↔ citation | Selection is URL-addressable and represented in DOM |
| Small multiples | Solver/model/heuristic comparisons | Shared scales, explicit units, downloadable data |
| Direct manipulation | Capacity slider, threshold slider, panorama look, model orbit | Keyboard equivalent and reset affordance |
| Progressive disclosure | Model cards, evidence details, advanced constraints | The initial claim is visible without opening a panel |
| Cinematic transition | Project cards, media comparisons, route camera | View Transitions enhancement; static navigation remains correct ([W3C](https://www.w3.org/TR/css-view-transitions-2/)) |
| Playable island | One game mechanic or simulation | Explicit load budget, focus management, pause, fallback |

## Architecture recommendation for this repository

### Rendering tiers

| Tier | Trigger | Implementation | Content guarantee |
| --- | --- | --- | --- |
| Static | no WebGL/WebGPU, reduced data, crawler, error | HTML, SVG, tables, poster image, recorded trace | Full claim, values, links, and controls represented in DOM |
| Reduced | low memory, coarse pointer, reduced motion, mobile | SVG/Canvas, lower DPR, event-driven updates, no post-processing | Same interaction model with fewer visual effects |
| Full | capable device and visitor opt-in | Three.js/WebGL, optional Mapbox/deck.gl, audio FFT, animated transitions | Enhanced spatial/temporal expression; never new information only available here |

### Boundaries

- `app/` owns routes and static content; `src/components/<domain>/` owns presentation; `src/interfaces/` owns contracts; `src/simulations/repository/` owns serializable types; `src/simulations/scenarios/` owns presets; `src/simulations/generator/` owns deterministic computation; `src/simulations/context/` owns lifecycle.
- Redux stores only cross-route experience state. A visualization’s hover, cursor, drag, and local form state stay local; worker jobs communicate through versioned discriminated messages.
- WebGL/WebGPU contexts are owned by one model/scene boundary and disposed explicitly. A route change must not create unbounded contexts.
- Heavy ML, routing, or graph generation runs in a worker or recorded trace. The UI gets progress, cancellation, timeout, structured error, and stale-response protection.
- External services (Mapbox tiles, geocoding, hosted media, model downloads) are adapters with a static fixture and no secret in the client bundle.

### Performance and privacy budgets

- Home LCP ≤ 2.5 s, INP ≤ 200 ms, CLS ≤ 0.1 at p75; no optional island may add a main-thread task over 100 ms during initial route load.
- Initial route JavaScript ≤ 200 kB gzip excluding framework runtime; each optional island ≤ 300 kB gzip; initial 3D/media transfer ≤ 2 MB unless explicitly opt-in.
- Full-tier target 60 fps on a representative laptop and 30 fps on constrained devices; suspend animation offscreen/hidden and clamp DPR.
- Local audio, notes, annotations, and model inputs stay local by default. Explain downloads, cache keys, deletion, and third-party map/media requests.
- Do not collect gaze, pointer trails, or performance fingerprints by default. Any future opt-in telemetry must have a purpose, retention, and disable path.

## Prioritized delivery plan

| ID | Feature | Evidence pattern | Scope | Exit gate |
| --- | --- | --- | --- | --- |
| RR1 | Research report citation cards and source graph | narrative visualization + reading room | S | Static graph/list, source URLs, provenance fields |
| RR2 | Fleet route playback with table-linked selection | TripsLayer/Mapbox/OR-Tools | L | Seeded route, scrubber, map/table parity, metrics |
| RR3 | Solver comparison/Pareto panel | OR-Tools + linked views | L | Feasibility, gap/bound, export, deterministic tests |
| RR4 | ML training replay and model card | Manifold + interactive ML | L | Recorded trace, slice/error view, limitations |
| RR5 | Audio spectrogram with local/demo source | AnalyserNode | M | Gesture start, teardown, text summary, no-audio fallback |
| RR6 | Media/reading timeline and argument graph | narrative visualization | M | URL-addressable filters, list fallback, citations |
| RR7 | Game prototype island and devlog chapters | Bruno Simon/creative studio patterns | L | Keyboard/touch, pause/restart, static storyboard, budget |
| RR8 | 360° annotated media room | Three.js panorama pattern | L | Keyboard hotspot order, flat fallback, asset license |
| RR9 | WebGPU compute experiment for route/graph aggregation | WebGPU limitations + deck.gl | XL | Capability gate, WebGL/SVG fallback, device matrix |
| RR10 | Local ML/solver worker protocol and replay export | interactive ML + static constraint | L | Cancellation, transferables, schema, crash recovery |

## Risks and decisions

| Risk | Signal | Mitigation / decision |
| --- | --- | --- |
| Visual novelty hides the research | Visitors can describe effect but not claim | Put claim/metric/CTA before canvas; usability test first viewport |
| WebGL context loss or memory growth | Frame time/heap rises after navigation loop | Singleton boundary, disposal test, hidden-tab suspension |
| Map/vendor lock-in | Token or tile failure blanks route story | SVG fixture and data table; adapter boundary; no token in repo |
| Solver result overclaim | “Optimal” shown without proof | Display feasibility, incumbent, bound, gap, timeout, and solver status |
| ML privacy/energy cost | Large model download or local data upload | Tiny opt-in model, worker, explicit size/privacy, CPU/static fallback |
| Motion sickness/cognitive overload | User cannot pause or follow chapter | reduced-motion, pause, chapter headings, no forced camera motion |
| Accessibility is bolted on late | Hover-only insight or chart with no navigation | DOM-first contract, keyboard tree/list, live summary, data export, user testing |

## Decision log

1. Start with SVG/Canvas and recorded traces; adopt Mapbox/deck.gl/WebGPU only after a dataset or frame-budget threshold is measured.
2. Use Three.js for the model/panorama tier because the repository already has a tested model boundary; keep WebGPU experimental.
3. Use `AnalyserNode` for audio because it is broadly available; never require microphone permission for a portfolio demo.
4. Use framework-neutral simulation contracts so React and the optional Aurelia islands share identical scenarios and results.
5. Treat narrative structure, accessible alternatives, and provenance as feature requirements, not documentation afterthoughts.

## Source register

- Segel, E. and Heer, J. “Narrative Visualization: Telling Stories with Data.” IEEE TVCG, 2010. [PubMed](https://pubmed.ncbi.nlm.nih.gov/20975152/) · [open PDF](https://scivis.github.io/courses/visualstorytelling/segel_heer_2010.pdf)
- Zong, J. et al. “Rich Screen Reader Experiences for Accessible Data Visualization.” EuroVis, 2022. [MIT Visualization Group](https://vis.csail.mit.edu/pubs/rich-screen-reader-vis-experiences/) · [arXiv](https://arxiv.org/abs/2205.04917)
- Isenberg, P. et al. “A Survey of Immersive Analytics.” IEEE TVCG, 2019. [record](https://www.researchgate.net/publication/334641937_Survey_of_Immersive_Analytics)
- Weng, D. et al. “An Interactive Machine Learning Framework.” [arXiv](https://arxiv.org/abs/1610.05463)
- Zhang, X. et al. “Manifold: A Model-Agnostic Visual Debugging Tool for Machine Learning.” [USENIX](https://www.usenix.org/conference/opml19/presentation/li-lezhi)
- Three.js. [Fundamentals](https://threejs.org/manual/en/fundamentals.html) · [examples](https://threejs.org/examples/?q=3d)
- MDN. [AnalyserNode](https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode) · [WebGPU API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) · [View Transition API](https://developer.mozilla.org/en-US/docs/Web/API/View_Transition_API)
- W3C. [CSS View Transitions Module Level 2](https://www.w3.org/TR/css-view-transitions-2/)
- vis.gl. [deck.gl TripsLayer](https://deck.gl/docs/api-reference/geo-layers/trips-layer)
- Mapbox. [GL JS examples](https://docs.mapbox.com/mapbox-gl-js/example/) · [cinematic route animations](https://www.mapbox.com/blog/building-cinematic-route-animations-with-mapboxgl)
- Google. [OR-Tools VRP guide](https://developers.google.com/optimization/routing/vrp)
- Bruno Simon. [Interactive portfolio](https://bruno-simon.com/)
- Active Theory. [Portfolio showcase](https://www.webgpu.com/showcase/active-theory-portfolio/)
- Codrops. [Creative web experiments](https://tympanus.net/codrops/)
- Three.js community. [mesh3d.gallery showcase](https://discourse.threejs.org/t/mesh3d-gallery-a-curated-gallery-of-3d-web-experiences/91297)
- ZK. [Personal portfolio reconstruction](https://issamzk.com/blog/rebuilding-my-personal-website/)

## Document history

- 2026-08-08 — Initial research synthesis; mapped evidence to fleet-routing, ML, game-development, media, and reading-room features; added RR1–RR10 delivery candidates.
