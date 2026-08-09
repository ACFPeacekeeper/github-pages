# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Flattened the documentation dashboard from `docs/website/react/` into `docs/website/` (PMF-style package root), retargeted Docusaurus/TypeDoc/Storybook paths, and pointed Storybook at `src/frameworks/react/components/`.
- Moved Astro island routes from `src/pages/` into `src/frameworks/astro/pages/` and pointed `astro.config.mjs` `srcDir` at `./src/frameworks/astro` so Astro sources (pages, components, SFCs) live under one framework tree.

### Added

- Docs website parity modules under `docs/website/src/`: `configs/`, `constants/`, `enums/`, `graphql/`, `hooks/`, `interfaces/`, `simulations/`, `utils/`, `stories/` (research lore), `libraries/{form,motion,router,redux}`, `frameworks/react` + `frameworks/astro` (ResearchOrbit island + iframe wrapper).
- `docs/website/stack/{eslint,next}` with root `eslint.config.js` / `next.config.js` re-exports; `postcss.config.cjs` + Astro build into `static/astro-island/`; root package scripts `docs:*` and `docs:next:*`.
- Multi-framework platform roadmap ([`docs/moon/roadmaps/multi_framework_platform.md`](roadmaps/multi_framework_platform.md), MFP1–MFP16) covering React host + Vue/Astro/Aurelia islands, GraphQL schema/Apollo singleton, static-export fixtures, and WASM workers, grounded in architecture research and the current `src/frameworks` / `src/graphql` layout.
- Master roadmap R4 updates: workstream table entry, phase gate G6, timeline node R8, MFP implementation slices, risks X9–X11, and research anchors for multi-framework docs.
- Cross-links from simulations and infrastructure roadmaps into MFP for Apollo/WASM/island testing ownership.
- Architecture Decision Record (ADR 0002) for the graphics renderer lifecycle, deciding on isolated lazy islands over a persistent singleton canvas to respect strict bundle budgets (IF1).
- Capability-gated 3D hero model with intersection observer lazy-loading, strict resource disposal, webgl context loss recovery, and performance monitoring (IF2).
- Shared typed visualization primitives (scales, palettes, legends, tooltips) with accessible summaries and keyboard roving, integrating them into the `ResearchConstellation` component (IF3).
- Expanded interactive research constellation with nodes and edges linking core themes to specific projects (PCVRP, Audio) and publications, completing IF4.
- Reusable glTF/GLB model viewer (`ModelViewer.tsx`) with DRACO compression support, camera presets, collision-aware DOM annotations, and loading progress indicators (IF5).
- Equirectangular 360° panorama viewer (`PanoramaViewer.tsx`) with pointer drag, keyboard look, minimap, hotspot DOM overlays, and static image fallback (IF6).
- Audio-reactive signal-processing exhibit (`AudioExhibit.tsx`) with FFT visualization via Web Audio API, user-gesture requirement, and non-audio demo mode (IF7).
- Cinematic bounded effects (`Effects.tsx`) providing toggleable cursor spotlight, card tilt, particles, bloom/noise, and page distortion with reduced-motion preference awareness (IF8).
- Astro island architecture simulation via Web Components, mirroring the Aurelia integration pattern for multi-framework rendering.
- Architecture Decision Record (ADR 0003) and `GeospatialRenderer` component evaluating and implementing a progressive Canvas 2D / SVG hybrid rendering strategy for large graph/geospatial data (IF9).
- Research-observatory homepage foundation with semantic visual tokens, an accessible interactive research constellation, a capability-aware Three.js model, and a deterministic optimization-convergence simulation.
- Framework-neutral simulation layers under `src/simulations/` and a lazy Aurelia 2 island boundary under `src/aurelia/` so simulation engines can be shared across frontends.
- Unit and interaction tests for visualization utilities, constellation selection, deterministic simulation generation, lifecycle transitions, and simulation controls.
- A root ESLint configuration so `npm run lint` performs a non-interactive `next/core-web-vitals` check.
- Moved shared content types to `src/interfaces/types.ts`; implemented typed Redux actions, reducers, store hooks, provider, and browser-safe theme persistence under `src/redux/`.
- Reorganized interactive domains into `components/audio`, `books`, `canvas`, `games`, `graph`, `image`, `maps`, `models`, `routes`, and `video`; added fleet routing, ML spectrum, research shelf, game prototype, media mosaic, and storyboard elements.
- A research-driven, milestone-based product roadmap for progressively delivering an accessible visual system, interactive data stories, 3D/360 experiences, browser ML, and mathematical-optimization labs.
- Detailed acceptance criteria, dependency maps, device-quality tiers, performance budgets, fallbacks, accessibility requirements, and release gates across every feature roadmap.
- Dark/light theme toggle in the header, with the sidebar keeping the current theme across navigation.
- "AI" section under Tools; "Other" content section.
- PCVRP report and the audio signal processing report/post, with dedicated styling.
- Cypress e2e tests (`cypress/e2e/`) and Jest unit tests (`src/components/__tests__/`).
- License, research, and reports content; notebooks workspace switched from Conda to `uv`.
- Feature-themed roadmaps under `docs/moon/roadmaps/` (user interface, interactive features, mathematical optimization, machine learning, documentation).
- Renamed simulation boundaries to `src/simulations/repository` (contracts/types) and `src/simulations/scenarios` (presets), with imports and tests updated.
- Added the [Interactive Features and Visual Storytelling Research report](research/Interactive%20Features%20and%20Visual%20Storytelling%20Research.md), covering academic, standards, geospatial, corporate, and practitioner references for fleet optimization, ML, games, media, books, 3D/360, audio, and WebGPU.
- Added Image-Toolkit-style roadmap parity: timeline/phase gates, stable RR1–RR10 packages, acceptance evidence, risk registers, effort matrix, and document history across all moon roadmaps.
- `test/integration/`: RTL integration tests exercising `ClientLayoutWrapper` (Header + Sidebar + Footer composed together) plus an MSW-backed network-layer test (`test/integration/mocks/`).
- `test/cypress/smoke/`: fast Cypress smoke tests — every top-level route renders its layout shell, the homepage logs no console errors, and the theme toggle works.
- Root `benchmark/` performance harness with static-export route checks, payload totals, largest-asset reporting, configurable budgets, and reproducible JSON output.
- Comprehensive root README documentation for launching the documentation website, navigating the new source structure, running benchmarks, authoring content, and maintaining roadmap/issue state.
- Expanded `docs/ARCHITECTURE.md` with system/build/request Mermaid diagrams, module contracts, rendering tiers, simulation/Redux/worker boundaries, and TypeScript/React implementation excerpts.

### Changed

- Replaced Jest with Vitest for unit tests; moved `src/components/__tests__/` to `test/unit/components/`, mirroring `src/components/`'s `layout/`/`ui/` split.
- Moved `cypress/` to `test/cypress/` (config included); CI's Cypress step and the `npm run cypress:*` scripts run from that directory since Cypress resolves spec globs relative to the current working directory, not `--config-file`.
- `npm run build` now runs a `postbuild` step that symlinks `out/github-pages -> .`, so `npm start` (a plain static file server) answers under `/github-pages` locally the same way GitHub Pages does — needed for Cypress/Lighthouse CI jobs that serve the production build rather than `next dev`.
- Infrastructure roadmap now includes the R3 benchmark implementation slice and documents the initial baseline's existing media and aggregate-bundle budget exceptions.
- Fixed the master roadmap Mermaid timeline by replacing invalid chained class syntax with portable `class` assignments; benchmark output now includes a Markdown review summary alongside JSON.

## [0.2.0] — migration to Next.js

### Changed

- Moved from Gatsby to Next.js as the site framework.
- Updated PostCSS/Tailwind configuration for the new framework.

## [0.1.0] — Gatsby + TypeScript skeleton

### Added

- Replaced the previous Jekyll implementation with a Gatsby + React + TypeScript skeleton.
- Jupyter notebooks, layouts/pages, doc assets, and the markdown generator for publications/talks/references.

## [0.0.1] — Jekyll (minima)

### Added

- Initial site built on Jekyll's `minima` theme: layouts, Sass styles, example posts, and GitHub Pages config.
