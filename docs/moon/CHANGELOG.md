# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
