# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Dark/light theme toggle in the header, with the sidebar keeping the current theme across navigation.
- "AI" section under Tools; "Other" content section.
- PCVRP report and the audio signal processing report/post, with dedicated styling.
- Cypress e2e tests (`cypress/e2e/`) and Jest unit tests (`src/components/__tests__/`).
- License, research, and reports content; notebooks workspace switched from Conda to `uv`.
- Feature-themed roadmaps under `docs/moon/roadmaps/` (user interface, interactive features, mathematical optimization, machine learning, documentation).
- `test/integration/`: RTL integration tests exercising `ClientLayoutWrapper` (Header + Sidebar + Footer composed together) plus an MSW-backed network-layer test (`test/integration/mocks/`).
- `test/cypress/smoke/`: fast Cypress smoke tests — every top-level route renders its layout shell, the homepage logs no console errors, and the theme toggle works.

### Changed

- Replaced Jest with Vitest for unit tests; moved `src/components/__tests__/` to `test/unit/components/`, mirroring `src/components/`'s `layout/`/`ui/` split.
- Moved `cypress/` to `test/cypress/` (config included); CI's Cypress step and the `npm run cypress:*` scripts run from that directory since Cypress resolves spec globs relative to the current working directory, not `--config-file`.
- `npm run build` now runs a `postbuild` step that symlinks `out/github-pages -> .`, so `npm start` (a plain static file server) answers under `/github-pages` locally the same way GitHub Pages does — needed for Cypress/Lighthouse CI jobs that serve the production build rather than `next dev`.

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
