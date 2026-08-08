# Documentation and Content Roadmap

Goal: make every visual experiment maintainable, reproducible and understandable as both a portfolio story and an engineering system.

| ID | Deliverable | Effort | Depends on | Status |
| --- | --- | --- | --- | --- |
| DOC1 | Immersive portfolio architecture research under `docs/moon/research` | L | — | ✅ |
| DOC2 | Architecture Decision Record system under `docs/adr` | S | — | ✅ |
| DOC3 | Keep development, testing, dependency and troubleshooting guides current | S | — | 🚧 |
| DOC4 | Publish an edited research report on accessible progressive immersion | M | DOC1 | 📋 |
| DOC5 | Typed content schema for projects/case studies: outcomes, technologies, metrics, media, visualization data and related work | L | UI9 | 📋 |
| DOC6 | Interactive embed authoring guide with accessibility summaries, loading/fallback slots and asset budgets | M | IF3 | 📋 |
| DOC7 | 3D/360 asset handbook covering capture/export, axes/scale, compression, thumbnails, licensing and annotations | M | IT8 | 📋 |
| DOC8 | Visualization style guide: encodings, palettes, legends, uncertainty, responsive behaviour and data-table equivalence | M | IF3 | 📋 |
| DOC9 | Performance playbook and benchmark journal for renderer, worker, ML and solver decisions | M | IT6 | 📋 |
| DOC10 | Visitor-facing accessibility/motion/data controls help and privacy statement for local computation | S | UI6, ML4 | 📋 |
| DOC11 | Contributor templates/checklists for issues, ADRs, visual features and reproducible project case studies | S | DOC5–DOC9 | 📋 |

## Documentation quality bar

- Each feature document explains purpose, architecture, data flow, browser support, fallback, keyboard model, performance budget, test plan and teardown lifecycle.
- Case studies distinguish measured outcomes from aspirations and link to reproducible code/data when publication constraints allow.
- Diagrams have text alternatives; screenshots include captions; code samples are minimal, tested and compatible with the static export.
- ADRs cover persistent versus isolated canvas, graphics library choice, state ownership, worker protocol, asset formats and any backend fork.
- Roadmap IDs remain stable and map to GitHub issues; completion moves user-visible results to the changelog without deleting historical rationale.

## R2 research integration

The [research report](../research/Interactive%20Features%20and%20Visual%20Storytelling%20Research.md) is the source of truth for RR1–RR10. Each issue and feature page links its academic, standards, geospatial, or practitioner evidence; records licensing and browser support; distinguishes measured outcomes from illustrative demos; and documents the equivalent non-visual interaction. New roadmap entries use stable IDs so project status and changelog history remain auditable.
