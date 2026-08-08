# Architecture

`github-pages` is ACFHarbinger's personal website: a statically-exported Next.js site (blog, reports, project write-ups, tool notes) deployed to GitHub Pages, plus a standalone Python/uv notebooks workspace used for the research behind some reports.

## Overview

The site has two decoupled pieces:
- **Site (`app/`, `src/`, `lib/`)**: the Next.js App Router application — routes/layouts in `app/`, presentational React components in `src/components/`, and build-time Markdown loading/parsing in `lib/`. Built with `output: 'export'`, so there is no server runtime.
- **Notebooks (`notebooks/`)**: an independent Python/uv workspace used to run the exploratory analysis behind certain reports (e.g. audio signal processing, the PCVRP report) before they're written up as Markdown under `app/content/reports/`. Never imported by the site build.

## Module Boundaries

| Module | Language / Framework | Primary Responsibility |
| --- | --- | --- |
| `app/` | Next.js (App Router), TypeScript, React | Routing, layouts, and Markdown content under `app/content/<section>/` |
| `src/components/` | TypeScript, React, Tailwind CSS | Presentational/layout components consumed by `app/` |
| `lib/` | TypeScript | Build-time Markdown/front-matter loading used by `app/` |
| `notebooks/` | Python (>=3.11), uv | Exploratory data analysis backing written reports |

## Architecture Decision Records

Significant decisions are recorded under [`docs/adr/`](adr/) using the [Michael Nygard ADR format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).
