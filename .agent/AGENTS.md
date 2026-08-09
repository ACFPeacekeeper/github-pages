# AGENTS.md - Instructions for Coding Assistant LLMs

[![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)

> **Purpose**: Authoritative reference for AI assistants (Claude, GPT, Gemini, Copilot, etc.) working in this repository.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technical Stack](#2-technical-stack)
3. [Module Boundaries](#3-module-boundaries)
4. [Key Commands](#4-key-commands)
5. [Coding Standards](#5-coding-standards)
6. [Known Constraints](#6-known-constraints)

## 1. Project Overview

This is ACFHarbinger's personal website: a statically-exported Next.js blog/knowledge base covering posts, longer-form reports, project write-ups, tool notes, and media, deployed to GitHub Pages at
[acfharbinger.github.io/github-pages](https://acfharbinger.github.io/github-pages/). Content lives as Markdown under `app/content/<section>/` and is rendered through the App Router; `notebooks/` is a small, separate Python/uv workspace used to run the analysis behind some reports (e.g. audio signal processing, PCVRP) before writing them up.

## 2. Technical Stack

| Component | Specification | Notes |
| --- | --- | --- |
| Next.js | 14 (App Router, `output: 'export'`) | Static export deployed to GitHub Pages, `basePath: /github-pages` |
| React / TypeScript | 18 / 5 | `strict: true` in `tsconfig.json` |
| Styling | Tailwind CSS 3 | Config in `tailwind.config.js` |
| Content | Markdown + `gray-matter` / `remark` | Parsed at build time from `app/content/<section>/` |
| Unit tests | Vitest + Testing Library | `test/unit/`, mirroring `src/components/` |
| Integration tests | Vitest + Testing Library + MSW | `test/integration/` |
| E2E / smoke tests | Cypress | `test/cypress/e2e/`, `test/cypress/smoke/` |
| Notebooks | Python 3.11+, managed via `uv` | `notebooks/`, workspace member of the root `pyproject.toml` |

## 3. Module Boundaries

- `app/` — Next.js App Router: routes, layouts, and Markdown content under `app/content/<section>/` (`posts`, `reports`, `projects`, `tools`, `media`, `about`, `other`).
- `src/components/` — presentational and layout React components consumed by `app/`. Business logic (content loading/parsing) belongs in `lib/`, not inline in components.
- `lib/` — server-side helpers (Markdown loading/parsing, front-matter handling) used by `app/` at build time.
- `notebooks/` — independent Python/uv workspace for exploratory analysis backing written reports. Not part of the Next.js build; never imported from `src/`/`app/`.
- `public/` — static assets served as-is.
- `infra/global/` — optional external/public-facing deploy and host tooling (docker, k8s, helm, terraform, ansible, cloud, wordpress). Not used by the default GitHub Pages workflow.
- `infra/private/` — internal developer-only infra experiments (e.g. webpack).

## 4. Key Commands

| Command | Purpose |
| --- | --- |
| `npm run dev` | Local dev server |
| `npm run build` | Static export to `out/` |
| `npm run lint` | ESLint (Next.js config) |
| `npm test` / `npm run test:watch` | Vitest: unit (`test/unit/`) + integration (`test/integration/`) |
| `npm run cypress:run` / `npm run cypress:smoke` | Cypress e2e/smoke (against a running build/dev server) |
| `cd notebooks && uv sync --extra dev` | Set up the notebooks Python environment |

## 5. Coding Standards

- Follow the per-topic rules in [`.agent/rules/`](rules/) (`typescript_react.md`, `python.md`, plus the language-agnostic ones).
- Prefer small, reviewable diffs. Do not reformat files unrelated to the change.
- New components get a Vitest unit test in `test/unit/`; multi-component interactions get an integration test in `test/integration/` (mock any network calls with MSW); new user-facing flows get a Cypress spec in `test/cypress/e2e/`.
- Never commit secrets. This site has no runtime secrets today — flag it clearly if a change would introduce one.

## 6. Known Constraints

- The site is a fully static export (`output: 'export'`) — no server-side code, API routes, or runtime environment variables beyond the build-time `NEXT_PUBLIC_BASE_PATH`.
- `notebooks/` is exploratory/research tooling, not covered by the main CI build; it has its own lint/test story via `uv`.
