<div align="center">

# 📓 GitHub Pages — Personal Site & Knowledge Base

**ACFHarbinger's personal blog and knowledge base: posts, reports, project write-ups, and tool notes, statically exported and deployed to GitHub Pages.**

<a href="https://acfharbinger.github.io/github-pages/"><img alt="Live Site" src="https://img.shields.io/badge/live%20site-acfharbinger.github.io-2ea44f?logo=github&logoColor=white"></a>
<a href="https://github.com/ACFHarbinger/github-pages/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ACFHarbinger/github-pages/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://github.com/ACFHarbinger/github-pages/actions/workflows/deploy.yml"><img alt="Deploy" src="https://github.com/ACFHarbinger/github-pages/actions/workflows/deploy.yml/badge.svg"></a>
<a href="https://www.gnu.org/licenses/agpl-3.0"><img alt="License: AGPL v3" src="https://img.shields.io/badge/License-AGPL_v3-blue.svg"></a>

<br>

<a href="https://nextjs.org/"><img alt="Next.js" src="https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white"></a>
<a href="https://react.dev/"><img alt="React" src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white"></a>
<a href="https://www.typescriptlang.org/"><img alt="TypeScript" src="https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white"></a>
<a href="https://tailwindcss.com/"><img alt="Tailwind CSS" src="https://img.shields.io/badge/Tailwind_CSS-3-06B6D4?logo=tailwindcss&logoColor=white"></a>
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white"></a>

<br>

<a href="https://vitest.dev/"><img alt="Vitest" src="https://img.shields.io/badge/Vitest-6E9F18?logo=vitest&logoColor=white"></a>
<a href="https://testing-library.com/"><img alt="Testing Library" src="https://img.shields.io/badge/Testing_Library-E33332?logo=testing-library&logoColor=white"></a>
<a href="https://mswjs.io/"><img alt="MSW" src="https://img.shields.io/badge/MSW-FF6A33?logo=mockserviceworker&logoColor=white"></a>
<a href="https://www.cypress.io/"><img alt="Cypress" src="https://img.shields.io/badge/Cypress-17202C?logo=cypress&logoColor=white"></a>
<a href="https://eslint.org/"><img alt="ESLint" src="https://img.shields.io/badge/ESLint-4B32C3?logo=eslint&logoColor=white"></a>
<a href="https://github.com/astral-sh/uv"><img alt="uv" src="https://img.shields.io/badge/managed%20by-uv-261230.svg"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
<a href="https://jupyter.org/"><img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white"></a>

</div>

---

## Overview

This repository builds [acfharbinger.github.io/github-pages](https://acfharbinger.github.io/github-pages/) — a personal site combining a blog with a small knowledge base:

| Section | What lives there |
| --- | --- |
| **Posts** | Shorter-form writing |
| **Reports** | Longer, structured write-ups (e.g. audio signal processing, the PCVRP report) |
| **Projects** | Write-ups of things I've built |
| **Tools** | Notes on tools/utilities |
| **Media** | Media-focused content |
| **About / Other** | Everything else |

It's a fully static [Next.js](https://nextjs.org/) export (`output: 'export'`) — no server, no API routes, no database. Content is authored as Markdown and rendered at build time.

## Tech Stack

- **Framework:** [Next.js](https://nextjs.org/) (App Router) + [React](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- **Styling:** [Tailwind CSS](https://tailwindcss.com/)
- **Content:** Markdown, parsed with [`gray-matter`](https://github.com/jonschlinkert/gray-matter) and [`remark`](https://github.com/remarkjs/remark)
- **Testing:** [Vitest](https://vitest.dev/) + [Testing Library](https://testing-library.com/) (unit/integration) + [MSW](https://mswjs.io/) (network mocking) · [Cypress](https://www.cypress.io/) (e2e/smoke)
- **Linting:** [ESLint](https://eslint.org/) (`eslint-config-next`)
- **Icons:** [lucide-react](https://lucide.dev/)
- **Research workspace:** [Python](https://www.python.org/) 3.11+ managed with [uv](https://github.com/astral-sh/uv), used for the analysis behind some reports

## Project Structure

```
.
├── app/                    # Next.js App Router: routes + layouts
│   └── content/            # Markdown content, one folder per section
│       ├── posts/
│       ├── reports/
│       ├── projects/
│       ├── tools/
│       ├── media/
│       ├── about/
│       └── other/
├── src/
│   ├── components/         # React components (layout, ui, content wrappers)
│   └── styles/
├── lib/                    # Markdown loading/parsing helpers
├── test/
│   ├── unit/               # Vitest + Testing Library specs, mirrors src/components/
│   ├── integration/        # Vitest + Testing Library + MSW specs
│   └── cypress/            # e2e/ (one spec per section) + smoke/
├── infra/                  # Optional self-hosting / alt-deploy tooling
│   ├── global/             # External (public-facing) deploy & host configs
│   ├── private/            # Internal (developer-only) tooling
│   ├── cloud/              # Managed cloud static-host configs
│   └── server/             # nginx / Envoy reverse-proxy configs
├── notebooks/              # Python/uv workspace for report research
├── docs/moon/research/     # Longer design/research write-ups
└── public/                 # Static assets
```

## Getting Started

Prerequisites: [Node.js](https://nodejs.org/) 20+ and npm.

```bash
git clone https://github.com/ACFHarbinger/github-pages.git
cd github-pages
npm install
npm run dev
```

The dev server runs at `http://localhost:3000/github-pages` (the `basePath` matches the GitHub Pages deployment path).

### Building

```bash
npm run build     # static export to out/
npm run deploy    # build + touch out/.nojekyll
npm start          # serve the out/ export locally
```

## Testing

```bash
npm run lint             # ESLint
npx tsc --noEmit         # type check
npm test                 # Vitest: unit + integration
npm run test:watch       # Vitest in watch mode
npm run test:unit        # Vitest: test/unit/ only
npm run test:integration # Vitest: test/integration/ only
npm run cypress:open     # Cypress e2e + smoke (interactive)
npm run cypress:run      # Cypress e2e + smoke (headless)
npm run cypress:e2e      # Cypress: test/cypress/e2e/ only
npm run cypress:smoke    # Cypress: test/cypress/smoke/ only
```

Cypress runs against a served build, so run `npm run build && npm start` (or `npm run dev`) in another terminal first — see `test/cypress/cypress.config.js` for the `baseUrl`. `npm run build`'s `postbuild` step symlinks `out/github-pages -> .` so `npm start` (a plain static file server) answers under `/github-pages`, matching how GitHub Pages actually serves the site.

## Notebooks

`notebooks/` is a standalone Python workspace (not part of the site build) used to run the analysis behind some reports:

```bash
cd notebooks
uv sync --extra dev
uv run jupyter lab
```

## Deployment

Pushing to `main` triggers [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml), which builds the static export and publishes `out/` to the `gh-pages` branch via [`peaceiris/actions-gh-pages`](https://github.com/peaceiris/actions-gh-pages). [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs lint/typecheck/unit/e2e on every push and PR.

Optional alternatives to GitHub Pages (not used by the default workflow) live under [`infra/`](infra/README.md):

| Path | Purpose |
| --- | --- |
| [`infra/global/docker/`](infra/global/docker/) | Build + serve the export locally with Docker Compose / nginx |
| [`infra/global/k8s/`](infra/global/k8s/) · [`helm/`](infra/global/helm/) · [`terraform/`](infra/global/terraform/) · [`ansible/`](infra/global/ansible/) | Self-host the nginx container on a cluster or plain host |
| [`infra/cloud/`](infra/cloud/) | AWS (S3 + CloudFront / Serverless), Azure Static Web Apps, Firebase configs |
| [`infra/private/webpack/`](infra/private/webpack/) | Developer-only Webpack experiments |
| [`infra/private/wordpress/`](infra/private/wordpress/) | WordPress theme scaffolding for local/CMS experiments |
| [`infra/server/nginx/`](infra/server/nginx/) · [`proxy/`](infra/server/proxy/) | Standalone nginx and Envoy reverse-proxy configs |

Example local self-host:

```bash
docker compose -f infra/global/docker/docker-compose.yml up --build
```

## License

[AGPL-3.0](LICENSE)

---

Get help: [GitHub Discussions](https://github.com/orgs/skills/discussions/categories/github-pages) &bull; [GitHub Status](https://www.githubstatus.com/)

## Documentation website

The documentation website is this Next.js application. Markdown content is read during the Next.js build and emitted as static HTML, so documentation deploys to GitHub Pages without a runtime database.

### Launch locally

```bash
npm install
npm run dev
```

Open `http://localhost:3000/github-pages/`. The `/github-pages` base path mirrors the public deployment URL. For the production export:

```bash
npm run build
npm start
```

Open `http://localhost:3000/github-pages/`. The `postbuild` script creates `out/github-pages -> .` so local static serving reproduces GitHub Pages path resolution.

### Documentation routes

| URL | Purpose | Source |
| --- | --- | --- |
| `/content/posts/` | Short technical notes | `app/content/posts/` |
| `/content/reports/` | Long-form research reports | `app/content/reports/` |
| `/content/projects/` | Project case studies | `app/content/projects/` |
| `/content/tools/` | Tool and workflow notes | `app/content/tools/` |
| `/content/media/` | Anime, film, and television notes | `app/content/media/` |
| `/content/about/` | Biography and contact context | `app/content/about/` |
| `/content/other/` | Miscellaneous essays | `app/content/other/` |
| `/` | Interactive research observatory | `app/page.tsx` |

### Authoring a report

1. Choose the section matching visitor intent.
2. Create a Markdown file with stable filename and valid front matter.
3. Write the claim, assumptions, evidence, limitations, and next action before adding effects.
4. Link local figures and licensed assets with stable relative paths.
5. Run development, typecheck, tests, build, and benchmark.
6. Add a roadmap entry when the report introduces a new interactive commitment.

Do not put secrets, private API responses, unlicensed images, or notebook runtime code in a page. A private experiment needs a reproducible public summary or labelled recorded result.

### Front matter

```yaml
title: A descriptive visitor-facing title
date: 2026-08-08
description: A one-sentence summary used in indexes and metadata.
tags:
  - optimization
  - machine-learning
```

Keep titles concise for cards and browser tabs. Use stable lowercase tags. Dates represent publication or substantive revision.

## Project structure in detail

```text
.
├── app/                         # Next.js routes, layouts, and content pages
├── benchmark/                   # static export performance harness
│   ├── measure.mjs
│   └── README.md
├── docs/                        # architecture, testing, standards, moon roadmaps
│   ├── ARCHITECTURE.md
│   ├── adr/
│   └── moon/{research,roadmaps}/
├── infra/                       # optional self-hosting / alt-deploy tooling
│   ├── global/                  # external public-facing deploy & host configs
│   │   ├── ansible/ docker/ helm/ k8s/ terraform/
│   ├── private/                 # internal developer-only tooling
│   │   └── webpack/ wordpress/
│   ├── cloud/                   # AWS / Azure / Firebase / Serverless
│   └── server/                  # nginx / Envoy reverse-proxy configs
│       └── nginx/ proxy/
├── lib/                         # build-time Markdown/front-matter helpers
├── notebooks/                   # independent Python/uv research workspace
├── public/                      # static, licensed browser assets
├── src/                         # components, state, interfaces, simulations
│   ├── aurelia/                 # optional Aurelia islands
│   ├── components/              # audio, books, canvas, games, graph, image,
│   │   ├── maps/ models/ routes/ video/
│   │   ├── layout/ ui/
│   │   └── ...
│   ├── configs/constants/context/enums/hooks/interfaces/
│   ├── redux/ routes/ types/ utils/
│   └── simulations/{context,generator,repository,scenarios}/
├── test/{unit,integration,cypress}/
└── package.json
```

### Choosing a source directory

- Put a shared type in `src/interfaces`.
- Put a cross-route preference in `src/redux`.
- Put a primitive in `src/components/ui`.
- Put a domain feature in its matching `src/components` subdirectory.
- Put deterministic data in `src/simulations/scenarios`.
- Put simulation contracts in `src/simulations/repository`.
- Put lifecycle orchestration in `src/simulations/context`.
- Put pure transformations in `src/utils`.
- Put route composition in `app/` or `src/routes`.

Avoid adding a generic `components/interactive` directory. Extract the smallest shared contract or hook when a feature crosses domains, while leaving rendering with the owning domain.

## Interactive research features

| Area | Current foundation | Planned slice |
| --- | --- | --- |
| Fleet routing | SVG/canvas routes and deterministic convergence | playback, constraints, solver/Pareto comparisons |
| Machine learning | spectrum and research visual language | policy replay, model cards, local fixtures |
| Game development | prototype card and case-study framing | isolated playable mechanic and devlog graph |
| 3D | capability-aware procedural hero model | reusable glTF viewer and quality tiers |
| 360 media | static-first roadmap contract | panorama room with hotspot list |
| Reading/media | shelf, mosaic, reel, and content routes | source graph and timeline |
| Audio | static spectrum presentation | explicit-gesture Web Audio analysis |

Every feature begins with a semantic fallback and visitor question. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/moon/roadmaps/research_derived_interactions.md`](docs/moon/roadmaps/research_derived_interactions.md).

## Full quality workflow

```bash
npm run lint
npx tsc --noEmit --incremental false
npm test
npm run build
npm run benchmark
```

For browser smoke tests, run `npm run build` and `npm start`, then use `npm run cypress:smoke` in another terminal. The benchmark requires `out/`; it writes ignored `benchmark/results/latest.json` with export bytes, JavaScript/CSS totals, route responses, largest assets, and budget checks. It does not measure real-user Web Vitals or GPU memory.

## Benchmark interpretation

Investigate failures by checking eager dependencies, media sizes, optional-island boundaries, duplicated fallbacks, and documented value. Initial guardrails are 200 kB JavaScript, 80 kB CSS, 2 MB homepage transfer, 3 MB route response, and 1 MB largest asset. Exceptions need a reduced/static alternative and follow-up issue.

## Roadmap and issue workflow

1. Find the workstream in [`docs/moon/ROADMAP.md`](docs/moon/ROADMAP.md).
2. Read its detailed file under [`docs/moon/roadmaps/`](docs/moon/roadmaps/).
3. Create or update the matching GitHub issue and project item.
4. Implement the smallest fallback-first slice.
5. Add tests and benchmark evidence.
6. Update status, changelog, architecture notes, and issue comments.
7. Commit with a focused message.

Use `Backlog` for unstarted work, `In progress` for an active or partial slice, `In review` for a finished change awaiting review, and `Done` only when acceptance criteria and evidence are complete.

## Git workflow

```bash
git status
git diff --check
git add <files>
git commit -m "feat: describe the visitor-facing change"
```

Do not commit `out/`, `.next/`, `tsconfig.tsbuildinfo`, benchmark result JSON, screenshots, videos, local credentials, or notebook caches. Do commit source fixtures, documentation, and reproducibility instructions.

## Environment and configuration

The website has no required runtime environment variables. `next/next.config.js (re-exported at root)` defines the static export and GitHub Pages base path. Browser-only APIs belong inside client components and effects, never during static rendering.

If an optional integration needs a token, prefer a public build-time fixture or remove it from the default site. Never place a secret in `NEXT_PUBLIC_*`, Markdown, `public/`, or a committed notebook.

## Content and media licensing

Before adding an image, model, panorama, audio clip, font, or video:

- record creator, source URL, license, and retrieval date;
- confirm redistribution is allowed;
- provide alt text, captions, poster, or transcript;
- provide a low-cost/static alternative;
- avoid trackers or remote code;
- keep private research data out of public assets.

## Accessibility expectations

The site supports keyboard navigation, visible focus, semantic landmarks, reduced motion, light/dark themes, text summaries, list/table equivalents, and graceful no-WebGL behavior. Interactive charts must not require hover. Audio requires a user gesture. 3D controls need reset/pause and an ordered annotation list. An effect that cannot meet these requirements is optional research, not a default feature.

## Troubleshooting quick reference

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `/` is blank in dev | Base path omitted | Open `/github-pages/`. |
| `npm start` cannot find export | Build has not run | Run `npm run build`. |
| Benchmark says `out/` missing | No production export | Run `npm run benchmark:build`. |
| TypeScript sees stale paths | Simulation directory moved | Check `repository` and `scenarios`. |
| Three.js view is absent | Capability/reduced tier/failure | Use static poster and inspect console. |
| Cypress cannot connect | Server is not running | Run `npm start` in another terminal. |
| Markdown route is missing | Invalid slug/front matter | Inspect filename and run build. |
| Vitest network warning | MSW rejected unhandled request | Add a fixture only when intended. |

## Documentation index

- [Architecture](docs/ARCHITECTURE.md)
- [Moon master roadmap](docs/moon/ROADMAP.md)
- [Research-derived roadmap](docs/moon/roadmaps/research_derived_interactions.md)
- [Interactive research report](docs/moon/research/Interactive%20Features%20and%20Visual%20Storytelling%20Research.md)
- [Testing guide](docs/TESTING.md)
- [Development guide](docs/DEVELOPMENT.md)
- [Documentation standards](docs/DOCUMENTATION_STANDARDS.md)
- [Dependency policy](docs/DEPENDENCY_POLICY.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Maintainer release checklist

- [ ] Content builds with no missing static routes.
- [ ] Lint and typecheck pass.
- [ ] Unit and integration tests pass.
- [ ] Browser smoke tests pass against the production export.
- [ ] Benchmark budgets are green or exceptions documented.
- [ ] New assets have license/provenance records.
- [ ] Reduced-motion, keyboard, no-WebGL, and failure paths exercised.
- [ ] Changelog and roadmap status are current.
- [ ] GitHub issue/project status reflects reality.
- [ ] Commit history explains material architectural decisions.

## Why this structure is explicit

This is a personal site and a public notebook for engineering work. Explicit boundaries show a beautiful result without hiding assumptions, costs, or limitations. A new route story can begin as Markdown and SVG, grow into a focused React island, and only then become a 3D, audio, map, or GPU experiment if evidence and budgets justify it.

That progression protects readers on older devices, keeps GitHub Pages reliable, and makes research claims easier to audit. It gives collaborators a predictable place to add work: contracts in interfaces, behavior in the owning domain, computation in simulations/workers, and evidence in reports and roadmaps.

## README history

| Date | Revision | Change |
| --- | --- | --- |
| 2026-08-08 | R3 | Expanded structure, launch instructions, benchmark workflow, feature guidance, issue workflow, accessibility, licensing, troubleshooting, and release procedures. |
