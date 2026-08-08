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

<a href="https://jestjs.io/"><img alt="Jest" src="https://img.shields.io/badge/Jest-C21325?logo=jest&logoColor=white"></a>
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
- **Testing:** [Jest](https://jestjs.io/) + [Testing Library](https://testing-library.com/) (unit) · [Cypress](https://www.cypress.io/) (e2e)
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
│   │   └── __tests__/      # Jest + Testing Library specs
│   └── styles/
├── lib/                    # Markdown loading/parsing helpers
├── cypress/e2e/            # End-to-end specs, one per section
├── notebooks/              # Python/uv workspace for report research
├── docs/research/          # Longer design/research write-ups
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
npm run lint       # ESLint
npx tsc --noEmit   # type check
npm test           # Jest unit tests
npm run test:watch # Jest in watch mode
npx cypress open   # Cypress e2e (interactive)
npx cypress run    # Cypress e2e (headless)
```

Cypress runs against a served build, so run `npm run build && npm start` (or `npm run dev`) in another terminal first — see `cypress.config.js` for the `baseUrl`.

## Notebooks

`notebooks/` is a standalone Python workspace (not part of the site build) used to run the analysis behind some reports:

```bash
cd notebooks
uv sync --extra dev
uv run jupyter lab
```

## Deployment

Pushing to `main` triggers [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml), which builds the static export and publishes `out/` to the `gh-pages` branch via [`peaceiris/actions-gh-pages`](https://github.com/peaceiris/actions-gh-pages). [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs lint/typecheck/unit/e2e on every push and PR.

## License

[AGPL-3.0](LICENSE)

---

Get help: [GitHub Discussions](https://github.com/orgs/skills/discussions/categories/github-pages) &bull; [GitHub Status](https://www.githubstatus.com/)
