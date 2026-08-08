# Development Guide

## Prerequisites

- **Git**, [Node.js](https://nodejs.org/) >= 20, `npm`
- **Notebooks (optional):** `python` (>= 3.11) + [`uv`](https://github.com/astral-sh/uv)
- `pre-commit` (`pip install pre-commit && pre-commit install`)

## Local Setup

```bash
git clone https://github.com/ACFHarbinger/github-pages.git
cd github-pages
npm install
npm run dev
```

The dev server runs at `http://localhost:3000/github-pages` (the `basePath` matches the GitHub Pages deployment path).

## Notebooks

```bash
cd notebooks
uv sync --extra dev
uv run jupyter lab
```

## Module Execution & Development

| Task | Command |
| --- | --- |
| Dev server | `npm run dev` |
| Static export build | `npm run build` |
| Serve the export locally | `npm start` |
| Lint | `npm run lint` |
| Unit tests | `npm test` / `npm run test:watch` |
| E2E tests | `npx cypress open` / `npx cypress run` |
