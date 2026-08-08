# Testing Guide

| Layer | Framework | Command |
| --- | --- | --- |
| Unit (components/logic) | Jest + Testing Library | `npm test` |
| E2E (user flows) | Cypress | `npx cypress run` (needs `npm run build && npm start`, or `npm run dev`, running) |
| Notebooks lint | ruff | `cd notebooks && uv run ruff check .` |

Tests live under `src/components/__tests__/` (Jest) and `cypress/e2e/` (one spec per content section: posts, reports, projects, tools, media, about, other).

## Coverage

`npm test -- --coverage` reports Jest coverage. There is no dedicated coverage service configured for this repo today.

## Writing Tests

- New components get a Jest test covering render, interaction, and at least one empty/error state.
- New or changed user-facing flows get a Cypress spec.
- Keep Cypress specs scoped to one content section per file, matching the existing `cypress/e2e/*.cy.js` layout.
