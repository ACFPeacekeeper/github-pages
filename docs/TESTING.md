# Testing Guide

| Layer | Framework | Command |
| --- | --- | --- |
| Unit (components/logic) | Vitest + Testing Library | `npm run test:unit` |
| Integration (composed components) | Vitest + Testing Library + MSW | `npm run test:integration` |
| E2E (user flows) | Cypress | `npm run cypress:e2e` (needs `npm run build && npm start`, or `npm run dev`, running) |
| Smoke (fast sanity check) | Cypress | `npm run cypress:smoke` |

Tests live under `test/unit/` (mirroring `src/components/`), `test/integration/`, and `test/cypress/` (`e2e/` — one spec per content section — plus `smoke/`).

## Coverage

`npx vitest run --coverage` reports coverage (requires `@vitest/coverage-v8`, not currently installed). There is no dedicated coverage service configured for this repo today.

## Writing Tests

- New components get a unit test in `test/unit/components/` covering render, interaction, and at least one empty/error state.
- New multi-component interactions (e.g. anything wiring into `ClientLayoutWrapper`) get an integration test in `test/integration/`. Mock network calls with MSW (`test/integration/mocks/handlers.ts`) rather than real `fetch`.
- New or changed user-facing flows get a Cypress spec under `test/cypress/e2e/`, scoped to one content section per file. Build-breaking regressions should also be catchable by `test/cypress/smoke/`.
