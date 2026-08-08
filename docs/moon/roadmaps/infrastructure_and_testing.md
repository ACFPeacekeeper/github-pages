# Software Infrastructure & Test Suite Roadmap

Planned work on the site's tooling, CI, and test infrastructure — see [`docs/moon/ROADMAP.md`](../ROADMAP.md) for the project-level roadmap this rolls up into.

| # | Item | Effort | Status |
| --- | --- | --- | --- |
| IT1 | Replace Jest with Vitest for unit tests; move `src/components/__tests__/` to `test/unit/components/` | S | ✅ Done |
| IT2 | `test/integration/`: RTL integration tests for composed components (`ClientLayoutWrapper`) + an MSW-backed network-layer harness (`test/integration/mocks/`) | M | ✅ Done |
| IT3 | Move `cypress/` to `test/cypress/`; add `test/cypress/smoke/` for fast layout/console-error/theme-toggle checks | S | ✅ Done |
| IT4 | Fix `npm start` (static `serve`) not answering under the `/github-pages` basePath locally — added a `postbuild` step symlinking `out/github-pages -> .` | S | ✅ Done |
| IT5 | Wire CI (GitHub/Gitea/Forgejo/GitLab) to run `test:unit`, `test:integration`, and Cypress (e2e + smoke) on every push/PR | S | ✅ Done |
| IT6 | Add `@vitest/coverage-v8` and publish coverage (Codecov or similar) | S | 📋 Pending |
| IT7 | Fix the two pre-existing Cypress e2e failures in `test/cypress/e2e/navigation.cy.js` and `other.cy.js` (predate the `test/` move) | S | 📋 Pending |
| IT8 | Extend MSW integration coverage once the site adds a real network call (e.g. an ML/optimization demo from the feature roadmaps) | M | 📋 Pending |

> **TODO:** Revisit IT6/IT7 next; IT8 is blocked on `machine_learning.md`/`mathematical_optimization.md` landing a client-side feature that actually fetches something.
