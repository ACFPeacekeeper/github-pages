# Skill: Build and Test Everything

Run the full build + test cycle for the site.

```bash
npm run lint          # eslint
npx tsc --noEmit      # type check
npm test              # vitest: unit (test/unit/) + integration (test/integration/)
npm run build         # static export to out/
npm run cypress:run   # e2e + smoke (test/cypress/), against a running dev/build server
```

For the notebooks workspace:

```bash
cd notebooks
uv sync --extra dev
uv run ruff check .
uv run mypy .          # advisory
```

Use this before opening a PR, or whenever asked to "make sure everything still works."
Report which step failed and the first failing assertion/error, not just "tests failed."
