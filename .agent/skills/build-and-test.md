# Skill: Build and Test Everything

Run the full build + test cycle for the site.

```bash
npm run lint          # eslint
npx tsc --noEmit      # type check
npm test              # jest unit tests
npm run build         # static export to out/
npx cypress run       # e2e, against a running dev/build server
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
