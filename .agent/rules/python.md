# Python Rules (notebooks/)

- Target Python 3.11+. Use `uv` for dependency management (`uv sync --extra dev`, `uv add`, `uv run`) inside `notebooks/`.
- Format and lint with `ruff` (`ruff format`, `ruff check --fix`). Do not hand-format code that `ruff format` would rewrite.
- Type-check with `mypy` where practical; notebooks are exploratory, so this is advisory, not blocking.
- Strip notebook output before committing (`nbstripout`, wired via `.pre-commit-config.yaml`) — outputs bloat diffs and can leak local file paths.
- Keep reusable logic in plain `.py` modules next to the notebook that uses it; don't duplicate the same analysis code across multiple notebooks.
- `notebooks/` is a standalone workspace, not part of the Next.js build — never import from it in `src/`/`app/`, and don't add it as a build dependency.
