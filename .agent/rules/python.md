# Python Rules

- Target Python 3.11+. Use `uv` for dependency management (`uv sync`, `uv add`, `uv run`).
- Format and lint with `ruff` (`ruff format`, `ruff check --fix`). Do not hand-format code that `ruff format` would rewrite.
- Type-check with `mypy`; new modules should be clean under `--strict` where practical.
- Use `pathlib.Path` instead of `os.path`; use `dataclasses` or `pydantic` models instead of loose dicts for structured data.
- Tests live under `python/test/`, mirroring the `python/src/` package layout. Shared fixtures go in `python/test/conftest.py`.
- Prefer dependency injection over module-level singletons so components stay testable.
- Log with the standard `logging` module (never `print`) using module-level loggers (`logging.getLogger(__name__)`).
