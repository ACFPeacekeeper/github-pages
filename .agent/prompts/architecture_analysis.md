# Prompt: Architecture Analysis

Given a request to analyze or propose architecture changes:

1. Read [`.agent/AGENTS.md`](../AGENTS.md) §3 for current module boundaries (`app/`, `src/components/`, `lib/`, `notebooks/`).
2. Identify which of those boundaries the change affects, and whether it's compatible with a static export (`output: 'export'`) — no server runtime.
3. Present trade-offs (at least two options) rather than a single prescriptive answer, unless the choice is clear-cut.
4. Note the migration cost for existing content/pages if the proposal changes how content is loaded or routed.
