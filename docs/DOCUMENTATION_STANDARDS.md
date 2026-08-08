# Documentation Standards

- **Doc-comments**: TSDoc for TypeScript/React, Google-style docstrings for Python (`notebooks/`, `git/scripts/`). Every exported function/component gets one when its behavior isn't obvious from its signature.
- **Markdown docs** live under `docs/`; each page starts with a one-paragraph summary before any headings.
- **Diagrams**: a simple Markdown table or a small Mermaid diagram inline beats an external diagramming tool for a site this size.
- **Code examples** in docs must be runnable against the current codebase; stale examples are worse than no examples.
- **ADRs** (`docs/adr/`) record decisions, not designs-in-progress — write one only once a decision is made; never edit a merged ADR, supersede it with a new one instead.
- **Inclusive language**: avoid ableist/exclusionary phrasing; the markdown link-checker in `.pre-commit-config.yaml` and [`.github/workflows/docs.yml`](../.github/workflows/docs.yml) also catch dead links.
