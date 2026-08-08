# Prompt: Documentation Update

Given a request to update documentation:

1. Identify every doc surface affected: `README.md`, `docs/research/*.md`, `.agent/AGENTS.md`, inline comments.
2. Match the existing tone and structure of the surrounding document — don't introduce a new format for one section.
3. Verify any commands/code examples actually run against the current codebase before including them.
4. Check for now-stale links elsewhere in the repo that reference the changed section.
