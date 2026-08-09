# Workflow: TypeScript/React Feature

1. Define the component's props/types first; let TypeScript surface integration issues before runtime.
2. Build and test the component in isolation (e.g. via a story or a standalone test) before wiring it into a page.
3. Run `tsc --noEmit` and the linter before committing.
4. Add a Vitest + Testing Library test covering render, interaction, and at least one error/empty state.
5. Verify in a running dev server, not just in tests.
