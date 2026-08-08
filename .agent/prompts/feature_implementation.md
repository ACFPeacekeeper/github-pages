# Prompt: Feature Implementation

Given a feature request:

1. Restate the feature as concrete acceptance criteria.
2. Identify which layer(s) it touches (`app/` routing/content, `src/components/`, `lib/`) and read [`.agent/rules/typescript_react.md`](../rules/typescript_react.md).
3. Propose the smallest design that satisfies the criteria; flag any ambiguity as a question rather than assuming.
4. Implement with tests written alongside the code, not after (Jest for components/logic, Cypress for a new user-facing flow).
5. Confirm the change still builds as a static export (`npm run build`).
