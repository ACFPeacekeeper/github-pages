# TypeScript / React Rules

- Target TypeScript 5, `strict: true` in `tsconfig.json`. No `any` without a `// TODO` explaining why.
- Format/lint with `eslint` (`eslint-config-next`); run `npm run lint` before committing.
- Prefer function components with hooks.
- State management: local `useState`/`useReducer` first; this site has no shared/global store today — don't add one for a single component's state.
- Unit tests live under `test/unit/`, mirroring `src/components/`'s layout, using Vitest + Testing Library. Snapshot tests are a last resort, not a default. Tests spanning more than one component (e.g. anything through `ClientLayoutWrapper`) belong in `test/integration/` instead, mocking any network calls with MSW.
- Remember this is a static export (`output: 'export'`): no API routes, no server components that need a runtime, no browser-only API calls outside `useEffect`/client components.
- Content is Markdown parsed at build time via `lib/markdown.ts`; new content sections belong under `app/content/<section>/`, not hardcoded in a component.
