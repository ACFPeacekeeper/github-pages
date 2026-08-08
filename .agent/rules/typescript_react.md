# TypeScript / React Rules

- Target TypeScript 5, `strict: true` in `tsconfig.json`. No `any` without a `// TODO` explaining why.
- Format/lint with `eslint` + `prettier` (or the project's configured equivalents); run `npm run lint` before committing.
- Prefer function components with hooks; colocate a component's styles/tests next to the component file.
- State management: local `useState`/`useReducer` first; reach for a store (Zustand/Redux) only when state is shared across distant components.
- Tests live under `typescript/test/`, using Vitest + Testing Library. Snapshot tests are a last resort, not a default.
- Never call browser/Node-only APIs from shared code that also runs in a Tauri/Electron `src-tauri` context without a platform check.
