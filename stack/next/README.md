# stack/next/

Next.js project configuration and TypeScript env references for the App Router host.

| File | Role |
| --- | --- |
| `next.config.js` | Static export, `basePath`, image unoptimized, public env |
| `next-env.d.ts` | Auto-managed Next.js TypeScript references (do not hand-edit) |

Root `next.config.js` re-exports this file so the Next CLI discovers config at the package root. TypeScript includes `stack/next/next-env.d.ts` via `tsconfig.json`.
