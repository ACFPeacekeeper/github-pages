# next/

Next.js project configuration and TypeScript env references for the App Router host.

| File | Role |
| --- | --- |
| `next.config.js` | Static export, `basePath`, image unoptimized, public env |
| `next-env.d.ts` | Auto-managed Next.js TypeScript references (do not hand-edit) |

## Usage

Root `next.config.js` re-exports `./next/next.config.js` so the Next CLI (which discovers config at the package root) continues to work. TypeScript includes `next/next-env.d.ts` via `tsconfig.json`.
