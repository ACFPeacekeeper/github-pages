# stack/eslint/

ESLint configuration for this repository (Next.js core-web-vitals ruleset).

| File | Role |
| --- | --- |
| `.eslintrc.json` | Extends `eslint-config-next` (`next/core-web-vitals`) |

## Usage

From the repo root:

```bash
npm run lint
# or
npx eslint -c stack/eslint/.eslintrc.json .
```

Root `.eslintrc.cjs` re-exports this config so `next lint` and pre-commit keep working.
