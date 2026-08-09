# eslint/

ESLint configuration for this repository (Next.js core-web-vitals ruleset).

| File | Role |
| --- | --- |
| `.eslintrc.json` | Extends `eslint-config-next` (`next/core-web-vitals`) |

## Usage

From the repo root:

```bash
npm run lint
# or
npx eslint -c eslint/.eslintrc.json .
```

A root `.eslintrc.cjs` re-exports this config so `next lint` and pre-commit keep working without changing their discovery path.
