# Skill: Debug a Build or Runtime Failure

1. Get a reliable repro: the exact command (`npm run dev`/`npm run build`/`npm test`/`npm run cypress:run`) and the full error output.
2. Capture the full stack trace/browser console output — do not summarize it.
3. For a build-only failure, check first whether it's a static-export constraint (no API routes, no server-only Node APIs in client components, no `window`/`document` access outside `useEffect`).
4. Bisect via `git bisect` if the failure is a regression against a known-good commit.
5. Once fixed, add a regression test (Vitest for components/logic, Cypress for a broken user flow) and note the root cause in the commit message.
