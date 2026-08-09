# Docker

Self-hosted alternative to the GitHub Pages deployment: builds the Next.js static export and serves it with nginx.

## Quick start

```bash
docker compose -f infra/global/docker/docker-compose.yml up --build
```

Visit `http://localhost:8080`.

## Files

| File | Purpose |
| --- | --- |
| `Dockerfile` | Two-stage build: `npm run build` in a Node image, then serve `out/` with nginx |
| `nginx.conf` | Static-file serving, falling back to the custom `404.html` |
| `docker-compose.yml` | Local stack: the `site` service |
| `docker-compose.prod.yml` | Production overrides (apply with `-f infra/global/docker/docker-compose.yml -f infra/global/docker/docker-compose.prod.yml`) |

## Notes

- Build context is the **repository root**, not `infra/global/docker/` — the Dockerfile needs `package.json` and the rest of the site.
- `next.config.js` sets `basePath: '/github-pages'` for the GitHub Pages deployment. Self-hosting at a domain's root will need that removed or made conditional, or every link will expect the `/github-pages` prefix.
