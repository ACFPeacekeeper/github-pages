# Builds the Next.js static export and serves it with nginx — a
# self-hosted alternative to the GitHub Pages deployment this repo
# actually uses.
#
# Build context is the repository ROOT (see docker-compose.yml:
#   build: { context: .., dockerfile: infra/docker/Dockerfile }).

# ---------------------------------------------------------------------------
# Stage 1 — build the static export
# ---------------------------------------------------------------------------
FROM node:20-slim AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — serve it
# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

COPY infra/docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/out /usr/share/nginx/html

EXPOSE 80
