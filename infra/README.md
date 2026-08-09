# infra/

Infrastructure-as-code for self-hosting `github-pages` as an alternative to its actual deployment, which is the GitHub Pages static export via [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml).

Layout is split by audience:

| Directory | Scope | Purpose |
| --- | --- | --- |
| [`global/`](global/) | External / public-facing | Tools used to deploy or host the site for public consumption |
| [`private/`](private/) | Internal / developer-only | Local developer tooling not used for public deployment |

## global/ (external)

Every hosting option here containerizes the same thing: `npm run build` output, served by nginx — or uses an equivalent public static host. Pick one deployment path — don't run more than one against the same target at once.

| Directory | What it does |
| --- | --- |
| `global/docker/` | Build + serve locally via Docker Compose |
| `global/k8s/` | Kubernetes manifests (base + dev/prod kustomize overlays) for the nginx container |
| `global/helm/` | Helm chart wrapping the `k8s/` manifests |
| `global/terraform/` | Cloud provisioning for wherever the container ends up running |
| `global/ansible/` | Playbook for installing/running the container on a plain host |
| `global/cloud/` | AWS / Azure Pipelines / Firebase / Serverless static-hosting configs |

## private/ (internal)

| Directory | What it does |
| --- | --- |
| `private/webpack/` | Webpack config for developer-side bundling experiments |
| `private/wordpress/` | WordPress theme scaffolding for local/CMS experiments |
