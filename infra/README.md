# infra/

Infrastructure-as-code for self-hosting `github-pages` as an alternative to its actual deployment, which is the GitHub Pages static export via [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) (also see [`cloud/`](../cloud/) for AWS/Azure static-hosting configs).

Every option here containerizes the same thing: `npm run build` output, served by nginx. Pick one deployment path — don't run more than one against the same target at once.

| Directory | What it does |
| --- | --- |
| `docker/` | Build + serve locally via Docker Compose |
| `k8s/` | Kubernetes manifests (base + dev/prod kustomize overlays) for the nginx container |
| `helm/` | Helm chart wrapping the `k8s/` manifests |
| `terraform/` | Cloud provisioning for wherever the container ends up running |
| `ansible/` | Playbook for installing/running the container on a plain host |
