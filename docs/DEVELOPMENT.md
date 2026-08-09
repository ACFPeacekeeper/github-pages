# Development Guide

## Prerequisites

- **Git**, [Node.js](https://nodejs.org/) >= 20, `npm`
- **Notebooks (optional):** `python` (>= 3.11) + [`uv`](https://github.com/astral-sh/uv)
- `pre-commit` (`pip install pre-commit && pre-commit install`)

## Local Setup

```bash
git clone https://github.com/ACFHarbinger/github-pages.git
cd github-pages
npm install
npm run dev
```

The dev server runs at `http://localhost:3000/github-pages` (the `basePath` matches the GitHub Pages deployment path).

## Notebooks

```bash
cd notebooks
uv sync --extra dev
uv run jupyter lab
```

## Module Execution & Development

| Task | Command |
| --- | --- |
| Dev server | `npm run dev` |
| Static export build | `npm run build` |
| Serve the export locally | `npm start` |
| Lint | `npm run lint` |
| Unit + integration tests | `npm test` / `npm run test:watch` |
| E2E / smoke tests | `npm run cypress:open` / `npm run cypress:run` / `npm run cypress:smoke` |

## Optional self-hosting (`infra/`)

Default deployment is GitHub Pages. Alternative container/cloud tooling lives under [`infra/`](../infra/README.md):

| Task | Command |
| --- | --- |
| Docker Compose (nginx serving `out/`) | `docker compose -f infra/global/docker/docker-compose.yml up --build` |
| Kubernetes (dev overlay) | `kubectl apply -k infra/global/k8s/overlays/dev` |
| Helm install | `helm install github-pages infra/global/helm/github-pages -f infra/global/helm/github-pages/values.yaml` |
| Terraform (ECR) | `cd infra/global/terraform && terraform init && terraform plan -var-file=environments/dev.tfvars` |
| Ansible (plain host) | `cd infra/global/ansible && ansible-playbook -i inventory/hosts.ini playbook.yml` |
| Serverless (S3 static) | `npm run build && npx serverless client deploy --config infra/cloud/serverless/serverless.yml` |
| AWS CloudFormation template | `infra/cloud/aws/cfn-template.yaml` (see stack Outputs for the post-build sync command) |
| Azure Static Web Apps pipeline | `infra/cloud/azure-pipelines/azure-pipelines.yml` (point Azure DevOps at this path) |
