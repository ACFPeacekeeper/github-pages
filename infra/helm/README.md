# helm/

Helm chart packaging the same resources as `infra/k8s/base/`, for anyone who prefers `helm install` over `kubectl apply -k`. Pick one, don't run both against the same cluster/namespace.

```bash
helm lint infra/helm/github-pages
helm install github-pages infra/helm/github-pages -f infra/helm/github-pages/values.yaml
```
