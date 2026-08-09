# helm/

Helm chart packaging the same resources as `infra/global/k8s/base/`, for anyone who prefers `helm install` over `kubectl apply -k`. Pick one, don't run both against the same cluster/namespace.

```bash
helm lint infra/global/helm/github-pages
helm install github-pages infra/global/helm/github-pages -f infra/global/helm/github-pages/values.yaml
```
