# Dependencies

| Module | Location | Manifest | Package Manager |
| --- | --- | --- | --- |
| Site | repo root (`app/`, `src/`, `lib/`) | `package.json` | `npm` |
| Notebooks | `notebooks/` | `pyproject.toml` | `uv` |
| Git automation | `git/` | `pyproject.toml` | `uv` |
| Optional self-hosting | `infra/global/` (docker, k8s, helm, terraform, ansible, cloud) | per-tool configs (no shared lockfile) | Docker / kubectl / helm / terraform / ansible / cloud CLIs |
| Dev-only infra | `infra/private/` (webpack, wordpress) | none (sample configs) | — |

See [`DEPENDENCY_POLICY.md`](DEPENDENCY_POLICY.md) for policies on adding, pinning, and upgrading dependencies.
