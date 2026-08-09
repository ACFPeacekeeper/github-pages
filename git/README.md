# git/

Local git automation for this repository.

| Directory | Purpose |
| --- | --- |
| `config/` | `automation_rules.yaml` (policy for the backlog sync agent) and `project_labels.json` (label taxonomy) |
| `scripts/` | `agent_tools.py` (ProjectV2 GraphQL client), `sync_backlog.py` (roadmap→board reconciler), `check_commit_ref.py` (commit-message ticket linker) |
| `hooks/` | Local git hooks (`pre-commit`, `post-commit`) plus `install.sh` to symlink them into `.git/hooks/` |

## Setup

```bash
bash git/hooks/install.sh
export PROJECT_ID="PVT_..."      # ProjectV2 node ID, see `gh project view <n> --owner <o> --format json`
export GITHUB_TOKEN="..."        # GitHub token with repo + project scopes
export GEMINI_API_KEY="..."      # Google Gemini API key with project + roadmap scopes
```

## CI

`.github/workflows/agent_sync.yml` runs `git/scripts/sync_backlog.py` on
every push to `docs/moon/ROADMAP.md` or `docs/moon/CHANGELOG.md`, or on demand via
`workflow_dispatch`. It needs two repository secrets
(`PROJECT_AUTOMATION_TOKEN`, `GEMINI_API_KEY`) and one repository variable
(`PROJECT_ID`) configured before it can mutate a live board.
