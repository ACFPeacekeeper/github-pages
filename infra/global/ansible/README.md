# ansible/

Installs Docker on a plain host (VM, bare metal) and runs the container built by `infra/global/docker/Dockerfile`, as an alternative to the `k8s/`/`helm/` paths when there's no cluster.

```bash
cd infra/global/ansible
ansible-playbook -i inventory/hosts.ini playbook.yml
```

Requires the `community.docker` collection: `ansible-galaxy collection install community.docker`.

| Path | Purpose |
| --- | --- |
| `ansible.cfg` | Local Ansible config (inventory path, SSH settings) |
| `inventory/hosts.ini` | Target hosts, grouped |
| `playbook.yml` | Entry-point playbook, applies the `app` role |
| `roles/app/` | Installs Docker, pulls `site_image`, runs it on port 80 |
