# terraform/

Provisions an AWS ECR repository to push the image built by `infra/global/docker/Dockerfile` to, for the `k8s/`/`helm/`/`ansible/` deployment paths to pull from.

```bash
cd infra/global/terraform
terraform init
terraform plan -var-file=environments/dev.tfvars
terraform apply -var-file=environments/dev.tfvars
```

| File | Purpose |
| --- | --- |
| `versions.tf` | Required Terraform + AWS provider versions, remote state backend (commented, fill in before first `init`) |
| `variables.tf` | Input variables |
| `main.tf` | The `aws_ecr_repository` resource |
| `outputs.tf` | `ecr_repository_url` — push the built image here |
| `environments/*.tfvars` | Per-environment variable values |
