locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# Container registry to push infra/global/docker/Dockerfile's built image to,
# for the k8s/helm/ansible deployment paths to pull from.
resource "aws_ecr_repository" "site" {
  name                 = local.name_prefix
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
