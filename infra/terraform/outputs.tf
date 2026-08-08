output "name_prefix" {
  description = "Resource name prefix in use for this environment."
  value       = local.name_prefix
}

output "ecr_repository_url" {
  description = "Push the image built by infra/docker/Dockerfile here."
  value       = aws_ecr_repository.site.repository_url
}
