output "helm_release_name" {
  description = "Name of the Helm release"
  value       = helm_release.hyperpod.name
}

output "helm_release_namespace" {
  description = "Namespace of the Helm release"
  value       = helm_release.hyperpod.namespace
}

output "helm_release_status" {
  description = "Status of the Helm release"
  value       = helm_release.hyperpod.status
}
