# ================================================================
# Inference Gateway — Outputs
# ================================================================
# Target: modules/hyperpod_cluster/outputs.tf
# Usage:  Append to the existing outputs.tf file.
# ================================================================

output "inference_gateway_cert_issuer_role_arn" {
  description = "ARN of the certificate-issuer IRSA role (auto-created or user-provided)"
  value       = local.inference_gateway_cert_issuer_role_arn
}

output "inference_gateway_enabled" {
  description = "Whether the Inference Gateway is enabled"
  value       = var.enable_inference_gateway
}