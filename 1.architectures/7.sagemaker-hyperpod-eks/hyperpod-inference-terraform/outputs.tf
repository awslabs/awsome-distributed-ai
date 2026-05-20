output "endpoint_url" {
  description = "Public inference endpoint URL"
  value       = "https://${var.domain_name}/v1/chat/completions"
}

output "api_gateway_url" {
  description = "API Gateway execute URL (alternative)"
  value       = local.deploy_apigw ? aws_apigatewayv2_api.inference[0].api_endpoint : "Not deployed yet - set internal_alb_arn and re-apply"
}

output "acm_certificate_arn" {
  description = "ACM certificate ARN"
  value       = aws_acm_certificate.inference.arn
}

output "tls_bucket" {
  description = "S3 bucket storing TLS certificates"
  value       = aws_s3_bucket.tls_certs.id
}

output "namespace" {
  description = "Kubernetes namespace for the inference deployment"
  value       = var.namespace
}

output "next_steps" {
  description = "Instructions"
  value       = local.deploy_apigw ? "Deployment complete! Use x-api-key header to authenticate." : "Phase 1 complete. Wait for ALB, then set internal_alb_arn and api_keys in terraform.tfvars and run: tofu apply"
}
