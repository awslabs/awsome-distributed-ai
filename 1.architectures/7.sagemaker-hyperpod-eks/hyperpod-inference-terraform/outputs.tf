output "endpoint_url" {
  description = "Public inference endpoint URL"
  value       = "https://${var.domain_name}/v1/chat/completions"
}

output "api_gateway_url" {
  description = "API Gateway execute URL (alternative)"
  value       = aws_apigatewayv2_api.inference[0].api_endpoint
}

output "internal_alb_arn" {
  description = "ARN of the internal ALB fronted by API Gateway (auto-discovered unless overridden)"
  value       = local.internal_alb_arn
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
  value       = "Deployment complete! Call ${var.domain_name} with an x-api-key (or Authorization: Bearer) header."
}
