output "endpoint_url" {
  description = "Public inference endpoint URL"
  value       = "https://${var.domain_name}/v1/chat/completions"
}

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name (fronts the private internal ALB)"
  value       = aws_cloudfront_distribution.inference.domain_name
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = aws_cloudfront_distribution.inference.id
}

output "internal_alb_arn" {
  description = "ARN of the internal ALB fronted by CloudFront (auto-discovered unless overridden)"
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
  value       = "Deployment complete! Call ${var.domain_name} with an x-api-key (or Authorization: Bearer) header. Long-running/streaming requests are supported via CloudFront (no 30s API Gateway cap)."
}
