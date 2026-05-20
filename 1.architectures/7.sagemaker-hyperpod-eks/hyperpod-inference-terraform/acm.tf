# ============================================================================
# ACM Certificate (publicly trusted)
# ============================================================================

resource "aws_acm_certificate" "inference" {
  domain_name       = var.domain_name
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name      = "hyperpod-inference-${var.endpoint_name}"
    ManagedBy = "terraform"
  }
}

resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.inference.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = var.hosted_zone_id
}

resource "aws_acm_certificate_validation" "inference" {
  certificate_arn         = aws_acm_certificate.inference.arn
  validation_record_fqdns = [for record in aws_route53_record.cert_validation : record.fqdn]
}

# ============================================================================
# S3 Bucket for TLS certificate output
# ============================================================================

locals {
  tls_bucket_name = var.tls_cert_s3_bucket != "" ? var.tls_cert_s3_bucket : "hyperpod-tls-${data.aws_caller_identity.current.account_id}-${var.region}"
}

resource "aws_s3_bucket" "tls_certs" {
  bucket        = local.tls_bucket_name
  force_destroy = true

  tags = {
    Name      = "hyperpod-inference-tls-certs"
    ManagedBy = "terraform"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tls_certs" {
  bucket = aws_s3_bucket.tls_certs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tls_certs" {
  bucket = aws_s3_bucket.tls_certs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
