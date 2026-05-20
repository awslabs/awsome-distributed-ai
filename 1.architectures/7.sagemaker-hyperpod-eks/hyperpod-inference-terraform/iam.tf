# ============================================================================
# IAM: Add permissions to the inference operator role
# ============================================================================

resource "aws_iam_role_policy" "operator_custom_cert" {
  name = "HyperPodCustomCertAndDNS"
  role = var.inference_operator_role_name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ACMAccess"
        Effect = "Allow"
        Action = [
          "acm:DescribeCertificate",
          "acm:GetCertificate"
        ]
        Resource = aws_acm_certificate.inference.arn
      },
      {
        Sid    = "S3CertUpload"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectTagging"
        ]
        Resource = "${aws_s3_bucket.tls_certs.arn}/*"
      },
      {
        Sid    = "Route53DNS"
        Effect = "Allow"
        Action = [
          "route53:GetHostedZone",
          "route53:ListResourceRecordSets",
          "route53:ChangeResourceRecordSets"
        ]
        Resource = "arn:aws:route53:::hostedzone/${var.hosted_zone_id}"
      }
    ]
  })
}
