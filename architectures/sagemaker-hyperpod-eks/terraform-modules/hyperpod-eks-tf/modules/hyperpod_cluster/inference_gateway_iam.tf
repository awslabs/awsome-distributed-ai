
# ================================================================
# Inference Gateway — Certificate Issuer IRSA Role
# ================================================================
# Creates an IAM role for the certificate-issuer controller that
# the Inference Gateway needs for TLS termination between the
# ALB and the vLLM pods.
#
# The role is assumed by the Kubernetes service account
# "certificate-issuer" in the "hyperpod-inference-system"
# namespace via IRSA (IAM Roles for Service Accounts).
#
# The Inference Operator add-on and gateway are installed via
# the SageMaker Console Quick Install — this file only provisions
# the IAM role that the gateway references at runtime.
#
# Creation conditions:
#   - enable_inference_gateway = true
#   - inference_gateway_certificate_issuer_role_arn is empty
#     (i.e., the user wants the module to auto-create the role)
#
# If the user provides their own role ARN, these resources are
# skipped entirely (count = 0).
#
# Target: modules/hyperpod_cluster/inference_gateway_iam.tf
# Usage:  Replace the existing file (or create new).
#
# Reference:
#   https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment-inference-gateway.html
# ================================================================

# ── IRSA Role for certificate-issuer ───────────────────────────

resource "aws_iam_role" "inference_gateway_cert_issuer" {
  count = (
    var.enable_inference_gateway &&
    var.inference_gateway_certificate_issuer_role_arn == ""
  ) ? 1 : 0

  name = "${var.eks_cluster_name}-igw-cert-issuer"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = var.oidc_provider_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${var.oidc_provider}:sub" = "system:serviceaccount:hyperpod-inference-system:certificate-issuer"
          "${var.oidc_provider}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })

  tags = merge(var.tags, {
    Purpose = "Inference Gateway certificate-issuer IRSA"
  })
}

# ── Policy: S3 access for TLS certificate storage ─────────────
# Grants least-privilege access to the certificate bucket.
# The certificate-issuer reads and writes TLS certs to S3
# for rotation and distribution to the Envoy sidecars.

resource "aws_iam_role_policy" "inference_gateway_cert_issuer" {
  count = (
    var.enable_inference_gateway &&
    var.inference_gateway_certificate_issuer_role_arn == ""
  ) ? 1 : 0

  name = "inference-gateway-cert-issuer"
  role = aws_iam_role.inference_gateway_cert_issuer[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ]
      Resource = [
        var.tls_certificate_bucket_arn,
        "${var.tls_certificate_bucket_arn}/*"
      ]
    }]
  })
}

# ── Output: resolved role ARN ──────────────────────────────────
# Uses the user-provided ARN if set; otherwise falls back to
# the auto-created role. Consumed by the Quick Install when
# configuring the gateway's certificate-issuer service account.

locals {
  inference_gateway_cert_issuer_role_arn = (
    var.inference_gateway_certificate_issuer_role_arn != ""
    ? var.inference_gateway_certificate_issuer_role_arn
    : try(aws_iam_role.inference_gateway_cert_issuer[0].arn, "")
  )
}

