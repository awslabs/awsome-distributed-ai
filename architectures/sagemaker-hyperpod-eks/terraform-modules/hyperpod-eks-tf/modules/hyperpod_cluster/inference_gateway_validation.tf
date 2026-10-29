# ================================================================
# Inference Gateway — Cross-Variable Validations
# ================================================================
# Terraform variable validation blocks cannot reference other
# variables. These check blocks enforce the dependency rules
# at plan time using cross-variable conditions.
#
# Requires Terraform >= 1.5.0 for check blocks.
#
# Target: modules/hyperpod_cluster/inference_gateway_validation.tf
# ================================================================

check "inference_gateway_requires_cert_manager" {
  assert {
    condition     = !var.enable_inference_gateway || var.enable_cert_manager
    error_message = "cert-manager must be enabled (enable_cert_manager = true) when using the Inference Gateway. The controller pod requires webhook-server-cert issued by cert-manager."
  }
}

check "inference_gateway_requires_inference_operator" {
  assert {
    condition     = !var.enable_inference_gateway || var.create_hyperpod_inference_operator_module
    error_message = "The Inference Operator must be enabled (create_hyperpod_inference_operator_module = true) when using the Inference Gateway."
  }
}

check "inference_gateway_requires_tls_bucket" {
  assert {
    condition = (
      !var.enable_inference_gateway ||
      var.inference_gateway_certificate_issuer_role_arn != "" ||
      var.tls_certificate_bucket_arn != ""
    )
    error_message = "tls_certificate_bucket_arn is required when enable_inference_gateway = true and no custom certificate_issuer_role_arn is provided."
  }
}