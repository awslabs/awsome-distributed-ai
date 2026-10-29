kubernetes_version    = "1.33"
eks_cluster_name      = "tf-eks-cluster"
hyperpod_cluster_name = "tf-hp-cluster"
resource_name_prefix  = "tf-eks-test"
aws_region            = "us-east-1"
instance_groups = [
  {
    name                      = "accelerated-instance-group-1"
    instance_type             = "ml.g5.8xlarge",
    instance_count            = 2,
    availability_zone_id      = "use1-az2",
    ebs_volume_size_in_gb     = 100,
    threads_per_core          = 1,
    enable_stress_check       = false,
    enable_connectivity_check = false,
    lifecycle_script          = "on_create.sh"
  }
]
create_observability_module               = true
network_metric_level                      = "ADVANCED"
logging_enabled                           = true
create_task_governance_module             = true
create_hyperpod_training_operator_module  = true
create_hyperpod_inference_operator_module = true
enable_guardduty_cleanup                  = true
create_new_fsx_filesystem                 = true


# ================================================================
# Inference Gateway — Example Configuration
# ================================================================
# Enables the Body-Based Router (BBR) and llm-d Endpoint Picker
# (EPP) on the HyperPod Inference Operator add-on.
#
# Prerequisites:
#   - Inference Operator enabled (create_hyperpod_inference_operator_module = true)
#   - cert-manager enabled (enable_cert_manager = true)
#
# The module automatically:
#   - Pins the add-on to v2.0.0-eksbuild.2
#   - Creates an IRSA role for the certificate-issuer (unless
#     you provide your own via inference_gateway_certificate_issuer_role_arn)
#
# After applying, deploy InferenceEndpointConfig resources with
# spec.inferenceGateway.enabled: true — see the example at:
#   examples/inference/sagemaker-hyperpod-inference/inference-gateway-multi-model/
#
# Reference:
#   https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment-inference-gateway.html
# ================================================================

# --- Inference Gateway ---
# Uncomment the lines below to enable.

# enable_cert_manager                              = true
# create_hyperpod_inference_operator_module        = true
# enable_inference_gateway                         = true
# inference_gateway_certificate_issuer_role_arn    = ""    # leave empty to auto-create via IRSA
# tls_certificate_bucket_arn                       = "arn:aws:s3:::my-tls-cert-bucket"
