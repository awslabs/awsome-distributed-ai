# =============================================================================
# Cluster stage outputs
# =============================================================================
# These values are exactly what the root (endpoint) stage needs in its
# terraform.tfvars. After `tofu apply` here, run:
#   tofu output -json | jq -r '...'   (see ../README.md "Stage 1" section)
# =============================================================================

output "region" {
  description = "AWS region"
  value       = var.region
}

output "vpc_id" {
  description = "VPC ID of the HyperPod EKS cluster"
  value       = module.vpc.vpc_id
}

output "public_subnet_ids" {
  description = "Public subnet IDs (used by the API Gateway VPC Link in the endpoint stage)"
  value       = [module.vpc.public_subnet_1_id, module.vpc.public_subnet_2_id]
}

output "private_subnet_ids" {
  description = "Private subnet IDs where the internal ALB and inference pods live"
  value       = module.private_subnet.private_subnet_ids
}

output "security_group_id" {
  description = "Cluster security group ID"
  value       = module.security_group.security_group_id
}

output "eks_cluster_name" {
  description = "EKS cluster name (feed into eks_cluster_name in the endpoint stage)"
  value       = module.eks_cluster.eks_cluster_name
}

output "eks_cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks_cluster.eks_cluster_arn
}

output "hyperpod_cluster_name" {
  description = "SageMaker HyperPod cluster name"
  value       = module.hyperpod_cluster.hyperpod_cluster_name
}

output "hyperpod_cluster_arn" {
  description = "SageMaker HyperPod cluster ARN (feed into hyperpod_cluster_arn in the endpoint stage)"
  value       = module.hyperpod_cluster.hyperpod_cluster_arn
}

output "hyperpod_cluster_status" {
  description = "SageMaker HyperPod cluster status"
  value       = module.hyperpod_cluster.hyperpod_cluster_status
}

output "sagemaker_iam_role_name" {
  description = "SageMaker execution role name"
  value       = module.sagemaker_iam_role.sagemaker_iam_role_name
}

output "inference_operator_role_arn" {
  description = "ARN of the inference operator IRSA role"
  value       = module.hyperpod_inference_operator.inference_operator_role_arn
}

output "inference_operator_role_name" {
  description = "Name of the inference operator IRSA role (feed into inference_operator_role_name in the endpoint stage)"
  value       = element(split("/", module.hyperpod_inference_operator.inference_operator_role_arn), length(split("/", module.hyperpod_inference_operator.inference_operator_role_arn)) - 1)
}

output "tls_certificates_bucket_name" {
  description = "S3 bucket the operator uses to publish TLS certificates"
  value       = module.hyperpod_inference_operator.tls_certificates_bucket_name
}

output "endpoint_stage_tfvars" {
  description = "Snippet to paste into the endpoint-stage terraform.tfvars"
  value       = <<-EOT
    region                       = "${var.region}"
    eks_cluster_name             = "${module.eks_cluster.eks_cluster_name}"
    hyperpod_cluster_arn         = "${module.hyperpod_cluster.hyperpod_cluster_arn}"
    vpc_id                       = "${module.vpc.vpc_id}"
    public_subnet_ids            = ["${module.vpc.public_subnet_1_id}", "${module.vpc.public_subnet_2_id}"]
    private_subnet_ids           = ${jsonencode(module.private_subnet.private_subnet_ids)}
    inference_operator_role_name = "${element(split("/", module.hyperpod_inference_operator.inference_operator_role_arn), length(split("/", module.hyperpod_inference_operator.inference_operator_role_arn)) - 1)}"
  EOT
}
