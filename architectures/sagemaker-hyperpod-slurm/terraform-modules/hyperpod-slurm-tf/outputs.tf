output "vpc_id" {
  description = "ID of the VPC"
  value       = local.vpc_id
}

output "private_subnet_id" {
  description = "ID of the private subnet"
  value       = local.private_subnet_id
}

output "security_group_id" {
  description = "ID of the security group"
  value       = local.security_group_id
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for lifecycle scripts"
  value       = local.s3_bucket_name
}

output "sagemaker_iam_role_name" {
  description = "Name of the SageMaker IAM role"
  value       = local.sagemaker_iam_role_name
}

output "sagemaker_iam_role_arn" {
  description = "ARN of the SageMaker IAM role"
  value       = var.create_sagemaker_iam_role_module ? module.sagemaker_iam_role[0].sagemaker_iam_role_arn : ""
}

output "fsx_lustre_dns_name" {
  description = "DNS name of the FSx Lustre file system"
  value       = local.fsx_lustre_dns_name
}

output "fsx_lustre_mount_name" {
  description = "Mount name of the FSx Lustre file system"
  value       = local.fsx_lustre_mount_name
}

output "fsx_lustre_id" {
  description = "ID of the FSx Lustre file system"
  value       = var.create_fsx_lustre_module ? module.fsx_lustre[0].fsx_lustre_id : ""
}

output "fsx_openzfs_dns_name" {
  description = "DNS name of the FSx OpenZFS file system"
  value       = local.fsx_openzfs_dns_name
}

output "fsx_openzfs_id" {
  description = "ID of the FSx OpenZFS file system"
  value       = var.create_fsx_openzfs_module ? module.fsx_openzfs[0].fsx_openzfs_id : ""
}

output "hyperpod_cluster_name" {
  description = "Name of the HyperPod cluster"
  value       = var.create_hyperpod_module ? module.hyperpod_cluster[0].hyperpod_cluster_name : ""
}

output "hyperpod_cluster_arn" {
  description = "ARN of the HyperPod cluster"
  value       = var.create_hyperpod_module ? module.hyperpod_cluster[0].hyperpod_cluster_arn : ""
}

output "hyperpod_cluster_id" {
  description = "ID of the HyperPod cluster"
  value       = var.create_hyperpod_module ? module.hyperpod_cluster[0].hyperpod_cluster_id : ""
}