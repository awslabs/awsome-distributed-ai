output "hyperpod_cluster_name" {
  description = "Name of the HyperPod cluster"
  value       = awscc_sagemaker_cluster.hyperpod_cluster.cluster_name
}

output "hyperpod_cluster_arn" {
  description = "ARN of the HyperPod cluster"
  value       = awscc_sagemaker_cluster.hyperpod_cluster.cluster_arn
}

output "hyperpod_cluster_id" {
  description = "ID of the HyperPod cluster"
  value       = awscc_sagemaker_cluster.hyperpod_cluster.id
}