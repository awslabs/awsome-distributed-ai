output "hyperpod_cluster_arn" {
  description = "The ARN of the HyperPod cluster"
  value       = awscc_sagemaker_cluster.hyperpod_cluster.cluster_arn
}

output "hyperpod_cluster_name" {
  description = "The name of the HyperPod cluster"
  value       = awscc_sagemaker_cluster.hyperpod_cluster.cluster_name
}

output "instance_group_names" {
  description = "Names of all instance groups created"
  value       = [for g in var.instance_groups : g.name]
}

output "restricted_instance_group_names" {
  description = "Names of restricted instance groups created"
  value       = [for g in var.restricted_instance_groups : g.name]
}

output "hyperpod_cluster_status" {
  description = "The status of the HyperPod cluster"
  value       = awscc_sagemaker_cluster.hyperpod_cluster.cluster_status

}

output "primary_subnet_id" {
  description = "Subnet ID used by the first instance group"
  value = length(var.instance_groups) > 0 ? local.az_to_subnet[var.instance_groups[0].availability_zone_id] : (
    length(var.restricted_instance_groups) > 0 ? local.az_to_subnet[var.restricted_instance_groups[0].availability_zone_id] : null
  )
}
