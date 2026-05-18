variable "resource_name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "hyperpod_cluster_name" {
  description = "Name of the HyperPod cluster"
  type        = string
}

variable "node_recovery" {
  description = "Node recovery mode"
  type        = string
}

variable "instance_groups" {
  description = "List of instance group configurations. Each item includes a name. Optionally include training_plan_arn per group."
  type = list(object({
    name              = string
    instance_type     = string
    instance_count    = number
    ebs_volume_size   = number
    threads_per_core  = number
    lifecycle_script  = string
    training_plan_arn = optional(string)
  }))
}

variable "private_subnet_id" {
  description = "ID of the private subnet"
  type        = string
}

variable "security_group_id" {
  description = "ID of the security group"
  type        = string
}

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for lifecycle scripts"
  type        = string
}

variable "sagemaker_iam_role_name" {
  description = "Name of the SageMaker IAM role"
  type        = string
}
