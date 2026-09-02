variable "resource_name_prefix" {
  description = "Prefix to be used for all resources created by this module"
  type        = string
  default     = "sagemaker-hyperpod-eks"
}

variable "hyperpod_cluster_name" {
  description = "Name of SageMaker HyperPod Cluster"
  type        = string
  default     = "ml-cluster"
}

variable "eks_cluster_arn" {
  description = "The ARN of the EKS cluster"
  type        = string
}

variable "auto_node_recovery" {
  description = "Whether to enable or disable the automatic node recovery feature"
  type        = bool
  default     = true
}

variable "continuous_provisioning_mode" {
  description = "whether to enable continuous node provisioning mode"
  type        = bool
}

variable "instance_groups" {
  description = "List of instance group configurations"
  type = list(object({
    name                      = string
    instance_type             = string
    instance_count            = number
    ebs_volume_size_in_gb     = number
    threads_per_core          = number
    enable_stress_check       = bool
    enable_connectivity_check = bool
    lifecycle_script          = string
    availability_zone_id      = string
    image_id                  = optional(string)
    training_plan_arn         = optional(string)
    labels                    = optional(map(string))
    taints = optional(list(object({
      key    = string
      value  = optional(string)
      effect = string
    })))
  }))
  default = []
}

variable "restricted_instance_groups" {
  description = "List of restricted instance group configurations"
  type = list(object({
    name                             = string
    instance_type                    = string
    instance_count                   = number
    ebs_volume_size_in_gb            = number
    threads_per_core                 = number
    enable_stress_check              = bool
    enable_connectivity_check        = bool
    fsxl_per_unit_storage_throughput = number
    fsxl_size_in_gi_b                = number
    availability_zone_id             = string
    training_plan_arn                = optional(string)
  }))
  default = []
}

variable "sagemaker_iam_role_name" {
  description = "The name of the IAM role that SageMaker will use"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for HyperPod cluster"
  type        = list(string)
}

variable "az_to_subnet_map" {
  description = "Map of availability zone IDs to subnet IDs"
  type        = map(string)
  default     = {}
}

variable "security_group_id" {
  description = "The Id of your cluster security group"
  type        = string
}

variable "eks_cluster_name" {
  description = "The name of the EKS cluster"
  type        = string
}

variable "s3_bucket_name" {
  description = "The name of the S3 bucket used to store the cluster lifecycle scripts"
  type        = string
}

variable "rig_mode" {
  description = "Whether restricted instance groups are configured"
  type        = bool
}

variable "karpenter_autoscaling" {
  description = "Whether to enable Karpenter autoscaling"
  type        = bool
}

variable "karpenter_role_arn" {
  description = "ARN of the Karpenter IAM role"
  type        = string
}

variable "wait_for_nodes" {
  description = "Whether to wait for HyperPod nodes (needed by external modules)"
  type        = bool
  default     = false
}

variable "enable_cert_manager" {
  description = "Whether to install cert-manager EKS add-on"
  type        = bool
  default     = false
}

variable "cert_manager_version" {
  description = "Version of the cert-manager EKS add-on"
  type        = string
  default     = "v1.18.2-eksbuild.2"
}