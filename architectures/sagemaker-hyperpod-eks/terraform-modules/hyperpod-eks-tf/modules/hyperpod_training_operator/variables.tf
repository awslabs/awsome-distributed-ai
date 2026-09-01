variable "resource_name_prefix" {
  description = "Prefix to be used for all resources created by this module"
  type        = string
  default     = "sagemaker-hyperpod-eks"
}

variable "eks_cluster_name" {
  description = "The name of the EKS cluster"
  type        = string
}