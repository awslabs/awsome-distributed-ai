variable "resource_name_prefix" {
  description = "Prefix to be used for all resources"
  type        = string
}

variable "eks_cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "hyperpod_cluster_arn" {
  description = "ARN of the HyperPod cluster"
  type        = string
}

variable "access_logs_bucket_name" {
  description = "Name of the S3 bucket for access logs"
  type        = string
}

# S3 CSI Driver (required for for HPIO - only set to false if previously installed)
variable "enable_s3_csi_driver" {
  description = "Install S3 Mountpoint CSI driver EKS addon"
  type        = bool
  default     = true
}

# ALB Controller (required for for HPIO - only set to false if previously installed)
variable "enable_alb_controller" {
  description = "Install the ALB Conroller (bundled with HPIO EKS addon)"
  type        = bool
  default     = true
}

# KEDA (required for for HPIO - only set to false if previously installed)
variable "enable_keda" {
  description = "Install KEDA (bundled with HPIO EKS addon)"
  type        = bool
  default     = true
}

# Metric Server (optional for HPIO)
variable "enable_metrics_server" {
  description = "Install metrics-server EKS addon"
  type        = bool
}
