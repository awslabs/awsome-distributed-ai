variable "eks_cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "cilium_mode" {
  description = "Cilium operating mode: overlay, eni, chaining, or custom."
  type        = string
  validation {
    condition     = contains(["overlay", "eni", "chaining", "custom"], var.cilium_mode)
    error_message = "cilium_mode must be one of: overlay, eni, chaining, custom."
  }
}

variable "cilium_version" {
  description = "Cilium Helm chart version to deploy."
  type        = string
  default     = "1.19.4"
}

variable "cilium_helm_values" {
  description = "Custom Helm values merged on top of mode-specific defaults. In custom mode, this IS the entire config."
  type        = any
  default     = {}
}

variable "sagemaker_execution_role_name" {
  description = "Name of the SageMaker execution IAM role (used for ENI mode IAM policy attachment)."
  type        = string
  default     = ""
}
