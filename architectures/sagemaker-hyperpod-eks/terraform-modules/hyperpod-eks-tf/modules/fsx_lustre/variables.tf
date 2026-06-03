variable "resource_name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "eks_cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for FSx filesystem"
  type        = string
}

variable "security_group_id" {
  description = "Security group ID"
  type        = string
}

variable "create_new_filesystem" {
  description = "Create new FSx filesystem"
  type        = bool
  default     = false
}

variable "storage_capacity" {
  description = "Storage capacity in GiB"
  type        = number
  default     = 1200
}

variable "throughput" {
  description = "Per unit storage throughput"
  type        = number
  default     = 250
}

variable "data_compression_type" {
  description = "Data compression type"
  type        = string
  default     = "LZ4"
}

variable "file_system_type_version" {
  description = "File system type version"
  type        = string
  default     = "2.15"
}

variable "inference_operator_enabled" {
  description = "Whether inference operator is enabled (requires FSx CSI driver)"
  type        = bool
  default     = false
}

variable "create_fsx_pvc_namespace" {
  description = "Create the namespace for FSx PVC (set to false if namespace already exists)"
  type        = bool
  default     = true
}

variable "fsx_pvc_namespace" {
  description = "Namespace for FSx PVC"
  type        = string
  default     = "default"
}
