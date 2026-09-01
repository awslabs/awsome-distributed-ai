variable "resource_name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "private_subnet_id" {
  description = "ID of the private subnet (for Single-AZ deployments)"
  type        = string
  default     = ""
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs (for Multi-AZ deployments)"
  type        = list(string)
  default     = []
}

variable "deployment_type" {
  description = "FSx OpenZFS deployment type"
  type        = string
  default     = "SINGLE_AZ_1"
  validation {
    condition     = contains(["SINGLE_AZ_1", "SINGLE_AZ_HA_1", "MULTI_AZ_1"], var.deployment_type)
    error_message = "Deployment type must be one of: SINGLE_AZ_1, SINGLE_AZ_HA_1, MULTI_AZ_1."
  }
}

variable "security_group_id" {
  description = "ID of the security group"
  type        = string
}

variable "storage_capacity" {
  description = "Storage capacity in GiB"
  type        = number
}

variable "throughput_capacity" {
  description = "Throughput capacity in MBps"
  type        = number
}

variable "compression_type" {
  description = "Data compression type"
  type        = string
}
