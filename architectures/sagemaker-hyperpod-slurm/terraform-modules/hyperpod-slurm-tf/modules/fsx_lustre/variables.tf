variable "resource_name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "private_subnet_id" {
  description = "ID of the private subnet"
  type        = string
}

variable "security_group_id" {
  description = "ID of the security group"
  type        = string
}

variable "storage_capacity" {
  description = "Storage capacity in GiB"
  type        = number
}

variable "throughput_per_unit" {
  description = "Provisioned Read/Write (MB/s/TiB)"
  type        = number
}

variable "compression_type" {
  description = "Data compression type"
  type        = string
}

variable "lustre_version" {
  description = "Lustre software version"
  type        = string
}