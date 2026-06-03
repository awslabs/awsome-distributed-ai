variable "resource_name_prefix" {
  description = "Prefix to be used for all resources created by this module"
  type        = string
}

variable "vpc_cidr" {
  description = "The IP range (CIDR notation) for the VPC"
  type        = string
  default     = "10.192.0.0/16"
}

variable "public_subnet_1_cidr" {
  description = "The IP range (CIDR notation) for the public subnet in the first Availability Zone"
  type        = string
  default     = "10.192.10.0/24"
}

variable "public_subnet_2_cidr" {
  description = "The IP range (CIDR notation) for the public subnet in the second Availability Zone"
  type        = string
  default     = "10.192.11.0/24"
}

variable "closed_network" {
  description = "Whether to deploy in closed network mode (no internet gateway, NAT gateway, or public subnets)"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Additional tags for all resources"
  type        = map(string)
  default     = {}
}
