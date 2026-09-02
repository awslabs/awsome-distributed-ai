variable "resource_name_prefix" {
  description = "Prefix to be used for all resources created by this module"
  type        = string
}

variable "vpc_id" {
  description = "The ID of the VPC"
  type        = string
}

variable "private_subnet_cidrs" {
  description = "The IP range (CIDR notation) for the private subnet"
  type        = list(string)
  default     = ["10.1.0.0/16", "10.2.0.0/16", "10.3.0.0/16", "10.4.0.0/16"]
}

variable "nat_gateway_id" {
  description = "The Id of a NAT Gateway to route internet bound traffic (ignored if closed_network is true)"
  type        = string
}

variable "closed_network" {
  description = "Whether to deploy in closed network mode (no NAT gateway routes)"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Additional tags for all resources"
  type        = map(string)
  default     = {}
}
