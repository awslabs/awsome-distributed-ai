variable "resource_name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "vpc_id" {
  description = "ID of the VPC"
  type        = string
}

variable "private_subnet_cidr" {
  description = "CIDR block for private subnet"
  type        = string
}

variable "availability_zone_id" {
  description = "Availability Zone ID for subnet"
  type        = string
}

variable "nat_gateway_id" {
  description = "ID of the NAT Gateway"
  type        = string
}