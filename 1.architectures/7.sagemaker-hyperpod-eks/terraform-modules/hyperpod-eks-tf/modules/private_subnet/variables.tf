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

variable "availability_zone_ids" {
  description = <<-EOT
    Optional list of Availability Zone IDs to place subnets in, 1:1 with
    private_subnet_cidrs. When empty (default), AZs are discovered automatically
    (standard opt-in-not-required zones only). Set this to target specific zones,
    including opt-in zones such as Local Zones (e.g. usw2-phx2-az1) that the
    discovery filter excludes.
  EOT
  type        = list(string)
  default     = []

  validation {
    condition     = length(var.availability_zone_ids) == 0 || length(var.availability_zone_ids) == length(var.private_subnet_cidrs)
    error_message = "When set, availability_zone_ids must have the same length as private_subnet_cidrs."
  }
}

variable "nat_gateway_id" {
  description = "The Id of a NAT Gateway to route internet bound traffic (ignored if closed_network is true)"
  type        = string
}

variable "nat_gateway_ids_by_zone_id" {
  description = <<-EOT
    Optional per-AZ-ID NAT gateway override. When a private subnet's AZ ID is
    a key in this map, its default route uses the mapped NAT gateway instead
    of var.nat_gateway_id. Used to route Local Zone subnets to an LZ-local
    NAT gateway. Falls back to var.nat_gateway_id for unmapped AZs.
    Default: empty map (all subnets use var.nat_gateway_id).
  EOT
  type        = map(string)
  default     = {}
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
