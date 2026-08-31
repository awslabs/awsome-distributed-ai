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

variable "local_zone_egress_zone_ids" {
  description = <<-EOT
    Optional list of Local Zone AZ IDs (e.g. usw2-phx2-az1) in which to create
    an LZ-local NAT gateway. When empty (default), no LZ-local NATs are created
    and any private subnet in those LZs will fall back to the regional NAT
    (which hairpins through the parent AZ - measured 24-35 ms per packet).

    When set, one LZ public subnet, one border-group-scoped EIP, and one NAT
    gateway are created per zone ID. The lists local_zone_public_subnet_cidrs
    and local_zone_network_border_groups must be 1:1 with this list.

    Consumed by the private_subnet module via the vpc module's
    nat_gateway_ids_by_zone_id output.
  EOT
  type        = list(string)
  default     = []
}

variable "local_zone_public_subnet_cidrs" {
  description = <<-EOT
    LZ public subnet CIDRs, 1:1 with local_zone_egress_zone_ids. Each CIDR
    must fit within one of the VPC's CIDR blocks (typically the primary CIDR
    since secondary CIDRs are usually consumed by the private worker subnet).
  EOT
  type        = list(string)
  default     = []

  validation {
    condition     = length(var.local_zone_public_subnet_cidrs) == length(var.local_zone_egress_zone_ids)
    error_message = "local_zone_public_subnet_cidrs must have the same length as local_zone_egress_zone_ids."
  }
}

variable "local_zone_network_border_groups" {
  description = <<-EOT
    NetworkBorderGroup names for the LZ NAT EIPs, 1:1 with
    local_zone_egress_zone_ids. Required: a plain vpc-scoped EIP cannot attach
    to a NAT gateway in an LZ subnet. The border-group name is the LZ zone
    name minus the trailing zone letter, e.g. us-west-2-phx-2a -> us-west-2-phx-2,
    us-west-2-lax-1a -> us-west-2-lax-1. Passed explicitly (rather than derived)
    because reliable suffix-strip on multi-letter zone names is awkward in HCL.
  EOT
  type        = list(string)
  default     = []

  validation {
    condition     = length(var.local_zone_network_border_groups) == length(var.local_zone_egress_zone_ids)
    error_message = "local_zone_network_border_groups must have the same length as local_zone_egress_zone_ids."
  }
}

variable "tags" {
  description = "Additional tags for all resources"
  type        = map(string)
  default     = {}
}
