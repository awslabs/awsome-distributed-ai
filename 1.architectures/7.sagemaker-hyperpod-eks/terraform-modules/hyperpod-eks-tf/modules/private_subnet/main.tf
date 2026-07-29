data "aws_availability_zones" "available" {
  state = "available"
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

# Subnet placement:
#  - By default, subnets are spread across discovered standard AZs (opt-in-not-required),
#    with count = min(#CIDRs, #AZs).
#  - When availability_zone_ids is set, subnets are placed in those exact AZ IDs
#    (1:1 with private_subnet_cidrs). This allows opt-in zones such as Local Zones,
#    which the discovery filter deliberately excludes.
locals {
  use_explicit_azs = length(var.availability_zone_ids) > 0
  subnet_count = local.use_explicit_azs ? length(var.private_subnet_cidrs) : min(
    length(var.private_subnet_cidrs), length(data.aws_availability_zones.available.names)
  )
  subnet_zone_ids = local.use_explicit_azs ? var.availability_zone_ids : data.aws_availability_zones.available.zone_ids
}

resource "aws_vpc_ipv4_cidr_block_association" "additional_cidr" {
  count      = local.subnet_count
  vpc_id     = var.vpc_id
  cidr_block = var.private_subnet_cidrs[count.index]
}

resource "aws_subnet" "private" {
  count                = local.subnet_count
  vpc_id               = var.vpc_id
  cidr_block           = var.private_subnet_cidrs[count.index]
  availability_zone_id = local.subnet_zone_ids[count.index]

  # Ensure the subnet is created after the CIDR block is associated
  depends_on = [aws_vpc_ipv4_cidr_block_association.additional_cidr]

  tags = merge(
    {
      Name = "${var.resource_name_prefix}-SMHP-Private${count.index + 1}"
    },
    var.tags
  )
}

resource "aws_route_table" "private" {
  count  = local.subnet_count
  vpc_id = var.vpc_id

  tags = merge(
    {
      Name = "${var.resource_name_prefix}-SMHP-Private-Routes-${count.index + 1}"
    },
    var.tags
  )
}

resource "aws_route_table_association" "private" {
  count          = local.subnet_count
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# NAT Gateway route - only created if NOT closed network
resource "aws_route" "nat_gateway" {
  count                  = var.closed_network ? 0 : local.subnet_count
  route_table_id         = aws_route_table.private[count.index].id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = var.nat_gateway_id
}
