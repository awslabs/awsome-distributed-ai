data "aws_availability_zones" "available" {
  state = "available"
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(
    {
      Name = "${var.resource_name_prefix}-SMHP-VPC"
    },
    var.tags
  )
}

# Internet Gateway - only created if NOT closed network
resource "aws_internet_gateway" "main" {
  count  = var.closed_network ? 0 : 1
  vpc_id = aws_vpc.main.id

  tags = merge(
    {
      Name = "${var.resource_name_prefix}-SMHP-IGW"
    },
    var.tags
  )
}

# Public subnets - only created if NOT closed network
resource "aws_subnet" "public_1" {
  count                   = var.closed_network ? 0 : 1
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_1_cidr
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = merge(
    {
      Name = "${var.resource_name_prefix}-SMHP-Public1"
    },
    var.tags
  )
}

resource "aws_subnet" "public_2" {
  count                   = var.closed_network ? 0 : 1
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_2_cidr
  availability_zone       = data.aws_availability_zones.available.names[1]
  map_public_ip_on_launch = true

  tags = merge(
    {
      Name = "${var.resource_name_prefix}-SMHP-Public2"
    },
    var.tags
  )
}

# Elastic IP for NAT Gateway - only created if NOT closed network
resource "aws_eip" "nat_1" {
  count  = var.closed_network ? 0 : 1
  domain = "vpc"

  depends_on = [aws_internet_gateway.main]

  tags = merge(
    {
      Name = "${var.resource_name_prefix}-SMHP-NAT1-EIP"
    },
    var.tags
  )
}

# NAT Gateway - only created if NOT closed network
resource "aws_nat_gateway" "nat_1" {
  count         = var.closed_network ? 0 : 1
  allocation_id = aws_eip.nat_1[0].id
  subnet_id     = aws_subnet.public_1[0].id

  depends_on = [aws_internet_gateway.main]

  tags = merge(
    {
      Name = "${var.resource_name_prefix}-SMHP-NAT1"
    },
    var.tags
  )
}

# Public route table - only created if NOT closed network
resource "aws_route_table" "public" {
  count  = var.closed_network ? 0 : 1
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main[0].id
  }

  tags = merge(
    {
      Name = "${var.resource_name_prefix}-SMHP-Public-Routes"
    },
    var.tags
  )
}

resource "aws_route_table_association" "public_1" {
  count          = var.closed_network ? 0 : 1
  route_table_id = aws_route_table.public[0].id
  subnet_id      = aws_subnet.public_1[0].id
}

resource "aws_route_table_association" "public_2" {
  count          = var.closed_network ? 0 : 1
  route_table_id = aws_route_table.public[0].id
  subnet_id      = aws_subnet.public_2[0].id
}

# ============================================================================
# Local Zone egress: LZ-local NAT gateway(s)
# ============================================================================
# When local_zone_egress_zone_ids is set, we create one LZ public subnet, one
# border-group-scoped EIP, and one NAT gateway per LZ. The corresponding
# private-subnet module then routes 0.0.0.0/0 to its LZ NAT instead of the
# regional NAT (which lives in a standard AZ and hairpins packets 24-35 ms
# across the region link).
#
# All resources are gated on `!closed_network AND length(...) > 0` so this
# is a strictly additive, opt-in feature.

locals {
  lz_egress_enabled = !var.closed_network && length(var.local_zone_egress_zone_ids) > 0
  lz_egress_count   = local.lz_egress_enabled ? length(var.local_zone_egress_zone_ids) : 0
}

resource "aws_subnet" "lz_public" {
  count                   = local.lz_egress_count
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.local_zone_public_subnet_cidrs[count.index]
  availability_zone_id    = var.local_zone_egress_zone_ids[count.index]
  map_public_ip_on_launch = true

  tags = merge(
    {
      Name = "${var.resource_name_prefix}-SMHP-LZ-Public-${count.index + 1}"
    },
    var.tags
  )
}

# NetworkBorderGroup on the EIP is REQUIRED - a region-scoped EIP cannot
# attach to a NAT gateway in an LZ subnet. AWS will reject the association
# with "EIP is not associated with the border group of the subnet".
resource "aws_eip" "lz_nat" {
  count                = local.lz_egress_count
  domain               = "vpc"
  network_border_group = var.local_zone_network_border_groups[count.index]

  depends_on = [aws_internet_gateway.main]

  tags = merge(
    {
      Name = "${var.resource_name_prefix}-SMHP-LZ-NAT-EIP-${count.index + 1}"
    },
    var.tags
  )
}

resource "aws_nat_gateway" "lz_nat" {
  count         = local.lz_egress_count
  allocation_id = aws_eip.lz_nat[count.index].id
  subnet_id     = aws_subnet.lz_public[count.index].id

  depends_on = [aws_internet_gateway.main]

  tags = merge(
    {
      Name = "${var.resource_name_prefix}-SMHP-LZ-NAT-${count.index + 1}"
    },
    var.tags
  )
}

# LZ public subnets share their own route table so they don't drag the
# regional public subnets around. AWS handles the LZ<->IGW path transparently.
resource "aws_route_table" "lz_public" {
  count  = local.lz_egress_count
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main[0].id
  }

  tags = merge(
    {
      Name = "${var.resource_name_prefix}-SMHP-LZ-Public-Routes-${count.index + 1}"
    },
    var.tags
  )
}

resource "aws_route_table_association" "lz_public" {
  count          = local.lz_egress_count
  route_table_id = aws_route_table.lz_public[count.index].id
  subnet_id      = aws_subnet.lz_public[count.index].id
}
