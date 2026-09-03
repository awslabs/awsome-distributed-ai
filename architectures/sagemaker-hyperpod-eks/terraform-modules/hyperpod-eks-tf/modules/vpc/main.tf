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
