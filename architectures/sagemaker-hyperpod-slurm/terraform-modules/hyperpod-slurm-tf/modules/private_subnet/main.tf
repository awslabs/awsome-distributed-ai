resource "aws_subnet" "private" {
  vpc_id               = var.vpc_id
  cidr_block           = var.private_subnet_cidr
  availability_zone_id = var.availability_zone_id

  tags = {
    Name = "${var.resource_name_prefix}-private-subnet"
  }
}

resource "aws_route_table" "private" {
  vpc_id = var.vpc_id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = var.nat_gateway_id
  }

  tags = {
    Name = "${var.resource_name_prefix}-private-rt"
  }
}

resource "aws_route_table_association" "private" {
  subnet_id      = aws_subnet.private.id
  route_table_id = aws_route_table.private.id
}