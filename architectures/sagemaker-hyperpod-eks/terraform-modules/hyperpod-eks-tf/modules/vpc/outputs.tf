output "vpc_id" {
  description = "The ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr" {
  description = "The CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_1_id" {
  description = "The ID of the first public subnet (empty if closed network)"
  value       = var.closed_network ? "" : aws_subnet.public_1[0].id
}

output "public_subnet_2_id" {
  description = "The ID of the second public subnet (empty if closed network)"
  value       = var.closed_network ? "" : aws_subnet.public_2[0].id
}

output "nat_gateway_1_id" {
  description = "The ID of the first NAT Gateway (empty if closed network)"
  value       = var.closed_network ? "" : aws_nat_gateway.nat_1[0].id
}

output "internet_gateway_id" {
  description = "The ID of the Internet Gateway (empty if closed network)"
  value       = var.closed_network ? "" : aws_internet_gateway.main[0].id
}

output "public_route_table_id" {
  description = "The ID of the public route table (empty if closed network)"
  value       = var.closed_network ? "" : aws_route_table.public[0].id
}

output "nat_gateway_1_eip" {
  description = "The Elastic IP address of the first NAT Gateway (empty if closed network)"
  value       = var.closed_network ? "" : aws_eip.nat_1[0].public_ip
}

output "availability_zones" {
  description = "List of availability zones used in the VPC"
  value       = var.closed_network ? [] : [aws_subnet.public_1[0].availability_zone, aws_subnet.public_2[0].availability_zone]
}
