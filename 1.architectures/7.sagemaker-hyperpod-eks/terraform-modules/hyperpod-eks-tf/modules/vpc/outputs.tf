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

output "nat_gateway_ids_by_zone_id" {
  description = <<-EOT
    Map of LZ AZ ID -> LZ NAT gateway ID. Consumed by the private_subnet
    module: when a private subnet is placed in one of these zones, its
    default route uses this NAT instead of the regional NAT.
    Empty map when no LZ egress NATs are configured.
  EOT
  value       = zipmap(var.local_zone_egress_zone_ids, aws_nat_gateway.lz_nat[*].id)
}

output "lz_nat_eips_by_zone_id" {
  description = "Map of LZ AZ ID -> LZ NAT public IP. Empty when no LZ egress NATs are configured."
  value       = zipmap(var.local_zone_egress_zone_ids, aws_eip.lz_nat[*].public_ip)
}
