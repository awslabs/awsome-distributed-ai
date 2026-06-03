output "private_subnet_ids" {
  description = "List of private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "private_subnet_cidrs" {
  description = "List of private subnet CIDR blocks"
  value       = aws_subnet.private[*].cidr_block
}

output "private_subnet_availability_zone_ids" {
  description = "List of private subnet Availability Zone IDs"
  value       = aws_subnet.private[*].availability_zone_id
}

output "private_route_table_ids" {
  description = "List of private route table IDs"
  value       = aws_route_table.private[*].id
}

output "additional_cidr_association_ids" {
  description = "List of additional CIDR block association IDs"
  value       = aws_vpc_ipv4_cidr_block_association.additional_cidr[*].id
}

output "az_to_subnet_map" {
  description = "Map of availability zone IDs to subnet IDs"
  value = zipmap(
    aws_subnet.private[*].availability_zone_id,
    aws_subnet.private[*].id
  )
}
