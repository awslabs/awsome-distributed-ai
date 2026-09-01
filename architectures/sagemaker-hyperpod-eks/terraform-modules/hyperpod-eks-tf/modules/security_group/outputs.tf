output "security_group_id" {
  description = "The ID of the security group"
  value       = local.security_group_id
}

output "security_group_name" {
  description = "The name of the security group"
  value       = var.create_new_sg ? aws_security_group.no_ingress[0].name : null
}

output "security_group_vpc_id" {
  description = "The VPC ID of the security group"
  value       = var.create_new_sg ? aws_security_group.no_ingress[0].vpc_id : null
}

output "security_group_arn" {
  description = "The ARN of the security group"
  value       = var.create_new_sg ? aws_security_group.no_ingress[0].arn : null
}
