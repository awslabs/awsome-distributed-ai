output "hpto_iam_role_arn" {
  description = "ARN of the HPTO IAM role"
  value       = aws_iam_role.hpto_role.arn
}

output "hpto_iam_role_name" {
  description = "Name of the HPTO IAM role"
  value       = aws_iam_role.hpto_role.name
}

output "hpto_addon_arn" {
  description = "ARN of the HPTO addon"
  value       = aws_eks_addon.hpto_addon.arn
}