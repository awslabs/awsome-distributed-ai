output "sagemaker_iam_role_name" {
  description = "SageMaker IAM role Name"
  value       = aws_iam_role.sagemaker_execution_role.name
}

output "sagemaker_iam_role_arn" {
  description = "SageMaker IAM role Arn"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "karpenter_role_arn" {
  description = "Karpenter IAM role Arn"
  value       = aws_iam_role.karpenter_role[*].arn
}
