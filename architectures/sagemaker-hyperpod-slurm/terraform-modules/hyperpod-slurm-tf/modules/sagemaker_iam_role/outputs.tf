output "sagemaker_iam_role_name" {
  description = "Name of the SageMaker IAM role"
  value       = aws_iam_role.sagemaker_execution_role.name
}

output "sagemaker_iam_role_arn" {
  description = "ARN of the SageMaker IAM role"
  value       = aws_iam_role.sagemaker_execution_role.arn
}