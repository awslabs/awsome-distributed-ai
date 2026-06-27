output "instance_id" {
  description = "EC2 instance ID of the GPU box."
  value       = aws_instance.this.id
}

output "instance_type" {
  description = "Launched instance type."
  value       = aws_instance.this.instance_type
}

output "ami_id" {
  description = "Resolved Deep Learning AMI ID."
  value       = nonsensitive(data.aws_ssm_parameter.dlami.value)
}

output "ssm_start_session_command" {
  description = "Ready-to-paste command to open an SSM shell on the instance."
  value       = "aws ssm start-session --target ${aws_instance.this.id} --region ${var.aws_region}"
}
