variable "aws_region" {
  description = "AWS region to deploy the GPU instance into."
  type        = string
  default     = "us-west-2"
}

variable "instance_type" {
  description = <<-EOT
    Single GPU instance type. Defaults to g6e.xlarge (1x L40S, Ada) which supports
    FP8 and INT8 quantization. Step up to p5.48xlarge (H100) for larger models or
    tensor-parallel paths, or p6-b200.48xlarge (Blackwell) to exercise NVFP4.
  EOT
  type        = string
  default     = "g6e.xlarge"
}

variable "root_volume_size_gb" {
  description = "Root EBS volume size in GiB. Needs room for the model weights (~15 GB) + FP8 export (~8 GB) + caches."
  type        = number
  default     = 200
}

variable "project_tag" {
  description = "Value applied to the Project tag on all created resources (used for cleanup/identification)."
  type        = string
  default     = "modelopt-runbook"
}

variable "name_prefix" {
  description = "Name prefix for created resources (IAM role, instance profile, security group, instance)."
  type        = string
  default     = "modelopt-runbook"
}

variable "dlami_ssm_parameter" {
  description = <<-EOT
    SSM Parameter Store path that resolves to the GPU Deep Learning AMI ID.
    Defaults to the base OSS NVIDIA-driver Amazon Linux 2023 DLAMI (driver + Docker,
    no frameworks - the ModelOpt/vLLM pip wheels bring their own CUDA).
  EOT
  type        = string
  default     = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id"
}
