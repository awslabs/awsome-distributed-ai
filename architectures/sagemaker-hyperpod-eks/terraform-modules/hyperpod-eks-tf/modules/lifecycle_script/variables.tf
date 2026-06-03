variable "resource_name_prefix" {
  description = "Prefix to be used for all resources created by this module"
  type        = string
  default     = "sagemaker-hyperpod-eks"
}

variable "s3_bucket_name" {
  description = "The name of the S3 bucket used to store the cluster lifecycle scripts"
  type        = string
  default     = "sagemaker-hyperpod-eks-bucket"
}

variable "script_urls" {
  description = "List of raw URLs of the script files to upload"
  type        = list(string)
  default = [
    "https://raw.githubusercontent.com/awslabs/awsome-distributed-ai/main/architectures/sagemaker-hyperpod-eks/LifecycleScripts/base-config/on_create.sh",
    "https://raw.githubusercontent.com/awslabs/awsome-distributed-ai/main/architectures/sagemaker-hyperpod-eks/LifecycleScripts/base-config/on_create_main.sh"
  ]
}
