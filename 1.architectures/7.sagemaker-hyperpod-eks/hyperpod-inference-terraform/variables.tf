variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "hyperpod_cluster_arn" {
  description = "SageMaker HyperPod cluster ARN (used in outputs and tags)"
  type        = string
}

variable "eks_cluster_name" {
  description = "EKS cluster name backing the HyperPod cluster"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID of the EKS cluster (used for documentation and future WAF integration)"
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnet IDs (reserved for future NLB or NAT gateway use)"
  type        = list(string)
  default     = []
}

variable "private_subnet_ids" {
  description = "Private subnet IDs where the internal ALB lives (used for API Gateway VPC Link)"
  type        = list(string)
}

variable "domain_name" {
  description = "Custom domain for the inference endpoint (e.g., inference.example.com)"
  type        = string
}

variable "hosted_zone_id" {
  description = "Route 53 hosted zone ID for the domain"
  type        = string
}

variable "namespace" {
  description = "Kubernetes namespace for the inference deployment"
  type        = string
  default     = "hyperpod-ns-customer-alpha"
}

variable "queue_name" {
  description = "Kueue local queue name (required if cluster has task governance; leave empty to skip)"
  type        = string
  default     = ""
}

variable "endpoint_name" {
  description = "Name for the inference endpoint"
  type        = string
  default     = "inference-public"
}

variable "model_id" {
  description = "Hugging Face model ID to deploy"
  type        = string
  default     = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
}

variable "instance_type" {
  description = "SageMaker instance type (must match cluster nodes)"
  type        = string
  default     = "ml.p5.48xlarge"
}

variable "gpu_count" {
  description = "Number of GPUs to allocate per replica"
  type        = number
  default     = 1
}

variable "memory" {
  description = "Memory request/limit per replica"
  type        = string
  default     = "32Gi"
}

variable "cpu" {
  description = "CPU request/limit per replica"
  type        = string
  default     = "8"
}

variable "tensor_parallel_size" {
  description = "Tensor parallel size for vLLM (should match gpu_count)"
  type        = number
  default     = 1
}

variable "max_model_len" {
  description = "Maximum sequence length for vLLM"
  type        = number
  default     = 4096
}

variable "vllm_image" {
  description = "vLLM container image"
  type        = string
  default     = "vllm/vllm-openai:v0.10.1"
}

variable "inference_operator_role_name" {
  description = "IAM role name of the HyperPod inference operator (IRSA)"
  type        = string
}

variable "tls_cert_s3_bucket" {
  description = "S3 bucket for TLS certificate output (starts with 'hyperpod-tls' for auto-perms)"
  type        = string
  default     = ""
}

variable "enable_waf" {
  description = "Reserved for future AWS WAF integration (not yet implemented)"
  type        = bool
  default     = false
}

variable "api_keys" {
  description = "List of API keys for WAF validation (header x-api-key)"
  type        = list(string)
  sensitive   = true
  default     = []
}

variable "internal_alb_arn" {
  description = "Optional override for the internal ALB ARN. Leave empty (default) to auto-discover the operator-managed ALB by its ingress tags in a single apply."
  type        = string
  default     = ""
}

variable "hf_token" {
  description = "HuggingFace API token for gated model access"
  type        = string
  sensitive   = true
  default     = ""
}

variable "throttle_rate_limit" {
  description = "API Gateway sustained request rate limit (requests/second)"
  type        = number
  default     = 10
}

variable "throttle_burst_limit" {
  description = "API Gateway burst request limit"
  type        = number
  default     = 20
}
