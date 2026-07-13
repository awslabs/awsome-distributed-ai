# =============================================================================
# Cluster stage variables — create a SageMaker HyperPod EKS cluster from scratch
# =============================================================================
# This stage provisions everything needed to host the HyperPod inference
# operator: a VPC, private subnets, security group, EKS cluster, S3 bucket,
# lifecycle scripts, SageMaker IAM role, the HyperPod cluster (with a GPU
# instance group), and the inference operator add-on (with its ALB controller,
# KEDA, S3 CSI, and metrics-server dependencies).
#
# The leaf modules are reused directly from the canonical reference stack at
# ../../terraform-modules/hyperpod-eks-tf/modules so there is a single source
# of truth for the underlying infrastructure.
# =============================================================================

variable "region" {
  description = "AWS region to deploy the HyperPod EKS cluster into"
  type        = string
  default     = "us-east-1"
}

variable "resource_name_prefix" {
  description = "Prefix used for naming all resources created by this stage"
  type        = string
  default     = "hpinf"
}

# --- Naming ---
variable "eks_cluster_name" {
  description = "Name of the EKS cluster backing the HyperPod cluster"
  type        = string
  default     = "hyperpod-inference-eks"
}

variable "hyperpod_cluster_name" {
  description = "Name of the SageMaker HyperPod cluster"
  type        = string
  default     = "hyperpod-inference-cluster"
}

variable "kubernetes_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.31"
}

# --- Networking ---
variable "vpc_cidr" {
  description = "Primary CIDR block for the VPC"
  type        = string
  default     = "10.192.0.0/16"
}

variable "public_subnet_1_cidr" {
  description = "CIDR for the first public subnet (used for NAT gateway and API Gateway VPC Link)"
  type        = string
  default     = "10.192.10.0/24"
}

variable "public_subnet_2_cidr" {
  description = "CIDR for the second public subnet"
  type        = string
  default     = "10.192.11.0/24"
}

variable "private_subnet_cidrs" {
  description = "Additional CIDR blocks for HyperPod private subnets (one per AZ, in zone-id order)"
  type        = list(string)
  default     = ["10.1.0.0/16", "10.2.0.0/16", "10.3.0.0/16", "10.4.0.0/16"]
}

variable "eks_private_subnet_1_cidr" {
  description = "CIDR for the first EKS control-plane private subnet"
  type        = string
  default     = "10.192.7.0/28"
}

variable "eks_private_subnet_2_cidr" {
  description = "CIDR for the second EKS control-plane private subnet"
  type        = string
  default     = "10.192.8.0/28"
}

variable "eks_endpoint_private_access" {
  description = "Enable the private EKS API server endpoint"
  type        = bool
  default     = true
}

variable "eks_endpoint_public_access" {
  description = "Enable the public EKS API server endpoint"
  type        = bool
  default     = true
}

# --- HyperPod instance group (the GPU nodes that serve inference) ---
variable "instance_groups" {
  description = <<-EOT
    List of HyperPod instance group configurations. The default provisions a
    single ml.p5.48xlarge group in use1-az6 (us-east-1c) for inference. Set
    availability_zone_id to match the AZ where your accelerated capacity lives.
    For On-Demand p5.48xlarge you typically need an On-Demand Capacity
    Reservation (ODCR) or a SageMaker training plan; pass the plan via
    training_plan_arn when applicable.
  EOT
  type = list(object({
    name                      = string
    instance_type             = string
    instance_count            = number
    ebs_volume_size_in_gb     = number
    threads_per_core          = number
    enable_stress_check       = bool
    enable_connectivity_check = bool
    lifecycle_script          = string
    availability_zone_id      = string
    image_id                  = optional(string)
    training_plan_arn         = optional(string)
    labels                    = optional(map(string))
    taints = optional(list(object({
      key    = string
      value  = optional(string)
      effect = string
    })))
  }))
  default = [
    {
      name                      = "inference-p5"
      instance_type             = "ml.p5.48xlarge"
      instance_count            = 1
      ebs_volume_size_in_gb     = 500
      threads_per_core          = 2
      enable_stress_check       = false
      enable_connectivity_check = false
      lifecycle_script          = "on_create.sh"
      availability_zone_id      = "use1-az6"
    }
  ]
}

variable "auto_node_recovery" {
  description = "Enable automatic node recovery on the HyperPod cluster"
  type        = bool
  default     = true
}

variable "continuous_provisioning_mode" {
  description = "Enable continuous node provisioning mode on the HyperPod cluster"
  type        = bool
  default     = true
}

# --- Helm chart (HyperPod dependencies) ---
variable "helm_repo_path" {
  description = "Path to the HyperPod dependencies Helm chart inside the cloned helm repo (/tmp/helm-repo)"
  type        = string
  default     = "helm_chart/HyperPodHelmChart"
}

variable "helm_release_name" {
  description = "Name of the HyperPod dependencies Helm release"
  type        = string
  default     = "hyperpod-dependencies"
}

variable "helm_namespace" {
  description = "Namespace for the HyperPod dependencies Helm release"
  type        = string
  default     = "kube-system"
}

# Pinned to a known-good revision of aws/sagemaker-hyperpod-cli helm charts.
# See https://github.com/aws/sagemaker-hyperpod-cli/tree/main/helm_chart
variable "helm_repo_revision" {
  description = "Git revision of the sagemaker-hyperpod-cli helm charts to check out under /tmp/helm-repo"
  type        = string
  default     = "9f496a6364759553f73ff534434a057ef4bdc004"
}

variable "helm_repo_url" {
  description = "Git URL of the sagemaker-hyperpod-cli repo that provides the HyperPod dependencies Helm chart. Cloned automatically to helm_repo_local_path."
  type        = string
  default     = "https://github.com/aws/sagemaker-hyperpod-cli.git"
}

variable "helm_repo_local_path" {
  description = "Local filesystem path the helm repo is cloned to. The upstream helm_chart module hardcodes /tmp/helm-repo, so changing this requires a matching module change."
  type        = string
  default     = "/tmp/helm-repo"
}

# --- Inference operator dependencies (bundled with the EKS add-on) ---
variable "enable_s3_csi_driver" {
  description = "Install the Mountpoint for S3 CSI driver (required by the inference operator)"
  type        = bool
  default     = true
}

variable "enable_alb_controller" {
  description = "Install the AWS Load Balancer Controller (bundled with the inference operator add-on)"
  type        = bool
  default     = true
}

variable "enable_keda" {
  description = "Install KEDA (bundled with the inference operator add-on)"
  type        = bool
  default     = true
}

variable "enable_metrics_server" {
  description = "Install metrics-server EKS add-on"
  type        = bool
  default     = true
}

variable "enable_cert_manager" {
  description = "Install the cert-manager EKS add-on (required by the inference operator)"
  type        = bool
  default     = true
}

variable "inference_operator_create_timeout" {
  description = "Create timeout for the inference operator EKS add-on. Raise on slow P5 bring-up; if it still times out, re-running `tofu apply` resumes idempotently."
  type        = string
  default     = "40m"
}

variable "gated_access" {
  description = "Include gated-model access permissions on the SageMaker execution role"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Additional tags applied to taggable resources"
  type        = map(string)
  default     = {}
}
