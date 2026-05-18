variable "resource_name_prefix" {
  description = "Prefix to be used for all resources"
  type        = string
  default     = "sagemaker-hyperpod-slurm"
}

variable "aws_region" {
  description = "AWS Region to be targeted for deployment"
  type        = string
  default     = "us-west-2"
}

# VPC Module Variables
variable "create_vpc_module" {
  description = "Whether to create VPC module"
  type        = bool
  default     = true
}

variable "vpc_cidr" {
  description = "The IP range (CIDR notation) for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
  description = "The IP range (CIDR notation) for the public subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "existing_vpc_id" {
  description = "The ID of an existing VPC to use if not creating a new one"
  type        = string
  default     = ""
}

# Private Subnet Module Variables
variable "create_private_subnet_module" {
  description = "Whether to create private subnet module"
  type        = bool
  default     = true
}

variable "availability_zone_id" {
  description = "The Availability Zone Id for subnets"
  type        = string
  default     = "usw2-az4"
}

variable "private_subnet_cidr" {
  description = "The IP range (CIDR notation) for the private subnet"
  type        = string
  default     = "10.1.0.0/16"
}

variable "existing_nat_gateway_id" {
  description = "The ID of an existing NAT Gateway"
  type        = string
  default     = ""
}

variable "existing_private_subnet_id" {
  description = "The ID of an existing private subnet"
  type        = string
  default     = ""
}

variable "existing_private_route_table_id" {
  description = "The ID of an existing private route table"
  type        = string
  default     = ""
}

variable "existing_public_route_table_id" {
  description = "The ID of an existing public route table"
  type        = string
  default     = ""
}

# Security Group Module Variables
variable "create_security_group_module" {
  description = "Whether to create security group module"
  type        = bool
  default     = true
}

variable "existing_security_group_id" {
  description = "The ID of an existing security group"
  type        = string
  default     = ""
}

# S3 Bucket Module Variables
variable "create_s3_bucket_module" {
  description = "Whether to create S3 bucket module"
  type        = bool
  default     = true
}

variable "existing_s3_bucket_name" {
  description = "The name of an existing S3 bucket"
  type        = string
  default     = ""
}

# S3 Endpoint Module Variables
variable "create_s3_endpoint_module" {
  description = "Whether to create S3 endpoint module"
  type        = bool
  default     = true
}

# FSx Lustre Module Variables
variable "create_fsx_lustre_module" {
  description = "Whether to create FSx Lustre module"
  type        = bool
  default     = true
}

variable "fsx_lustre_storage_capacity" {
  description = "Storage capacity in GiB (1200 or increments of 2400)"
  type        = number
  default     = 1200
}

variable "fsx_lustre_throughput_per_unit" {
  description = "Provisioned Read/Write (MB/s/TiB)"
  type        = number
  default     = 250
  validation {
    condition     = contains([125, 250, 500, 1000], var.fsx_lustre_throughput_per_unit)
    error_message = "Throughput per unit must be one of: 125, 250, 500, 1000."
  }
}

variable "fsx_lustre_compression_type" {
  description = "Data compression type"
  type        = string
  default     = "LZ4"
  validation {
    condition     = contains(["LZ4", "NONE"], var.fsx_lustre_compression_type)
    error_message = "Compression type must be either LZ4 or NONE."
  }
}

variable "fsx_lustre_version" {
  description = "Lustre software version"
  type        = string
  default     = "2.15"
  validation {
    condition     = contains(["2.15", "2.12"], var.fsx_lustre_version)
    error_message = "Lustre version must be either 2.15 or 2.12."
  }
}

variable "existing_fsx_lustre_dns_name" {
  description = "DNS name of existing FSx Lustre file system"
  type        = string
  default     = ""
}

variable "existing_fsx_lustre_mount_name" {
  description = "Mount name of existing FSx Lustre file system"
  type        = string
  default     = ""
}

# FSx OpenZFS Module Variables
variable "create_fsx_openzfs_module" {
  description = "Whether to create FSx OpenZFS module"
  type        = bool
  default     = false
}

variable "fsx_openzfs_storage_capacity" {
  description = "Storage capacity in GiB (64 or increments of 64)"
  type        = number
  default     = 64
}

variable "fsx_openzfs_throughput_capacity" {
  description = "Throughput capacity in MBps (Multi-AZ requires minimum 160 MBps)"
  type        = number
  default     = 64
  validation {
    condition     = contains([64, 128, 160, 256, 512, 1024, 2048, 3072, 4096], var.fsx_openzfs_throughput_capacity)
    error_message = "Throughput capacity must be one of: 64, 128, 160, 256, 512, 1024, 2048, 3072, 4096. Multi-AZ deployments require minimum 160 MBps."
  }
}

variable "fsx_openzfs_compression_type" {
  description = "Data compression type"
  type        = string
  default     = "ZSTD"
  validation {
    condition     = contains(["ZSTD", "LZ4", "NONE"], var.fsx_openzfs_compression_type)
    error_message = "Compression type must be one of: ZSTD, LZ4, NONE."
  }
}

variable "fsx_openzfs_deployment_type" {
  description = "FSx OpenZFS deployment type"
  type        = string
  default     = "SINGLE_AZ_1"
  validation {
    condition     = contains(["SINGLE_AZ_1", "SINGLE_AZ_HA_1", "MULTI_AZ_1"], var.fsx_openzfs_deployment_type)
    error_message = "Deployment type must be one of: SINGLE_AZ_1, SINGLE_AZ_HA_1, MULTI_AZ_1."
  }
}

variable "fsx_openzfs_subnet_ids" {
  description = "List of subnet IDs for Multi-AZ FSx OpenZFS deployment (must be in different AZs)"
  type        = list(string)
  default     = []
}

variable "existing_fsx_openzfs_dns_name" {
  description = "DNS name of existing FSx OpenZFS file system"
  type        = string
  default     = ""
}

# Lifecycle Script Module Variables
variable "create_lifecycle_script_module" {
  description = "Whether to create lifecycle script module"
  type        = bool
  default     = true
}

variable "lifecycle_scripts_path" {
  description = "Path to lifecycle scripts directory"
  type        = string
  default     = "../../LifecycleScripts/base-config"
}

# SageMaker IAM Role Module Variables
variable "create_sagemaker_iam_role_module" {
  description = "Whether to create SageMaker IAM role module"
  type        = bool
  default     = true
}

variable "existing_sagemaker_iam_role_name" {
  description = "The name of an existing SageMaker IAM role"
  type        = string
  default     = ""
}

# HyperPod Cluster Module Variables
variable "create_hyperpod_module" {
  description = "Whether to create HyperPod cluster module"
  type        = bool
  default     = true
}

variable "hyperpod_cluster_name" {
  description = "Name of the HyperPod cluster"
  type        = string
  default     = "ml-cluster"
}

variable "node_recovery" {
  description = "Node recovery mode"
  type        = string
  default     = "Automatic"
  validation {
    condition     = contains(["Automatic", "None"], var.node_recovery)
    error_message = "Node recovery must be either Automatic or None."
  }
}

variable "instance_groups" {
  description = "List of instance group configurations. Each item includes a name. Optionally include training_plan_arn per group."
  type = list(object({
    name              = string
    instance_type     = string
    instance_count    = number
    ebs_volume_size   = number
    threads_per_core  = number
    lifecycle_script  = string
    training_plan_arn = optional(string)
  }))

  default = [
    {
      name             = "controller-machine"
      instance_type    = "ml.c5.xlarge"
      instance_count   = 1
      ebs_volume_size  = 100
      threads_per_core = 1
      lifecycle_script = "on_create.sh"
    },
    {
      name             = "login-nodes"
      instance_type    = "ml.c5.large"
      instance_count   = 1
      ebs_volume_size  = 100
      threads_per_core = 1
      lifecycle_script = "on_create.sh"
    },
    {
      name             = "compute-nodes"
      instance_type    = "ml.trn1.32xlarge"
      instance_count   = 4
      ebs_volume_size  = 500
      threads_per_core = 1
      lifecycle_script = "on_create.sh"
      # training_plan_arn = "arn:aws:sagemaker:REGION:ACCOUNT:training-plan/YOUR_TP" # optional
    }
  ]
}
