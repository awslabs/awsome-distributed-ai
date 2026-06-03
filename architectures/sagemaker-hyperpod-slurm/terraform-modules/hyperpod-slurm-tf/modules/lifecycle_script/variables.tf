variable "resource_name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for lifecycle scripts"
  type        = string
}

variable "fsx_lustre_dns_name" {
  description = "DNS name of the FSx Lustre file system"
  type        = string
}

variable "fsx_lustre_mount_name" {
  description = "Mount name of the FSx Lustre file system"
  type        = string
}

variable "fsx_openzfs_dns_name" {
  description = "DNS name of the FSx OpenZFS file system"
  type        = string
}

variable "lifecycle_scripts_path" {
  description = "Path to lifecycle scripts directory"
  type        = string
}

variable "instance_groups" {
  description = "List of instance group configurations. Each item includes a name."
  type = list(object({
    name              = string
    instance_type     = string
    instance_count    = number
    ebs_volume_size   = number
    threads_per_core  = number
    lifecycle_script  = string
    training_plan_arn = optional(string)
  }))
}
