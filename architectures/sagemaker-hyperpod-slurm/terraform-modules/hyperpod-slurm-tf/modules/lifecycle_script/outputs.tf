output "lifecycle_scripts_s3_uri" {
  description = "S3 URI for lifecycle scripts"
  value       = "s3://${var.s3_bucket_name}/LifecycleScripts/base-config/"
}