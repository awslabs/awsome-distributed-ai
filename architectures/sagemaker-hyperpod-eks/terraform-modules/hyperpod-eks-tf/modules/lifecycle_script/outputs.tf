output "s3_object_keys" {
  description = "Map of uploaded script keys in S3"
  value       = { for k, v in aws_s3_object.scripts : k => v.key }
}

output "s3_object_version_ids" {
  description = "Map of version IDs of the uploaded scripts (if bucket versioning is enabled)"
  value       = { for k, v in aws_s3_object.scripts : k => v.version_id }
}
