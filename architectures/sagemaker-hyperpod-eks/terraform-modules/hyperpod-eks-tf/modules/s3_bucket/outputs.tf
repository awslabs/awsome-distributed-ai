output "s3_bucket_name" {
  description = "S3 Bucket Name"
  value       = aws_s3_bucket.bucket.id
}

output "s3_bucket_arn" {
  description = "S3 Bucket ARN"
  value       = aws_s3_bucket.bucket.arn
}

output "s3_logs_bucket_name" {
  description = "S3 Access Logs Bucket Name"
  value       = aws_s3_bucket.access_logs_bucket.id
}

output "s3_logs_bucket_arn" {
  description = "S3 Access Logs Bucket ARN"
  value       = aws_s3_bucket.access_logs_bucket.arn
}