data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

# Access logs bucket
resource "aws_s3_bucket" "access_logs_bucket" {
  bucket = "${var.resource_name_prefix}-logs-${data.aws_caller_identity.current.account_id}-${data.aws_region.current.region}"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "access_logs_encryption" {
  bucket = aws_s3_bucket.access_logs_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "access_logs_pab" {
  bucket = aws_s3_bucket.access_logs_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# LCS bucket
resource "aws_s3_bucket" "bucket" {
  bucket = "${var.resource_name_prefix}-bucket-${data.aws_caller_identity.current.account_id}-${data.aws_region.current.region}"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "bucket_encryption" {
  bucket = aws_s3_bucket.bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# write access logs for CLS bucket to Access logs bucket 
resource "aws_s3_bucket_logging" "bucket_logging" {
  bucket = aws_s3_bucket.bucket.id

  target_bucket = aws_s3_bucket.access_logs_bucket.id
  target_prefix = "bucket-logs/"
}