resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "lifecycle_scripts" {
  bucket = "${var.resource_name_prefix}-lifecycle-${random_id.bucket_suffix.hex}"

  tags = {
    Name = "${var.resource_name_prefix}-lifecycle-scripts"
  }
}

resource "aws_s3_bucket_versioning" "lifecycle_scripts" {
  bucket = aws_s3_bucket.lifecycle_scripts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lifecycle_scripts" {
  bucket = aws_s3_bucket.lifecycle_scripts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "lifecycle_scripts" {
  bucket = aws_s3_bucket.lifecycle_scripts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}