data "http" "scripts" {
  for_each = toset(var.script_urls)
  url      = each.value
}

resource "aws_s3_object" "scripts" {
  for_each     = data.http.scripts
  bucket       = var.s3_bucket_name
  key          = basename(each.value.url)
  content      = each.value.response_body
  content_type = "text/x-sh"
}
