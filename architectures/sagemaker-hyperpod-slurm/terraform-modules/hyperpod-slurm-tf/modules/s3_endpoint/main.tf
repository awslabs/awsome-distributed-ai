data "aws_region" "current" {}

resource "aws_vpc_endpoint" "s3" {
  vpc_id       = var.vpc_id
  service_name = "com.amazonaws.${data.aws_region.current.id}.s3"

  route_table_ids = compact([
    var.private_route_table_id,
    var.public_route_table_id
  ])

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = "*"
        Action    = "*"
        Resource  = "*"
      }
    ]
  })

  tags = {
    Name = "s3-endpoint"
  }
}