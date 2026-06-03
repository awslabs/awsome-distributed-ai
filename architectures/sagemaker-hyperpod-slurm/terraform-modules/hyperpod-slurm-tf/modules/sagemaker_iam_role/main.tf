data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution_role" {
  name               = "${var.resource_name_prefix}-sagemaker-execution-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json

  tags = {
    Name = "${var.resource_name_prefix}-sagemaker-execution-role"
  }
}

resource "aws_iam_role_policy_attachment" "sagemaker_cluster_instance_role_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerClusterInstanceRolePolicy"
}

data "aws_iam_policy_document" "sagemaker_vpc_policy" {
  statement {
    effect = "Allow"
    actions = [
      "ec2:CreateNetworkInterface",
      "ec2:CreateNetworkInterfacePermission",
      "ec2:DeleteNetworkInterface",
      "ec2:DeleteNetworkInterfacePermission",
      "ec2:DescribeNetworkInterfaces",
      "ec2:DescribeVpcs",
      "ec2:DescribeDhcpOptions",
      "ec2:DescribeSubnets",
      "ec2:DescribeSecurityGroups",
      "ec2:DetachNetworkInterface",
      "ec2:CreateTags"
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "sagemaker_vpc_policy" {
  name   = "${var.resource_name_prefix}-sagemaker-vpc-policy"
  role   = aws_iam_role.sagemaker_execution_role.id
  policy = data.aws_iam_policy_document.sagemaker_vpc_policy.json
}

data "aws_iam_policy_document" "s3_access_policy" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket"
    ]
    resources = [
      "arn:aws:s3:::${var.s3_bucket_name}",
      "arn:aws:s3:::${var.s3_bucket_name}/*"
    ]
  }
}

resource "aws_iam_role_policy" "s3_access_policy" {
  name   = "${var.resource_name_prefix}-s3-access-policy"
  role   = aws_iam_role.sagemaker_execution_role.id
  policy = data.aws_iam_policy_document.s3_access_policy.json
}