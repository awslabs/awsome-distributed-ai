# Resolve the GPU Deep Learning AMI from SSM Parameter Store (always latest).
data "aws_ssm_parameter" "dlami" {
  name = var.dlami_ssm_parameter
}

# Use the default VPC + a default subnet for a self-contained, no-frills test box.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

# ---------------------------------------------------------------------------
# IAM role + instance profile for SSM Session Manager (no inbound SSH needed).
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "ssm_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ssm" {
  name               = "${var.name_prefix}-ssm"
  assume_role_policy = data.aws_iam_policy_document.ssm_assume.json
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.ssm.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ssm" {
  name = "${var.name_prefix}-ssm"
  role = aws_iam_role.ssm.name
}

# ---------------------------------------------------------------------------
# Security group: egress only. SSM Session Manager needs no inbound rules.
# ---------------------------------------------------------------------------
resource "aws_security_group" "this" {
  name        = "${var.name_prefix}-sg"
  description = "ModelOpt runbook - egress only for SSM"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "Allow all outbound (SSM endpoints, package + model downloads)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# The single GPU instance.
# ---------------------------------------------------------------------------
resource "aws_instance" "this" {
  ami                    = data.aws_ssm_parameter.dlami.value
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.ssm.name
  vpc_security_group_ids = [aws_security_group.this.id]
  subnet_id              = data.aws_subnets.default.ids[0]

  root_block_device {
    volume_size           = var.root_volume_size_gb
    volume_type           = "gp3"
    delete_on_termination = true
  }

  # Enforce IMDSv2.
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  tags = {
    Name = var.name_prefix
  }
}
