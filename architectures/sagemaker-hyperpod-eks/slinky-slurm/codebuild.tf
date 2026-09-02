# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Terraform configuration for the Slurmd DLC CodeBuild pipeline.
# Provisions an ECR repository, IAM role, and CodeBuild project.
#
# Usage:
#   terraform init
#   terraform apply -var="source_s3_bucket=my-bucket"

variable "repository_name" {
  type        = string
  default     = "dlc-slurmd"
  description = "ECR repository name for the slurmd container image."
}

variable "image_tag" {
  type        = string
  default     = "25.11.1-ubuntu24.04"
  description = "Default image tag for the container build."
}

variable "source_s3_bucket" {
  type        = string
  description = "S3 bucket containing the build context (uploaded by setup.sh)."
}

variable "source_s3_key" {
  type        = string
  default     = "codebuild/slurmd-build-context.zip"
  description = "S3 key for the build context zip archive."
}

variable "create_ecr_repository" {
  type        = bool
  default     = true
  description = "Set to false to skip ECR repository creation when it already exists."
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  ecr_arn = var.create_ecr_repository ? aws_ecr_repository.slurmd[0].arn : "arn:aws:ecr:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:repository/${var.repository_name}"
  ecr_url = var.create_ecr_repository ? aws_ecr_repository.slurmd[0].repository_url : "${data.aws_caller_identity.current.account_id}.dkr.ecr.${data.aws_region.current.name}.amazonaws.com/${var.repository_name}"
}

###########################
## ECR Repository #########
###########################

resource "aws_ecr_repository" "slurmd" {
  count                = var.create_ecr_repository ? 1 : 0
  name                 = var.repository_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "slurmd" {
  count      = var.create_ecr_repository ? 1 : 0
  repository = aws_ecr_repository.slurmd[0].name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

###########################
## IAM Role ###############
###########################

resource "aws_iam_role" "codebuild" {
  name = "${var.repository_name}-codebuild-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "codebuild.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "codebuild_ecr" {
  name = "CodeBuildECRAccess"
  role = aws_iam_role.codebuild.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = local.ecr_arn
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "arn:aws:ecr:us-east-1:763104351884:repository/*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/codebuild/*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:GetObjectVersion"]
        Resource = "arn:aws:s3:::${var.source_s3_bucket}/${var.source_s3_key}"
      }
    ]
  })
}

###########################
## CodeBuild Project ######
###########################

resource "aws_codebuild_project" "slurmd" {
  name          = "${var.repository_name}-build"
  description   = "Builds the Slurmd Deep Learning Container image"
  service_role  = aws_iam_role.codebuild.arn
  build_timeout = 60

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_LARGE"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true

    environment_variable {
      name  = "AWS_ACCOUNT_ID"
      value = data.aws_caller_identity.current.account_id
    }

    environment_variable {
      name  = "AWS_DEFAULT_REGION"
      value = data.aws_region.current.name
    }

    environment_variable {
      name  = "REPO_NAME"
      value = var.repository_name
    }

    environment_variable {
      name  = "IMAGE_TAG"
      value = var.image_tag
    }
  }

  source {
    type     = "S3"
    location = "${var.source_s3_bucket}/${var.source_s3_key}"
  }
}

###########################
## Outputs ################
###########################

output "ecr_repository_uri" {
  description = "ECR repository URI for the slurmd container image."
  value       = local.ecr_url
}

output "codebuild_project_name" {
  description = "CodeBuild project name for triggering builds."
  value       = aws_codebuild_project.slurmd.name
}

output "codebuild_role_arn" {
  description = "IAM role ARN used by the CodeBuild project."
  value       = aws_iam_role.codebuild.arn
}
