packer {
  required_plugins {
    amazon = {
      version = "= 1.8.2"
      source  = "github.com/hashicorp/amazon"
    }
    ansible = {
      version = "= 1.1.6"
      source  = "github.com/hashicorp/ansible"
    }
  }
}

variable "ami_name" {
  type    = string
  default = "awsome-distributed-ai"
}
variable "ami_version" {
  type    = string
  default = "1"
}
variable "parallel_cluster_version" {
  type    = string
  default = "3.15.1"
}
variable "eks_version" {
  type    = string
  default = "1.35"
}
variable "aws_region" {
  type    = string
  default = "us-east-1"
}
variable "instance_type" {
  type    = string
  default = "g4dn.16xlarge"
}
variable "inventory_directory" {
  type    = string
  default = "inventory"
}

locals {
  timestamp         = regex_replace(timestamp(), "[- TZ:]", "")
  ubuntu_server_ssm = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
  dlami_ssm         = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
  eks_al2023_ssm    = "/aws/service/eks/optimized-ami/${var.eks_version}/amazon-linux-2023/x86_64/standard/recommended/image_id"
  eks_ubuntu_ssm    = "/aws/service/canonical/ubuntu/eks/24.04/${var.eks_version}/stable/current/amd64/hvm/ebs-gp3/ami-id"
}

data "amazon-parameterstore" "ubuntu_server" {
  name   = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
  region = var.aws_region
}
data "amazon-parameterstore" "dlami" {
  name   = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
  region = var.aws_region
}
data "amazon-parameterstore" "eks_al2023" {
  name   = "/aws/service/eks/optimized-ami/${var.eks_version}/amazon-linux-2023/x86_64/standard/recommended/image_id"
  region = var.aws_region
}
data "amazon-parameterstore" "eks_ubuntu" {
  name   = "/aws/service/canonical/ubuntu/eks/24.04/${var.eks_version}/stable/current/amd64/hvm/ebs-gp3/ami-id"
  region = var.aws_region
}

data "amazon-ami" "pcluster_ubuntu2404" {
  filters = {
    virtualization-type = "hvm"
    name                = "aws-parallelcluster-${var.parallel_cluster_version}-ubuntu-2404-lts-hvm-x86_64-*"
    architecture        = "x86_64"
    root-device-type    = "ebs"
    state               = "available"
  }
  most_recent = true
  owners      = ["amazon"]
  region      = var.aws_region
}

source "amazon-ebs" "ec2-ubuntu2404" {
  ami_name      = "${var.ami_name}-ec2-ubuntu2404-${var.ami_version}-${local.timestamp}"
  instance_type = var.instance_type
  region        = var.aws_region
  source_ami    = data.amazon-parameterstore.ubuntu_server.value
  ssh_username  = "ubuntu"
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }
  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 100
    throughput            = 1000
    iops                  = 10000
    volume_type           = "gp3"
    delete_on_termination = true
  }
  tags = { OS = "Ubuntu 24.04", ParentAMI = data.amazon-parameterstore.ubuntu_server.value, ParentLookup = local.ubuntu_server_ssm }
}
source "amazon-ebs" "ec2-ubuntu2404-dlami" {
  ami_name      = "${var.ami_name}-ec2-ubuntu2404-dlami-${var.ami_version}-${local.timestamp}"
  instance_type = var.instance_type
  region        = var.aws_region
  source_ami    = data.amazon-parameterstore.dlami.value
  ssh_username  = "ubuntu"
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }
  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 100
    throughput            = 1000
    iops                  = 10000
    volume_type           = "gp3"
    delete_on_termination = true
  }
  tags = { OS = "Ubuntu 24.04", ParentAMI = data.amazon-parameterstore.dlami.value, ParentLookup = local.dlami_ssm }
}
source "amazon-ebs" "pcluster-ubuntu2404" {
  ami_name      = "${var.ami_name}-pcluster-ubuntu2404-${var.ami_version}-${local.timestamp}"
  instance_type = var.instance_type
  region        = var.aws_region
  source_ami    = data.amazon-ami.pcluster_ubuntu2404.id
  ssh_username  = "ubuntu"
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }
  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 100
    throughput            = 1000
    iops                  = 10000
    volume_type           = "gp3"
    delete_on_termination = true
  }
  tags = { OS = "Ubuntu 24.04", ParentAMI = data.amazon-ami.pcluster_ubuntu2404.id, ParentLookup = "ParallelCluster ${var.parallel_cluster_version} Ubuntu 24.04 x86_64 official image filter", "parallelcluster:version" = var.parallel_cluster_version, "parallelcluster:build_status" = "available", "parallelcluster:os" = "ubuntu2404" }
}
source "amazon-ebs" "eks-al2023" {
  ami_name      = "${var.ami_name}-eks-al2023-${var.eks_version}-${var.ami_version}-${local.timestamp}"
  instance_type = var.instance_type
  region        = var.aws_region
  source_ami    = data.amazon-parameterstore.eks_al2023.value
  ssh_username  = "ec2-user"
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }
  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_size           = 100
    throughput            = 1000
    iops                  = 10000
    volume_type           = "gp3"
    delete_on_termination = true
  }
  tags = { OS = "AL2023", ParentAMI = data.amazon-parameterstore.eks_al2023.value, ParentLookup = local.eks_al2023_ssm }
}
source "amazon-ebs" "eks-ubuntu2404" {
  ami_name      = "${var.ami_name}-eks-ubuntu2404-${var.eks_version}-${var.ami_version}-${local.timestamp}"
  instance_type = var.instance_type
  region        = var.aws_region
  source_ami    = data.amazon-parameterstore.eks_ubuntu.value
  ssh_username  = "ubuntu"
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }
  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 100
    throughput            = 1000
    iops                  = 10000
    volume_type           = "gp3"
    delete_on_termination = true
  }
  tags = { OS = "Ubuntu 24.04", ParentAMI = data.amazon-parameterstore.eks_ubuntu.value, ParentLookup = local.eks_ubuntu_ssm, EFAInstaller = "1.50.0-minimal" }
}
source "amazon-ebs" "pcs-ubuntu2404" {
  ami_name      = "${var.ami_name}-pcs-ubuntu2404-${var.ami_version}-${local.timestamp}"
  instance_type = var.instance_type
  region        = var.aws_region
  source_ami    = data.amazon-parameterstore.dlami.value
  ssh_username  = "ubuntu"
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }
  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 100
    throughput            = 1000
    iops                  = 10000
    volume_type           = "gp3"
    delete_on_termination = true
  }
  tags = { OS = "Ubuntu 24.04", ParentAMI = data.amazon-parameterstore.dlami.value, ParentLookup = local.dlami_ssm, PCSAgent = "1.5.1-1", PCSAgentSHA256 = "9ac3fff38d75774852d947bf6823abc5a75e5c6325cd5aa5a3172b1defa72f9d", PCSSlurm = "25.11.7-3", PCSSlurmSHA256 = "9d2c7153583642d5714c4df3fb468c942b39ba995466b25a3b8dd6705e79aea9" }
}

build {
  name    = "ec2-ubuntu2404"
  sources = ["source.amazon-ebs.ec2-ubuntu2404"]
  provisioner "ansible" {
    user                = "ubuntu"
    playbook_file       = "playbook-ec2-ubuntu2404.yml"
    inventory_directory = var.inventory_directory
  }
}
build {
  name    = "ec2-ubuntu2404-dlami"
  sources = ["source.amazon-ebs.ec2-ubuntu2404-dlami"]
  provisioner "ansible" {
    user                = "ubuntu"
    playbook_file       = "playbook-ec2-ubuntu2404-dlami.yml"
    inventory_directory = var.inventory_directory
  }
}
build {
  name    = "pcluster-ubuntu2404"
  sources = ["source.amazon-ebs.pcluster-ubuntu2404"]
  provisioner "ansible" {
    user                = "ubuntu"
    playbook_file       = "playbook-pcluster-ubuntu2404.yml"
    inventory_directory = var.inventory_directory
  }
}
build {
  name    = "eks-al2023"
  sources = ["source.amazon-ebs.eks-al2023"]
  provisioner "ansible" {
    user                = "ec2-user"
    ansible_env_vars    = ["ANSIBLE_SCP_EXTRA_ARGS='-O'"]
    playbook_file       = "playbook-eks-al2023.yml"
    inventory_directory = var.inventory_directory
  }
}
build {
  name    = "eks-ubuntu2404"
  sources = ["source.amazon-ebs.eks-ubuntu2404"]
  provisioner "ansible" {
    user                = "ubuntu"
    playbook_file       = "playbook-eks-ubuntu2404.yml"
    inventory_directory = var.inventory_directory
  }
}
build {
  name    = "pcs-ubuntu2404"
  sources = ["source.amazon-ebs.pcs-ubuntu2404"]
  provisioner "ansible" {
    user                = "ubuntu"
    playbook_file       = "playbook-pcs-ubuntu2404.yml"
    inventory_directory = var.inventory_directory
    extra_arguments     = ["--extra-vars", "aws_region=${var.aws_region}"]
  }
}
