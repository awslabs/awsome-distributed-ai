packer {
  required_plugins {
    amazon = {
      version = ">= 1.0.9"
      source = "github.com/hashicorp/amazon"
    }
    ansible = {
      version = ">= 1.0.1"
      source = "github.com/hashicorp/ansible"
    }
  }
}

variable "ami_name" {
  type    = string
  default = "pcluster-efa"
}

variable "ami_version" {
  type    = string
  default = "1.0.0"
}

variable "parallel_cluster_version" {
  type    = string
  default = "3.7.0"
}

variable "eks_version" {
  type    = string
  default = "1.24"
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

variable "playbook_file" {
  type    = string
  default = "packer-playbook.yml"
}

variable "ssh_username" {
  type    = string
  default = "ec2-user"
}

# "timestamp" template function replacement
locals { timestamp = regex_replace(timestamp(), "[- TZ:]", "") }

data "amazon-ami" "pcluster-al2" {
  filters = {
    virtualization-type = "hvm"
    name = "aws-parallelcluster-${var.parallel_cluster_version}-amzn2-*"
    architecture= "x86_64"
    root-device-type = "ebs"
  }
  most_recent = true
  owners      = ["amazon"]
}

data "amazon-ami" "base-al2" {
  filters = {
    virtualization-type = "hvm"
    name = "amzn2-ami-kernel-5.10-hvm-*"
    architecture= "x86_64"
    root-device-type = "ebs"
  }
  most_recent = true
  owners      = ["amazon"]
}

data "amazon-ami" "eks-al2" {
  filters = {
    virtualization-type = "hvm"
    name = "amazon-eks-node-${var.eks_version}-v*"
    architecture= "x86_64"
    root-device-type = "ebs"
  }
  most_recent = true
  owners      = ["amazon"]
}

data "amazon-ami" "dlami-gpu-al2" {
  filters = {
    virtualization-type = "hvm"
    name = "Deep Learning AMI GPU PyTorch 2.0.1 (Amazon Linux 2) *"
    architecture= "x86_64"
    root-device-type = "ebs"
  }
  most_recent = true
  owners      = ["amazon"]
}

data "amazon-ami" "dlami-neuron-al2" {
  filters = {
    virtualization-type = "hvm"
    name = "Deep Learning AMI Neuron PyTorch 1.13 (Amazon Linux 2) *"
    architecture= "x86_64"
    root-device-type = "ebs"
  }
  most_recent = true
  owners      = ["amazon"]
}

source "amazon-ebs" "aws-pcluster-ami" {
  ami_name      = "${var.ami_name}-pcluster-${var.ami_version}-${local.timestamp}"
  instance_type = "${var.instance_type}"
  region        = "${var.aws_region}"
  source_ami     = data.amazon-ami.pcluster-al2.id
  ssh_username  = "ec2-user"
  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_size           = 100 
    throughput            = 1000
    iops                  = 10000
    volume_type           = "gp3"
    delete_on_termination = true
  }
  tags = {
    "OS" =  "Amazon Linux 2",
    "parallelcluster:version" =  "${var.parallel_cluster_version}"
    "parallelcluster:build_status" = "available"
    "parallelcluster:os" = "alinux2"
  }
  run_tags = {
    "Name" = "packer-builder-pcluster-${var.parallel_cluster_version}"
  }
}

source "amazon-ebs" "aws-base-ami" {
  ami_name      = "${var.ami_name}-base-${var.ami_version}-${local.timestamp}"
  instance_type = "${var.instance_type}"
  region        = "${var.aws_region}"
  source_ami     = data.amazon-ami.base-al2.id
  ssh_username  = "ec2-user"
  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_size           = 100
    throughput            = 1000
    iops                  = 10000
    volume_type           = "gp3"
    delete_on_termination = true
  }
  run_tags = {
    "Name" = "packer-builder-base-al2"
  }
}

source "amazon-ebs" "aws-eks-ami" {
  ami_name      = "${var.ami_name}-eks-${var.eks_version}-${var.ami_version}-${local.timestamp}"
  instance_type = "${var.instance_type}"
  region        = "${var.aws_region}"
  source_ami     = data.amazon-ami.eks-al2.id
  ssh_username  = "ec2-user"
  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_size           = 100
    throughput            = 1000
    iops                  = 10000
    volume_type           = "gp3"
    delete_on_termination = true
  }
  run_tags = {
    "Name" = "packer-builder-eks-al2-${var.eks_version}"
  }
}

source "amazon-ebs" "aws-dlami-gpu-ami" {
  ami_name      = "${var.ami_name}-dlami-${var.ami_version}-${local.timestamp}"
  instance_type = "${var.instance_type}"
  region        = "${var.aws_region}"
  source_ami     = data.amazon-ami.dlami-gpu-al2.id
  ssh_username  = "ec2-user"
  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_size           = 100
    throughput            = 1000
    iops                  = 10000
    volume_type           = "gp3"
    delete_on_termination = true
  }
  run_tags = {
    "Name" = "packer-builder-dlami-gpu-al2"
  }
}

source "amazon-ebs" "aws-dlami-neuron-ami" {
  ami_name      = "${var.ami_name}-dlami-${var.ami_version}-${local.timestamp}"
  instance_type = "${var.instance_type}"
  region        = "${var.aws_region}"
  source_ami     = data.amazon-ami.dlami-neuron-al2.id
  ssh_username  = "ec2-user"
  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_size           = 100
    throughput            = 1000
    iops                  = 10000
    volume_type           = "gp3"
    delete_on_termination = true
  }
  run_tags = {
    "Name" = "packer-builder-dlami-neuron-al2"
  }
} 

build {
  name    = "aws-base-gpu"
  sources = ["source.amazon-ebs.aws-base-ami"]

  provisioner "ansible" {
    user                = "ec2-user"
    ansible_env_vars    = ["ANSIBLE_SCP_EXTRA_ARGS='-O'"]
    playbook_file       = "playbook-pcluster-gpu.yml"
    inventory_directory = "${var.inventory_directory}"
  }
}

build {
  name    = "aws-pcluster-cpu"
  sources = ["source.amazon-ebs.aws-pcluster-ami"]

  provisioner "ansible" {
    user                = "ec2-user"
    ansible_env_vars    = ["ANSIBLE_SCP_EXTRA_ARGS='-O'"]
    playbook_file       = "playbook-pcluster-cpu.yml"
    inventory_directory = "${var.inventory_directory}"
  }
}

build {
  name    = "aws-pcluster-gpu"
  sources = ["source.amazon-ebs.aws-pcluster-ami"]

  provisioner "ansible" {
    user                = "ec2-user"
    ansible_env_vars    = ["ANSIBLE_SCP_EXTRA_ARGS='-O'"]
    playbook_file       = "playbook-pcluster-gpu.yml"
    inventory_directory = "${var.inventory_directory}"
  }
}

build {
  name    = "aws-pcluster-neuron"
  sources = ["source.amazon-ebs.aws-pcluster-ami"]

  provisioner "ansible" {
    user                = "ec2-user"
    ansible_env_vars    = ["ANSIBLE_SCP_EXTRA_ARGS='-O'"]
    playbook_file       = "playbook-pcluster-neuron.yml"
    inventory_directory = "${var.inventory_directory}"
  }
}

build {
  name    = "aws-eks-gpu"
  sources = ["source.amazon-ebs.aws-eks-ami"]

  provisioner "ansible" {
    user                = "ec2-user"
    ansible_env_vars    = ["ANSIBLE_SCP_EXTRA_ARGS='-O'"]
    playbook_file       = "playbook-eks-gpu.yml"
    inventory_directory = "${var.inventory_directory}"
  }
}

build {
  name    = "aws-dlami-gpu"
  sources = ["source.amazon-ebs.aws-dlami-gpu-ami"]

  provisioner "ansible" {
    user                = "ec2-user"
    # ansible_env_vars    = ["ANSIBLE_SCP_EXTRA_ARGS='-O'"]
    playbook_file       = "playbook-dlami-gpu.yml"
    inventory_directory = "${var.inventory_directory}"
  }
}

build {
  name    = "aws-dlami-neuron"
  sources = ["source.amazon-ebs.aws-dlami-neuron-ami"]

  provisioner "ansible" {
    user                = "ec2-user"
    ansible_env_vars    = ["ANSIBLE_SCP_EXTRA_ARGS='-O'"]
    playbook_file       = "playbook-dlami-neuron.yml"
    inventory_directory = "${var.inventory_directory}"
  }
}
