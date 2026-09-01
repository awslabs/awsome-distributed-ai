locals {
  subnet_ids = var.deployment_type == "MULTI_AZ_1" ? var.private_subnet_ids : [var.private_subnet_id]
}

# Validation for Multi-AZ requirements
check "multi_az_subnets" {
  assert {
    condition     = var.deployment_type != "MULTI_AZ_1" || length(var.private_subnet_ids) >= 2
    error_message = "Multi-AZ deployment requires at least 2 subnet IDs in different AZs."
  }
}

check "multi_az_throughput" {
  assert {
    condition     = var.deployment_type != "MULTI_AZ_1" || var.throughput_capacity >= 160
    error_message = "Multi-AZ deployment requires minimum 160 MBps throughput capacity."
  }
}

resource "aws_fsx_openzfs_file_system" "main" {
  storage_capacity    = var.storage_capacity
  subnet_ids          = local.subnet_ids
  security_group_ids  = [var.security_group_id]
  deployment_type     = var.deployment_type
  throughput_capacity = var.throughput_capacity

  root_volume_configuration {
    data_compression_type = var.compression_type
    nfs_exports {
      client_configurations {
        clients = "*"
        options = ["rw", "no_root_squash", "crossmnt"]
      }
    }
  }

  tags = {
    Name = "${var.resource_name_prefix}-fsx-openzfs"
  }
}
