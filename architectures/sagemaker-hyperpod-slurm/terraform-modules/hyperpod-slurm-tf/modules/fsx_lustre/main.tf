resource "aws_fsx_lustre_file_system" "main" {
  storage_capacity            = var.storage_capacity
  subnet_ids                  = [var.private_subnet_id]
  security_group_ids          = [var.security_group_id]
  file_system_type_version    = var.lustre_version
  storage_type                = "SSD"
  deployment_type             = "PERSISTENT_2"
  per_unit_storage_throughput = var.throughput_per_unit
  data_compression_type       = var.compression_type

  metadata_configuration {
    mode = "AUTOMATIC"
  }

  tags = {
    Name = "${var.resource_name_prefix}-fsx-lustre"
  }
}