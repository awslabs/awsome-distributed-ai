data "aws_s3_bucket" "existing_s3_bucket" {
  count  = var.create_s3_bucket_module ? 0 : (var.existing_s3_bucket_name != "" ? 1 : 0)
  bucket = var.existing_s3_bucket_name
}

locals {
  vpc_id                  = var.create_vpc_module ? module.vpc[0].vpc_id : var.existing_vpc_id
  private_subnet_id       = var.create_private_subnet_module ? module.private_subnet[0].private_subnet_id : var.existing_private_subnet_id
  security_group_id       = var.create_security_group_module ? module.security_group[0].security_group_id : var.existing_security_group_id
  s3_bucket_name          = var.create_s3_bucket_module ? module.s3_bucket[0].s3_bucket_name : var.existing_s3_bucket_name
  sagemaker_iam_role_name = var.create_sagemaker_iam_role_module ? module.sagemaker_iam_role[0].sagemaker_iam_role_name : var.existing_sagemaker_iam_role_name
  fsx_lustre_dns_name     = var.create_fsx_lustre_module ? module.fsx_lustre[0].fsx_lustre_dns_name : var.existing_fsx_lustre_dns_name
  fsx_lustre_mount_name   = var.create_fsx_lustre_module ? module.fsx_lustre[0].fsx_lustre_mount_name : var.existing_fsx_lustre_mount_name
  fsx_openzfs_dns_name    = var.create_fsx_openzfs_module ? module.fsx_openzfs[0].fsx_openzfs_dns_name : var.existing_fsx_openzfs_dns_name
  # For Multi-AZ, use provided subnet IDs; for Single-AZ, use single private subnet
  fsx_openzfs_subnet_ids = var.fsx_openzfs_deployment_type == "MULTI_AZ_1" ? var.fsx_openzfs_subnet_ids : []
}

module "vpc" {
  count  = var.create_vpc_module ? 1 : 0
  source = "./modules/vpc"

  resource_name_prefix = var.resource_name_prefix
  vpc_cidr             = var.vpc_cidr
  public_subnet_cidr   = var.public_subnet_cidr
  availability_zone_id = var.availability_zone_id
}

module "private_subnet" {
  count  = var.create_private_subnet_module ? 1 : 0
  source = "./modules/private_subnet"

  resource_name_prefix = var.resource_name_prefix
  vpc_id               = local.vpc_id
  availability_zone_id = var.availability_zone_id
  private_subnet_cidr  = var.private_subnet_cidr
  nat_gateway_id       = var.create_vpc_module ? module.vpc[0].nat_gateway_id : var.existing_nat_gateway_id
}

module "security_group" {
  count  = var.create_security_group_module ? 1 : 0
  source = "./modules/security_group"

  resource_name_prefix = var.resource_name_prefix
  vpc_id               = local.vpc_id
  vpc_cidr             = var.vpc_cidr
}

module "s3_bucket" {
  count  = var.create_s3_bucket_module ? 1 : 0
  source = "./modules/s3_bucket"

  resource_name_prefix = var.resource_name_prefix
}

module "s3_endpoint" {
  count  = var.create_s3_endpoint_module ? 1 : 0
  source = "./modules/s3_endpoint"

  vpc_id                 = local.vpc_id
  private_route_table_id = var.create_private_subnet_module ? module.private_subnet[0].private_route_table_id : var.existing_private_route_table_id
  public_route_table_id  = var.create_vpc_module ? module.vpc[0].public_route_table_id : var.existing_public_route_table_id
}

module "fsx_lustre" {
  count  = var.create_fsx_lustre_module ? 1 : 0
  source = "./modules/fsx_lustre"

  resource_name_prefix = var.resource_name_prefix
  private_subnet_id    = local.private_subnet_id
  security_group_id    = local.security_group_id
  storage_capacity     = var.fsx_lustre_storage_capacity
  throughput_per_unit  = var.fsx_lustre_throughput_per_unit
  compression_type     = var.fsx_lustre_compression_type
  lustre_version       = var.fsx_lustre_version
}

module "fsx_openzfs" {
  count  = var.create_fsx_openzfs_module ? 1 : 0
  source = "./modules/fsx_openzfs"

  resource_name_prefix = var.resource_name_prefix
  private_subnet_id    = local.private_subnet_id
  private_subnet_ids   = local.fsx_openzfs_subnet_ids
  security_group_id    = local.security_group_id
  storage_capacity     = var.fsx_openzfs_storage_capacity
  throughput_capacity  = var.fsx_openzfs_throughput_capacity
  compression_type     = var.fsx_openzfs_compression_type
  deployment_type      = var.fsx_openzfs_deployment_type
}

module "lifecycle_script" {
  count  = var.create_lifecycle_script_module ? 1 : 0
  source = "./modules/lifecycle_script"

  resource_name_prefix   = var.resource_name_prefix
  s3_bucket_name         = local.s3_bucket_name
  fsx_lustre_dns_name    = local.fsx_lustre_dns_name
  fsx_lustre_mount_name  = local.fsx_lustre_mount_name
  fsx_openzfs_dns_name   = local.fsx_openzfs_dns_name
  lifecycle_scripts_path = var.lifecycle_scripts_path
  instance_groups        = var.instance_groups
}

module "sagemaker_iam_role" {
  count  = var.create_sagemaker_iam_role_module ? 1 : 0
  source = "./modules/sagemaker_iam_role"

  resource_name_prefix = var.resource_name_prefix
  s3_bucket_name       = local.s3_bucket_name
}

module "hyperpod_cluster" {
  count  = var.create_hyperpod_module ? 1 : 0
  source = "./modules/hyperpod_cluster"

  depends_on = [
    module.vpc,
    module.private_subnet,
    module.security_group,
    module.s3_bucket,
    module.s3_endpoint,
    module.fsx_lustre,
    module.fsx_openzfs,
    module.lifecycle_script,
    module.sagemaker_iam_role
  ]

  resource_name_prefix    = var.resource_name_prefix
  hyperpod_cluster_name   = var.hyperpod_cluster_name
  node_recovery           = var.node_recovery
  instance_groups         = var.instance_groups
  private_subnet_id       = local.private_subnet_id
  security_group_id       = local.security_group_id
  s3_bucket_name          = local.s3_bucket_name
  sagemaker_iam_role_name = local.sagemaker_iam_role_name
}
