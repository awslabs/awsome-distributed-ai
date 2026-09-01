locals {
  # Generate provisioning parameters JSON
  provisioning_parameters = merge(
    {
      version          = "1.0.0"
      workload_manager = "slurm"
      controller_group = "controller-machine"
      login_group      = "login-nodes"
      worker_groups = [
        for ig in var.instance_groups : {
          instance_group_name = ig.name
          partition_name      = ig.name == "controller-machine" ? null : "dev"
        } if ig.name != "controller-machine"
      ]
      fsx_dns_name  = var.fsx_lustre_dns_name
      fsx_mountname = var.fsx_lustre_mount_name
    },
    var.fsx_openzfs_dns_name != "" ? { fsx_openzfs_dns_name = var.fsx_openzfs_dns_name } : {}
  )
}

# Upload lifecycle scripts to S3
resource "aws_s3_object" "lifecycle_scripts" {
  for_each = fileset(var.lifecycle_scripts_path, "**/*")

  bucket = var.s3_bucket_name
  key    = "LifecycleScripts/base-config/${each.value}"
  source = "${var.lifecycle_scripts_path}/${each.value}"
  etag   = filemd5("${var.lifecycle_scripts_path}/${each.value}")

  tags = {
    Name = "${var.resource_name_prefix}-lifecycle-script-${each.value}"
  }
}

# Upload the generated provisioning parameters
resource "aws_s3_object" "provisioning_parameters" {
  bucket  = var.s3_bucket_name
  key     = "LifecycleScripts/base-config/provisioning_parameters.json"
  content = jsonencode(local.provisioning_parameters)

  tags = {
    Name = "${var.resource_name_prefix}-provisioning-parameters"
  }
}
