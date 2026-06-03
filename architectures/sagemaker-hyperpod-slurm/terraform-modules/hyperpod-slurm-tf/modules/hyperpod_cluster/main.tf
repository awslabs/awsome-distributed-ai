data "aws_partition" "current" {}
data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

# Add delay to allow IAM role propagation
resource "time_sleep" "wait_for_iam_role" {
  create_duration = "30s"
}

locals {
  # Create configurations for each instance group
  # - instance_group_name comes from ig.name
  # - training_plan_arn is optional per instance group (only included when set)
  instance_groups_list = [
    for ig in var.instance_groups : merge(
      {
        instance_group_name = ig.name
        instance_type       = ig.instance_type
        instance_count      = ig.instance_count
        threads_per_core    = ig.threads_per_core
        execution_role      = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:role/${var.sagemaker_iam_role_name}"

        instance_storage_configs = [
          {
            ebs_volume_config = {
              volume_size_in_gb = ig.ebs_volume_size
            }
          }
        ]

        life_cycle_config = {
          on_create     = ig.lifecycle_script
          source_s3_uri = "s3://${var.s3_bucket_name}/LifecycleScripts/base-config/"
        }
      },

      # Include training_plan_arn only when provided for this instance group
      try(ig.training_plan_arn, null) != null ? {
        training_plan_arn = ig.training_plan_arn
      } : {}
    )
  ]

  # Optional safety check: allow at most one instance group to carry a training plan ARN
  training_plan_groups = [
    for ig in var.instance_groups : ig.name
    if try(ig.training_plan_arn, null) != null
  ]
}

resource "awscc_sagemaker_cluster" "hyperpod_cluster" {
  depends_on = [time_sleep.wait_for_iam_role]

  cluster_name    = var.hyperpod_cluster_name
  instance_groups = local.instance_groups_list
  node_recovery   = var.node_recovery

  vpc_config = {
    security_group_ids = [var.security_group_id]
    subnets            = [var.private_subnet_id]
  }

  tags = [
    {
      key   = "Name"
      value = "${var.resource_name_prefix}-hyperpod-cluster"
    }
  ]

  lifecycle {
    precondition {
      condition     = length(local.training_plan_groups) <= 1
      error_message = "At most one instance group may include training_plan_arn. Found training_plan_arn in: ${join(", ", local.training_plan_groups)}"
    }
  }
}
