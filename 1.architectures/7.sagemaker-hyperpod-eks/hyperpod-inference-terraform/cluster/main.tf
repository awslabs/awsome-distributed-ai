# =============================================================================
# Cluster stage — full HyperPod EKS cluster + inference operator from scratch
# =============================================================================
# Reuses the canonical leaf modules from the reference stack so that the
# underlying infrastructure stays in lockstep with the upstream terraform
# modules:
#   ../../terraform-modules/hyperpod-eks-tf/modules/*
#
# Resource creation order (enforced via depends_on):
#   vpc -> private_subnet -> security_group -> eks_cluster -> s3_bucket
#     -> lifecycle_script -> vpc_endpoints -> sagemaker_iam_role
#     -> helm_chart -> hyperpod_cluster (+ cert-manager)
#     -> hyperpod_inference_operator
# =============================================================================

locals {
  modules_root = "../../terraform-modules/hyperpod-eks-tf/modules"

  eks_private_subnet_cidrs = [
    var.eks_private_subnet_1_cidr,
    var.eks_private_subnet_2_cidr,
  ]

  # AZ-id -> subnet-id map for the HyperPod private subnets.
  az_to_subnet_map = module.private_subnet.az_to_subnet_map
}

data "aws_region" "current" {}

# --- VPC (with public subnets, IGW, and a NAT gateway) ---
module "vpc" {
  source = "../../terraform-modules/hyperpod-eks-tf/modules/vpc"

  resource_name_prefix = var.resource_name_prefix
  vpc_cidr             = var.vpc_cidr
  public_subnet_1_cidr = var.public_subnet_1_cidr
  public_subnet_2_cidr = var.public_subnet_2_cidr
  closed_network       = false
  tags                 = var.tags
}

# --- HyperPod private subnets (one large subnet per AZ, NAT-routed) ---
module "private_subnet" {
  source = "../../terraform-modules/hyperpod-eks-tf/modules/private_subnet"

  resource_name_prefix = var.resource_name_prefix
  vpc_id               = module.vpc.vpc_id
  private_subnet_cidrs = var.private_subnet_cidrs
  nat_gateway_id       = module.vpc.nat_gateway_1_id
  closed_network       = false
  tags                 = var.tags
}

# --- Cluster security group (no ingress; intra-SG traffic allowed) ---
module "security_group" {
  source = "../../terraform-modules/hyperpod-eks-tf/modules/security_group"

  resource_name_prefix             = var.resource_name_prefix
  vpc_id                           = module.vpc.vpc_id
  create_new_sg                    = true
  existing_security_group_id       = ""
  create_vpc_endpoint_ingress_rule = true
}

# --- EKS cluster (creates its own /28 control-plane subnets + core addons) ---
module "eks_cluster" {
  source = "../../terraform-modules/hyperpod-eks-tf/modules/eks_cluster"

  resource_name_prefix    = var.resource_name_prefix
  vpc_id                  = module.vpc.vpc_id
  eks_cluster_name        = var.eks_cluster_name
  kubernetes_version      = var.kubernetes_version
  security_group_id       = module.security_group.security_group_id
  create_eks_subnets      = true
  existing_eks_subnet_ids = []
  private_subnet_cidrs    = local.eks_private_subnet_cidrs
  nat_gateway_id          = module.vpc.nat_gateway_1_id
  endpoint_private_access = var.eks_endpoint_private_access
  endpoint_public_access  = var.eks_endpoint_public_access
}

# --- S3 bucket for lifecycle scripts (+ access logs bucket) ---
module "s3_bucket" {
  source = "../../terraform-modules/hyperpod-eks-tf/modules/s3_bucket"

  resource_name_prefix = var.resource_name_prefix
  tags                 = var.tags
}

# --- VPC endpoints (S3 gateway endpoint by default) ---
module "vpc_endpoints" {
  source = "../../terraform-modules/hyperpod-eks-tf/modules/vpc_endpoints"

  resource_name_prefix    = var.resource_name_prefix
  vpc_id                  = module.vpc.vpc_id
  private_route_table_ids = module.private_subnet.private_route_table_ids
  private_subnet_ids      = module.private_subnet.private_subnet_ids
  security_group_id       = module.security_group.security_group_id
  rig_mode                = false
  rig_rft_lambda_access   = false
  rig_rft_sqs_access      = false
  tags                    = var.tags

  depends_on = [
    module.private_subnet,
    module.security_group,
  ]
}

# --- Lifecycle scripts uploaded to the S3 bucket ---
module "lifecycle_script" {
  source = "../../terraform-modules/hyperpod-eks-tf/modules/lifecycle_script"

  resource_name_prefix = var.resource_name_prefix
  s3_bucket_name       = module.s3_bucket.s3_bucket_name
}

# --- SageMaker execution role for the HyperPod cluster nodes ---
module "sagemaker_iam_role" {
  source = "../../terraform-modules/hyperpod-eks-tf/modules/sagemaker_iam_role"

  resource_name_prefix  = var.resource_name_prefix
  s3_bucket_name        = module.s3_bucket.s3_bucket_name
  rig_input_s3_bucket   = null
  rig_output_s3_bucket  = null
  eks_cluster_arn       = module.eks_cluster.eks_cluster_arn
  security_group_id     = module.security_group.security_group_id
  private_subnet_ids    = module.private_subnet.private_subnet_ids
  vpc_id                = module.vpc.vpc_id
  rig_mode              = false
  gated_access          = var.gated_access
  rig_rft_lambda_access = false
  rig_rft_sqs_access    = false
  karpenter_autoscaling = false
}

# --- Bootstrap: clone the HyperPod dependencies Helm chart repo ---
# The upstream helm_chart module sources the chart from a local clone at
# /tmp/helm-repo (it runs `git checkout <revision>` there and reads the chart
# off disk). Rather than make this an out-of-band manual prerequisite, we clone
# it idempotently here so the stack works from a clean machine or CI runner.
resource "null_resource" "helm_repo" {
  # Re-evaluate if the target revision or repo URL changes.
  triggers = {
    repo_url = var.helm_repo_url
    revision = var.helm_repo_revision
    path     = var.helm_repo_local_path
  }

  provisioner "local-exec" {
    interpreter = ["/bin/bash", "-c"]
    command     = <<-EOT
      set -euo pipefail
      REPO_DIR="${var.helm_repo_local_path}"
      if [ -d "$REPO_DIR/.git" ]; then
        echo "Helm repo already present at $REPO_DIR; fetching latest..."
        cd "$REPO_DIR"
        # Make sure it points at the expected origin, then refresh.
        git remote set-url origin "${var.helm_repo_url}" 2>/dev/null || \
          git remote add origin "${var.helm_repo_url}"
        git fetch --all --tags --prune
      else
        echo "Cloning ${var.helm_repo_url} to $REPO_DIR..."
        rm -rf "$REPO_DIR"
        git clone "${var.helm_repo_url}" "$REPO_DIR"
      fi
      # Verify the pinned revision is resolvable so the helm_chart module's
      # `git checkout <revision>` cannot fail later in the apply.
      cd "$REPO_DIR"
      git checkout "${var.helm_repo_revision}"
      echo "Helm repo ready at $REPO_DIR @ ${var.helm_repo_revision}"
    EOT
  }
}

# --- HyperPod dependencies Helm chart (device plugins, operators, etc.) ---
module "helm_chart" {
  source = "../../terraform-modules/hyperpod-eks-tf/modules/helm_chart"

  resource_name_prefix                = var.resource_name_prefix
  helm_repo_path                      = var.helm_repo_path
  helm_release_name                   = var.helm_release_name
  helm_repo_revision                  = var.helm_repo_revision
  helm_repo_revision_rig              = var.helm_repo_revision
  namespace                           = var.helm_namespace
  eks_cluster_name                    = module.eks_cluster.eks_cluster_name
  enable_gpu_operator                 = false
  enable_mlflow                       = false
  enable_kubeflow_training_operators  = true
  enable_cluster_role_and_bindings    = false
  enable_namespaced_role_and_bindings = false
  enable_team_role_and_bindings       = false
  enable_nvidia_device_plugin         = true
  enable_neuron_device_plugin         = true
  enable_mpi_operator                 = true
  enable_deep_health_check            = true
  enable_job_auto_restart             = true
  enable_hyperpod_patching            = true
  rig_mode                            = false

  depends_on = [
    module.eks_cluster,
    null_resource.helm_repo,
  ]
}

# --- HyperPod cluster (the GPU nodes) + cert-manager add-on ---
module "hyperpod_cluster" {
  source = "../../terraform-modules/hyperpod-eks-tf/modules/hyperpod_cluster"

  resource_name_prefix         = var.resource_name_prefix
  hyperpod_cluster_name        = var.hyperpod_cluster_name
  auto_node_recovery           = var.auto_node_recovery
  instance_groups              = var.instance_groups
  restricted_instance_groups   = []
  private_subnet_ids           = module.private_subnet.private_subnet_ids
  az_to_subnet_map             = local.az_to_subnet_map
  security_group_id            = module.security_group.security_group_id
  eks_cluster_name             = module.eks_cluster.eks_cluster_name
  eks_cluster_arn              = module.eks_cluster.eks_cluster_arn
  s3_bucket_name               = module.s3_bucket.s3_bucket_name
  sagemaker_iam_role_name      = module.sagemaker_iam_role.sagemaker_iam_role_name
  rig_mode                     = false
  karpenter_autoscaling        = false
  continuous_provisioning_mode = var.continuous_provisioning_mode
  karpenter_role_arn           = null
  wait_for_nodes               = true
  enable_cert_manager          = var.enable_cert_manager

  depends_on = [
    module.helm_chart,
    module.eks_cluster,
    module.private_subnet,
    module.security_group,
    module.s3_bucket,
    module.vpc_endpoints,
    module.sagemaker_iam_role,
    module.lifecycle_script,
  ]
}

# --- FSx for Lustre CSI driver (required by the inference operator) ---
# The inference operator's init container (check-csi-drivers) requires the FSx
# Lustre CSI driver to be present. We install only the driver (no filesystem).
module "fsx_lustre" {
  source = "../../terraform-modules/hyperpod-eks-tf/modules/fsx_lustre"

  resource_name_prefix       = var.resource_name_prefix
  eks_cluster_name           = module.eks_cluster.eks_cluster_name
  subnet_id                  = module.hyperpod_cluster.primary_subnet_id
  security_group_id          = module.security_group.security_group_id
  create_new_filesystem      = false
  inference_operator_enabled = true

  depends_on = [module.hyperpod_cluster]
}

# --- HyperPod inference operator add-on (+ ALB controller, KEDA, S3 CSI) ---
module "hyperpod_inference_operator" {
  source = "../../terraform-modules/hyperpod-eks-tf/modules/hyperpod_inference_operator"

  resource_name_prefix    = var.resource_name_prefix
  eks_cluster_name        = module.eks_cluster.eks_cluster_name
  hyperpod_cluster_arn    = module.hyperpod_cluster.hyperpod_cluster_arn
  access_logs_bucket_name = module.s3_bucket.s3_logs_bucket_name
  enable_s3_csi_driver    = var.enable_s3_csi_driver
  enable_alb_controller   = var.enable_alb_controller
  enable_keda             = var.enable_keda
  enable_metrics_server   = var.enable_metrics_server

  inference_operator_create_timeout = var.inference_operator_create_timeout

  depends_on = [
    module.hyperpod_cluster,
    module.fsx_lustre,
  ]
}
