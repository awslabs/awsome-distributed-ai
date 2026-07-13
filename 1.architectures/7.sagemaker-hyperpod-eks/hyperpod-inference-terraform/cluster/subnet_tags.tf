# =============================================================================
# Subnet tagging so the operator's internal ALB lands in the GPU AZ
# =============================================================================
# The AWS Load Balancer Controller (bundled with the inference operator)
# auto-discovers subnets for an internal ALB via the tag:
#   kubernetes.io/role/internal-elb = 1
#
# By default only the small /28 EKS control-plane subnets carry that tag, and
# they live in the first two AZs (e.g. us-east-1a/1b). If your GPU instance
# group runs in a different AZ (e.g. use1-az6 / us-east-1c), the ALB lands in
# the wrong AZs and the inference pod targets are stuck in "unused"
# (Target.NotInUse) because an ALB only routes to IP targets in its enabled
# AZs.
#
# Fix: make the large HyperPod private subnets (one per AZ, including the GPU
# AZ) the eligible internal-elb subnets, and remove the tag from the /28 EKS
# subnets so there is exactly one eligible subnet per AZ. The ALB controller
# rejects (or non-deterministically resolves) more than one tagged subnet per
# AZ, so both halves of this fix are required.
# =============================================================================

# 1. Tag every HyperPod private subnet for internal-ELB discovery + cluster
#    association (declaratively — these subnets have no inline tags to fight).
resource "aws_ec2_tag" "private_subnet_internal_elb" {
  for_each    = toset(module.private_subnet.private_subnet_ids)
  resource_id = each.value
  key         = "kubernetes.io/role/internal-elb"
  value       = "1"

  depends_on = [module.eks_cluster]
}

resource "aws_ec2_tag" "private_subnet_cluster" {
  for_each    = toset(module.private_subnet.private_subnet_ids)
  resource_id = each.value
  key         = "kubernetes.io/cluster/${module.eks_cluster.eks_cluster_name}"
  value       = "shared"

  depends_on = [module.eks_cluster]
}

# 2. Remove the internal-elb tag from the /28 EKS control-plane subnets.
#    The eks_cluster module sets this tag inline on the subnet resource, so it
#    cannot be removed with aws_ec2_tag (Terraform would fight over it) and the
#    module re-adds it on every apply. We therefore remove it with an
#    idempotent local-exec that runs on every apply, ordered (via depends_on)
#    to execute AFTER the module has re-applied its inline tag.
resource "null_resource" "untag_eks_subnets_internal_elb" {
  # Always run so we win against the module re-adding the inline tag.
  triggers = {
    always_run     = timestamp()
    region         = var.region
    eks_subnet_ids = join(",", module.eks_cluster.private_subnet_ids)
  }

  provisioner "local-exec" {
    command = <<-EOT
      set -e
      for SUBNET in ${join(" ", module.eks_cluster.private_subnet_ids)}; do
        echo "Removing kubernetes.io/role/internal-elb tag from EKS subnet $SUBNET"
        aws ec2 delete-tags --region ${var.region} \
          --resources "$SUBNET" \
          --tags Key=kubernetes.io/role/internal-elb || true
      done
    EOT
  }

  depends_on = [
    module.eks_cluster,
    aws_ec2_tag.private_subnet_internal_elb,
  ]
}
