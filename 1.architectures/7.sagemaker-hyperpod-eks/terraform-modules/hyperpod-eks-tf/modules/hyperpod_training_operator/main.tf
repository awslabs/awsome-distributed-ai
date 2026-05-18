# IAM Role for HPTO
resource "aws_iam_role" "hpto_role" {
  name = "${var.resource_name_prefix}-hpto-role"
  path = "/"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowEksAuthToAssumeRoleForPodIdentity"
        Effect = "Allow"
        Principal = {
          Service = "pods.eks.amazonaws.com"
        }
        Action = [
          "sts:AssumeRole",
          "sts:TagSession"
        ]
      }
    ]
  })
}

# IAM Policy for HPTO
resource "aws_iam_role_policy_attachment" "hpto-policy" {
  role       = aws_iam_role.hpto_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerHyperPodTrainingOperatorAccess"
}

# Pod Identity association for HPTO
#
# This is declared as a standalone aws_eks_pod_identity_association resource
# rather than an inline `pod_identity_association` block inside the
# aws_eks_addon resource below. The inline form requires the addon's metadata
# to advertise Pod Identity support via `requiresIamPermissions: true` in
# DescribeAddonVersions. The SageMaker HyperPod Training Operator addon
# v1.2.1-eksbuild.1 (the current default across K8s 1.28-1.35) regressed this
# flag from `true` (v1.2.0) to `false`, causing the EKS CreateAddon API to
# reject inline associations with:
#
#   InvalidParameterException: Pod Identity feature is not supported for
#   addon version: v1.2.1-eksbuild.1
#
# The standalone resource pattern matches the AWS CLI install flow documented
# at https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-eks-operator-install.html
# and is immune to future addon-metadata changes.
#
# See: https://github.com/awslabs/awsome-distributed-training/issues/1075
resource "aws_eks_pod_identity_association" "hpto_association" {
  cluster_name    = var.eks_cluster_name
  namespace       = "aws-hyperpod"
  service_account = "hp-training-operator-controller-manager"
  role_arn        = aws_iam_role.hpto_role.arn
}

# EKS Addon for HPTO
resource "aws_eks_addon" "hpto_addon" {
  cluster_name                = var.eks_cluster_name
  addon_name                  = "amazon-sagemaker-hyperpod-training-operator"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "PRESERVE"

  # Ensure the Pod Identity association exists before the addon's controller
  # pod starts so it has AWS credentials available on first boot.
  depends_on = [
    aws_eks_pod_identity_association.hpto_association
  ]
}
