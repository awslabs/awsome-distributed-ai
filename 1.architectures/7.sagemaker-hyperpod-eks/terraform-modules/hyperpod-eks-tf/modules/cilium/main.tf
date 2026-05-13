locals {
  overlay_values = {
    routingMode    = "tunnel"
    tunnelProtocol = "vxlan"
    ipam = {
      mode = "cluster-pool"
    }
  }

  eni_values = {
    eni = {
      enabled = true
    }
    ipam = {
      mode = "eni"
    }
    routingMode          = "native"
    enableIPv4Masquerade = false
  }

  chaining_values = {
    cni = {
      chainingMode = "aws-cni"
      exclusive    = false
    }
    enableIPv4Masquerade = false
    routingMode          = "native"
  }

  base_values = {
    overlay  = local.overlay_values
    eni      = local.eni_values
    chaining = local.chaining_values
    custom   = {}
  }

  # Merge: user values override base values. In custom mode, base is empty.
  effective_values = merge(local.base_values[var.cilium_mode], var.cilium_helm_values)
}

resource "helm_release" "cilium" {
  name       = "cilium"
  repository = "https://helm.cilium.io/"
  chart      = "cilium"
  version    = var.cilium_version
  namespace  = "kube-system"

  values = [yamlencode(local.effective_values)]

  # Do not wait for DaemonSet readiness — Cilium pods can only schedule
  # once HyperPod nodes join the cluster, which happens after this module.
  wait          = false
  wait_for_jobs = false
  timeout       = 600
}

# IAM policy for ENI mode — allows Cilium operator to manage ENIs.
# Attached to the SageMaker execution role used by HyperPod instances.
resource "aws_iam_role_policy" "cilium_eni" {
  count = var.cilium_mode == "eni" && var.sagemaker_execution_role_name != "" ? 1 : 0
  name  = "cilium-eni-policy"
  role  = var.sagemaker_execution_role_name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ec2:CreateNetworkInterface",
        "ec2:AttachNetworkInterface",
        "ec2:DeleteNetworkInterface",
        "ec2:DescribeNetworkInterfaces",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeVpcs",
        "ec2:AssignPrivateIpAddresses",
        "ec2:UnassignPrivateIpAddresses",
        "ec2:ModifyNetworkInterfaceAttribute",
      ]
      Resource = "*"
    }]
  })
}
