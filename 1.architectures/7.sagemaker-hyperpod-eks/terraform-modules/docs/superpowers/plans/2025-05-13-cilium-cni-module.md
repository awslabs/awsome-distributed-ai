# Cilium CNI Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `modules/cilium/` Terraform module that deploys Cilium via Helm chart with four mode presets (overlay, eni, chaining, custom), and wire it into the existing HyperPod EKS infrastructure with conditional VPC CNI skip and security group modifications.

**Architecture:** New self-contained module `modules/cilium/` deployed via `helm_release`. Root `main.tf` orchestrates with `count` and `depends_on`. EKS module gains `skip_vpc_cni` flag. Security group module gains conditional VXLAN rule.

**Tech Stack:** Terraform (HCL), AWS provider, Helm provider, Cilium Helm chart

---

### Task 1: Add `skip_vpc_cni` variable to EKS cluster module

**Files:**
- Modify: `hyperpod-eks-tf/modules/eks_cluster/variables.tf` (append new variable)
- Modify: `hyperpod-eks-tf/modules/eks_cluster/main.tf` (add count to vpc_cni resource)

- [ ] **Step 1: Add variable to `modules/eks_cluster/variables.tf`**

Append at end of file:

```hcl
variable "skip_vpc_cni" {
  description = "Skip deploying the VPC CNI EKS addon (used when Cilium replaces VPC CNI)."
  type        = bool
  default     = false
}
```

- [ ] **Step 2: Make `aws_eks_addon.vpc_cni` conditional in `modules/eks_cluster/main.tf`**

Change lines 103-108 from:

```hcl
resource "aws_eks_addon" "vpc_cni" {
  cluster_name      = aws_eks_cluster.cluster.name
  addon_name        = "vpc-cni"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
}
```

To:

```hcl
resource "aws_eks_addon" "vpc_cni" {
  count             = var.skip_vpc_cni ? 0 : 1
  cluster_name      = aws_eks_cluster.cluster.name
  addon_name        = "vpc-cni"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
}
```

- [ ] **Step 3: Add `eks_cluster_role_arn` output to `modules/eks_cluster/outputs.tf`**

Append at end of file (needed for ENI mode IAM policy):

```hcl
output "eks_cluster_role_arn" {
  description = "ARN of the EKS cluster IAM role"
  value       = aws_iam_role.eks_cluster_role.arn
}
```

- [ ] **Step 4: Validate**

Run: `terraform fmt -recursive modules/eks_cluster/`
Run: `terraform validate` (from `hyperpod-eks-tf/`)

- [ ] **Step 5: Commit**

```bash
git add modules/eks_cluster/
git commit -m "feat(eks_cluster): add skip_vpc_cni flag and expose cluster role ARN"
```

---

### Task 2: Add VXLAN security group rule to security group module

**Files:**
- Modify: `hyperpod-eks-tf/modules/security_group/variables.tf` (append new variable)
- Modify: `hyperpod-eks-tf/modules/security_group/main.tf` (append VXLAN rules at end)

- [ ] **Step 1: Add variable to `modules/security_group/variables.tf`**

Append at end of file:

```hcl
variable "enable_vxlan_rule" {
  description = "Add UDP 8472 intra-SG rules for Cilium VXLAN overlay mode."
  type        = bool
  default     = false
}
```

- [ ] **Step 2: Add VXLAN ingress/egress rules to `modules/security_group/main.tf`**

Append at end of file (after line 182):

```hcl

# Cilium VXLAN overlay rules
resource "aws_vpc_security_group_ingress_rule" "cilium_vxlan_ingress" {
  count = var.enable_vxlan_rule ? 1 : 0

  description                  = "Cilium VXLAN overlay traffic"
  from_port                    = 8472
  to_port                      = 8472
  ip_protocol                  = "udp"
  security_group_id            = local.security_group_id
  referenced_security_group_id = local.security_group_id
}

resource "aws_vpc_security_group_egress_rule" "cilium_vxlan_egress" {
  count = var.enable_vxlan_rule ? 1 : 0

  description                  = "Cilium VXLAN overlay traffic"
  from_port                    = 8472
  to_port                      = 8472
  ip_protocol                  = "udp"
  security_group_id            = local.security_group_id
  referenced_security_group_id = local.security_group_id
}
```

- [ ] **Step 3: Validate**

Run: `terraform fmt -recursive modules/security_group/`
Run: `terraform validate` (from `hyperpod-eks-tf/`)

- [ ] **Step 4: Commit**

```bash
git add modules/security_group/
git commit -m "feat(security_group): add conditional Cilium VXLAN UDP 8472 rules"
```

---

### Task 3: Create `modules/cilium/` module

**Files:**
- Create: `hyperpod-eks-tf/modules/cilium/variables.tf`
- Create: `hyperpod-eks-tf/modules/cilium/main.tf`
- Create: `hyperpod-eks-tf/modules/cilium/outputs.tf`

- [ ] **Step 1: Create `modules/cilium/variables.tf`**

```hcl
variable "eks_cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "cilium_mode" {
  description = "Cilium operating mode: overlay, eni, chaining, or custom."
  type        = string
  validation {
    condition     = contains(["overlay", "eni", "chaining", "custom"], var.cilium_mode)
    error_message = "cilium_mode must be one of: overlay, eni, chaining, custom."
  }
}

variable "cilium_version" {
  description = "Cilium Helm chart version to deploy."
  type        = string
  default     = "1.19.4"
}

variable "cilium_helm_values" {
  description = "Custom Helm values merged on top of mode-specific defaults. In custom mode, this IS the entire config."
  type        = any
  default     = {}
}

variable "sagemaker_execution_role_name" {
  description = "Name of the SageMaker execution IAM role (used for ENI mode IAM policy attachment)."
  type        = string
  default     = ""
}
```

- [ ] **Step 2: Create `modules/cilium/main.tf`**

```hcl
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

  wait          = true
  wait_for_jobs = true
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
```

- [ ] **Step 3: Create `modules/cilium/outputs.tf`**

```hcl
output "cilium_release_name" {
  description = "Name of the Cilium Helm release"
  value       = helm_release.cilium.name
}

output "cilium_release_namespace" {
  description = "Namespace of the Cilium Helm release"
  value       = helm_release.cilium.namespace
}

output "cilium_mode" {
  description = "Cilium operating mode that was deployed"
  value       = var.cilium_mode
}
```

- [ ] **Step 4: Validate**

Run: `terraform fmt -recursive modules/cilium/`
Run: `terraform validate` (from `hyperpod-eks-tf/`)

- [ ] **Step 5: Commit**

```bash
git add modules/cilium/
git commit -m "feat(cilium): add Cilium CNI module with overlay/eni/chaining/custom modes"
```

---

### Task 4: Add Cilium variables to root `variables.tf`

**Files:**
- Modify: `hyperpod-eks-tf/variables.tf` (append Cilium variables section)

- [ ] **Step 1: Append Cilium variables at end of root `variables.tf`**

```hcl

# ==========================================
# Cilium CNI Configuration
# ==========================================

variable "enable_cilium" {
  description = "Enable Cilium CNI. When true and creating a new EKS cluster, deploys Cilium and conditionally skips the VPC CNI addon based on cilium_mode."
  type        = bool
  default     = false
}

variable "cilium_mode" {
  description = "Cilium operating mode: overlay (VXLAN tunnel), eni (native ENI routing), chaining (policy-only on top of VPC CNI), or custom (user provides all Helm values via cilium_helm_values)."
  type        = string
  default     = "overlay"
  validation {
    condition     = contains(["overlay", "eni", "chaining", "custom"], var.cilium_mode)
    error_message = "cilium_mode must be one of: overlay, eni, chaining, custom."
  }
}

variable "cilium_version" {
  description = "Cilium Helm chart version to deploy."
  type        = string
  default     = "1.19.4"
}

variable "cilium_helm_values" {
  description = "Custom Helm values merged on top of mode-specific defaults. In custom mode, this IS the entire Helm config (no base defaults applied). For overlay/eni/chaining modes, these values override the base defaults."
  type        = any
  default     = {}
}
```

- [ ] **Step 2: Validate**

Run: `terraform fmt variables.tf`
Run: `terraform validate`

- [ ] **Step 3: Commit**

```bash
git add variables.tf
git commit -m "feat: add Cilium CNI variables to root variables.tf"
```

---

### Task 5: Wire Cilium module into root `main.tf`

**Files:**
- Modify: `hyperpod-eks-tf/main.tf` (add locals, module block, modify existing module params)

- [ ] **Step 1: Add Cilium-related locals**

In the `locals` block (after line 43, the `enable_guardduty_cleanup` line), add:

```hcl
  # Cilium CNI
  skip_vpc_cni  = var.enable_cilium && var.cilium_mode != "chaining"
  create_cilium = var.enable_cilium && var.create_eks_module
```

- [ ] **Step 2: Pass `skip_vpc_cni` to EKS module**

In the `module "eks_cluster"` block (lines 102-117), add the parameter after `endpoint_public_access`:

```hcl
  skip_vpc_cni            = local.skip_vpc_cni
```

- [ ] **Step 3: Pass `enable_vxlan_rule` to security group module**

In the `module "security_group"` block (lines 91-100), add the parameter after `create_vpc_endpoint_ingress_rule`:

```hcl
  enable_vxlan_rule                = var.enable_cilium && var.cilium_mode == "overlay"
```

- [ ] **Step 4: Add cilium module block**

Insert after the `module "eks_cluster"` block (after line 117) and before the `module "s3_bucket"` block:

```hcl
module "cilium" {
  count  = local.create_cilium ? 1 : 0
  source = "./modules/cilium"

  eks_cluster_name              = module.eks_cluster[0].eks_cluster_name
  cilium_mode                   = var.cilium_mode
  cilium_version                = var.cilium_version
  cilium_helm_values            = var.cilium_helm_values
  sagemaker_execution_role_name = local.sagemaker_iam_role_name

  depends_on = [module.eks_cluster]
}
```

- [ ] **Step 5: Add cilium to helm_chart depends_on**

Change line 211 in `module "helm_chart"` from:

```hcl
  depends_on = [module.eks_cluster]
```

To:

```hcl
  depends_on = [module.eks_cluster, module.cilium]
```

- [ ] **Step 6: Validate**

Run: `terraform fmt main.tf`
Run: `terraform validate`

- [ ] **Step 7: Commit**

```bash
git add main.tf
git commit -m "feat: wire Cilium module into root orchestration with conditional VPC CNI skip"
```

---

### Task 6: Add example tfvars and documentation

**Files:**
- Modify: `hyperpod-eks-tf/terraform.tfvars` (add commented Cilium examples)
- Modify: `terraform-modules/README.md` (add Cilium section)

- [ ] **Step 1: Add Cilium examples to `terraform.tfvars`**

Append at end of file:

```hcl

# ==========================================
# Cilium CNI (optional)
# ==========================================
# enable_cilium = true
# cilium_mode   = "overlay"  # Options: overlay, eni, chaining, custom
# cilium_version = "1.19.4"
# cilium_helm_values = {}    # Custom values merged on top of mode defaults
```

- [ ] **Step 2: Add Cilium section to `terraform-modules/README.md`**

Add a new section before the "GuardDuty Cleanup" section with the following content:

```markdown
## Cilium CNI (Optional)

You can replace the default AWS VPC CNI with [Cilium](https://cilium.io) by setting `enable_cilium = true`. This supports three pre-configured modes plus a fully custom option:

| Mode | Description | VPC CNI |
|------|-------------|---------|
| `overlay` | VXLAN tunnel, non-VPC-routable pod IPs, highest pod density | Removed |
| `eni` | Native ENI routing, VPC-routable pod IPs (like VPC CNI) | Removed |
| `chaining` | VPC CNI handles networking, Cilium adds eBPF policy/LB | Kept |
| `custom` | User provides all Helm values, no defaults applied | Removed |

### New EKS Cluster with Cilium

```hcl
enable_cilium  = true
cilium_mode    = "overlay"
cilium_version = "1.19.4"

# Optional: override specific Helm values on top of mode defaults
cilium_helm_values = {
  hubble = {
    enabled = true
  }
}
```

### Existing EKS Cluster with Cilium Already Installed

If you are integrating HyperPod with an existing EKS cluster that already has Cilium running:

```hcl
create_eks_module = false
existing_eks_cluster_name = "my-cilium-cluster"
enable_cilium = true
cilium_mode   = "overlay"  # Match your existing Cilium configuration
```

Setting `enable_cilium = true` with `create_eks_module = false` will:
- Skip Cilium deployment (it's already on your cluster)
- Add appropriate security group rules (e.g., VXLAN UDP 8472 for overlay mode)
- Skip VPC CNI addon creation

### Custom Mode

For full control over the Cilium Helm chart configuration:

```hcl
enable_cilium = true
cilium_mode   = "custom"
cilium_helm_values = {
  # Your complete Cilium Helm values here
  routingMode = "native"
  ipam = {
    mode = "eni"
  }
  eni = {
    enabled = true
  }
  hubble = {
    enabled = true
    relay = {
      enabled = true
    }
  }
}
```

### Limitations

- **Closed network:** Cilium images must be pre-staged to ECR in closed-network deployments. Extend `tools/copy-images-to-ecr.sh` as needed.
- **ENI mode:** IPv4 only. Pod count bounded by ENI/IP limits per instance type.
- **Overlay mode:** Pod-to-VPC traffic is SNATed. Webhooks must be host-networked or exposed via Service/Ingress.
- **Chaining mode:** Some Cilium features limited (L7 policy, IPsec encryption).
```

- [ ] **Step 3: Validate formatting**

Run: `terraform fmt terraform.tfvars`

- [ ] **Step 4: Commit**

```bash
git add terraform.tfvars
git add ../../README.md
git commit -m "docs: add Cilium CNI configuration examples and README section"
```

---

### Task 7: Final validation

**Files:** None (validation only)

- [ ] **Step 1: Format all Terraform files**

Run: `terraform fmt -recursive .` (from `hyperpod-eks-tf/`)

- [ ] **Step 2: Validate full configuration**

Run: `terraform validate` (from `hyperpod-eks-tf/`)

Note: This may fail without AWS credentials or a valid state. If it passes syntax/reference validation, that's sufficient.

- [ ] **Step 3: Review `terraform plan` output (if credentials available)**

Run: `terraform plan -var-file="custom.tfvars" -var="enable_cilium=true" -var='cilium_mode=overlay'`

Verify: Plan shows the cilium module resources being created, VPC CNI addon being skipped, VXLAN SG rules being added.

- [ ] **Step 4: Final commit (if fmt made changes)**

```bash
git add -A
git commit -m "chore: terraform fmt"
```
