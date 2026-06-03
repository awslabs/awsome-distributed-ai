---
id: deployment-automation
name: Deployment Automation Scripts
type: Feature
priority: P1
effort: Large
impact: High
created: 2026-03-06
---

# Deployment Automation Scripts

## Problem Statement

The `README.md` contains approximately 50 manual CLI commands spanning 12
distinct phases to go from a provisioned HyperPod EKS cluster to a running
Slurm cluster with an FSDP test job. The `Docker-Build-README.md` adds another
10+ commands for the container image workflow. This makes the deployment
error-prone, time-consuming, and difficult to reproduce consistently.

## Proposed Solution

Create bash scripts (following the `ai-on-eks` reference pattern of `setup.sh`
+ `install.sh`) that automate the **Day-2 operations** -- everything after the
HyperPod EKS infrastructure is already deployed via CloudFormation or Terraform.

### Script Architecture

```
slinky-slurm/
  setup.sh        # Phase 1: Docker build + SSH key + values template
  install.sh      # Phase 2: Orchestrates all Helm installs + config
  destroy.sh      # Phase 3: Reverse teardown of Day-2 resources
```

### Phase 1: `setup.sh` (Container Image + Values Preparation)

Modeled after `ai-on-eks/.../setup.sh`. Handles:

1. **ECR authentication** to DLC base image registry
2. **Docker image build** (platform-aware: `buildx` on macOS, `build` on Linux)
3. **ECR repo creation** (idempotent)
4. **Image tag and push** to user's ECR
5. **SSH key generation** (idempotent, `~/.ssh/id_ed25519_slurm`)
6. **Values file generation** via `envsubst` from a template
   (inject `image_repository`, `image_tag`, `ssh_key`)

Flags:
- `--repo-name <name>` (default: `dlc-slurmd`)
- `--tag <tag>` (default: `25.11.1-ubuntu24.04`)
- `--region <region>` (default: AWS CLI configured region)
- `--skip-build` (use existing ECR image)
- `--profile <g5|p5>` (select hardware profile)
- `--help`

### Phase 2: `install.sh` (Cluster Installation)

Orchestrates the full Day-2 deployment. Assumes:
- HyperPod EKS cluster is deployed and `kubectl` context is configured
- Required env vars are set (`EKS_CLUSTER_NAME`, `VPC_ID`,
  `PRIVATE_SUBNET_ID`, `SECURITY_GROUP_ID`, `AWS_ACCOUNT_ID`, `AWS_REGION`)

Steps automated:

| Step | README Section | Commands |
|------|---------------|----------|
| 1 | FSx Lustre CSI | OIDC provider, IAM SA, Helm install, StorageClass |
| 2 | FSx OpenZFS CSI (optional) | IAM SA, Helm install, StorageClass |
| 3 | AWS LB Controller | IAM policy, IAM SA, Helm install |
| 4 | Slinky Prerequisites | Helm install cert-manager + prometheus |
| 5 | Slurm Operator | Download values, Helm install |
| 6 | Slurm Namespace + PVCs | Create namespace, apply PVC manifests |
| 7 | Slurm Cluster | Helm install with generated values |
| 8 | NLB Configuration | Find public subnets, annotate login service |

Flags:
- `--skip-setup` (use previously generated `slurm-values.yaml`)
- `--profile <g5|p5>` (select hardware profile)
- `--skip-fsx-openzfs` (skip optional OpenZFS, default: skip)
- `--help`

### Phase 3: `destroy.sh` (Cleanup)

Automates the README "Clean Up" section in reverse dependency order:

1. Helm uninstall Slurm cluster
2. Helm uninstall Slurm operator
3. Helm uninstall cert-manager + prometheus
4. Delete FSx PVCs and StorageClasses
5. Helm uninstall FSx CSI drivers + delete IAM service accounts
6. Helm uninstall LB Controller + delete IAM SA + IAM policy
7. (Optional, with `--include-infra` flag) Delete CFN stack or TF destroy

### Values Template

Create `g5/g5-values.yaml.template` and `p5/p5-values.yaml.template` files
with `${image_repository}`, `${image_tag}`, and `${ssh_key}` placeholders,
following the `ai-on-eks` pattern of using `envsubst` or `sed` for injection.

### Prerequisites Check

Each script should validate prerequisites at the start:
- `aws`, `kubectl`, `helm`, `eksctl`, `docker`, `envsubst` are installed
- Required env vars are set
- AWS credentials are valid
- `kubectl` context points to the correct cluster

### Key References

- **ai-on-eks reference scripts:**
  - `ai-on-eks/.../slinky-slurm/install.sh` (93 lines)
  - `ai-on-eks/.../slinky-slurm/setup.sh` (107 lines)
  - `ai-on-eks/infra/slinky-slurm/terraform/blueprint.tfvars` (component flags)
- **Manual steps to automate:**
  - `README.md` (phases 3-12 of the deployment guide)
  - `Docker-Build-README.md` (full container build workflow)

### Design Decisions

1. **Bash over Makefile** -- consistent with the ai-on-eks reference project,
   no extra tooling dependencies
2. **Day-2 scope only** -- infra deployment (CFN/TF) is a long-running
   operation (~30 min) that users typically manage separately
3. **Idempotent operations** -- scripts should be safe to re-run (use
   `--create-namespace`, `create-repository || echo exists`, etc.)
4. **Profile-aware** -- `--profile g5|p5` flag selects the appropriate values
   and parameters throughout

## Success Criteria

- [ ] TBD

## Notes

Created via feature-capture
