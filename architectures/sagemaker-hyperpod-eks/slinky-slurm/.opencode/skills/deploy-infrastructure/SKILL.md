---
name: deploy-infrastructure
description: Deploy HyperPod EKS infrastructure using deploy.sh via CloudFormation or Terraform, including AZ resolution, stack outputs, and kubeconfig setup
---

# Deploy HyperPod EKS Infrastructure

## Overview

Use this skill to deploy the HyperPod EKS infrastructure that hosts the Slurm
cluster. This is **Phase 1** of the slinky-slurm deployment workflow and must
complete before running `setup.sh` or `install.sh`.

The `deploy.sh` script supports two infrastructure backends:
- **CloudFormation** (`--infra cfn`) -- uses an AWS-hosted S3 template
- **Terraform** (`--infra tf`) -- uses local Terraform modules

Deployment takes approximately 20-30 minutes and provisions:
- VPC with public/private subnets
- EKS cluster
- HyperPod instance groups (management + accelerated)
- FSx Lustre file system
- Security groups and IAM roles

## Prerequisites

Before running `deploy.sh`, verify:

1. **AWS CLI** installed and configured with valid credentials
2. **jq** installed (for `--infra cfn`) or **terraform** installed (for
   `--infra tf`)
3. A valid AWS region and AZ ID for instance placement
4. No existing stack with the same name (default: `hp-eks-slinky-stack`)

See the `deployment-preflight` skill for detailed prerequisite validation.

## Steps

### Step 1: Choose instance type and infrastructure backend

Any SageMaker-supported GPU instance type can be used. GPU count, model,
and EFA interfaces are auto-discovered via the EC2 API. Two common
configurations:

| Instance Type | Count | GPUs | EFA |
|---------------|-------|------|-----|
| `ml.g5.8xlarge` (default) | 4 | 1 A10G per node | 1 per node |
| `ml.p5.48xlarge` | 2 | 8 H100 per node | 32 per node |

### Step 2: Run deploy.sh

**CloudFormation path (recommended for first-time users):**

```bash
# Deploy g5 in us-west-2 (defaults)
bash deploy.sh --instance-type ml.g5.8xlarge --infra cfn

# Deploy p5 in us-east-1 with a specific AZ
bash deploy.sh --instance-type ml.p5.48xlarge --instance-count 2 --infra cfn --region us-east-1 --az-id use1-az2

# Deploy with custom stack name
bash deploy.sh --instance-type ml.g5.8xlarge --infra cfn --stack-name my-slinky-stack
```

**Terraform path:**

```bash
# Deploy g5 using Terraform
bash deploy.sh --instance-type ml.g5.8xlarge --infra tf

# Deploy p5 in us-east-1
bash deploy.sh --instance-type ml.p5.48xlarge --instance-count 2 --infra tf --region us-east-1 --az-id use1-az2
```

The Terraform path will show a plan and prompt for confirmation before
applying.

### Step 3: Source environment variables

After `deploy.sh` completes, it writes `env_vars.sh` containing stack
outputs:

```bash
source env_vars.sh
```

This sets:
- `AWS_ACCOUNT_ID`
- `AWS_REGION`
- `EKS_CLUSTER_NAME`
- `VPC_ID`
- `PRIVATE_SUBNET_ID`
- `SECURITY_GROUP_ID`
- `STACK_ID`

### Step 4: Update kubeconfig

```bash
aws eks update-kubeconfig --name $EKS_CLUSTER_NAME --region $AWS_REGION
```

### Step 5: Verify cluster connectivity

```bash
kubectl cluster-info
kubectl get nodes
```

Expected output: nodes with instance types matching the chosen profile
(`ml.m5.4xlarge` for management, `ml.g5.8xlarge` or `ml.p5.48xlarge` for
accelerated).

## What deploy.sh Does Internally

### CloudFormation Path

1. Sources `lib/deploy_helpers.sh`
2. Calls `resolve_instance_profile()` to set instance type/count for the profile
3. Validates AWS credentials via `aws sts get-caller-identity`
4. Resolves AZ IDs for the region via `aws ec2 describe-availability-zones`
5. Validates the specified `--az-id` against the resolved list
6. Reads `params.json` (40 CloudFormation parameters, g5 defaults)
7. Calls `resolve_cfn_params()` to substitute AZ IDs and instance overrides
8. Calls `validate_cfn_template()` to:
   - Validate the S3-hosted template is reachable and syntactically valid
     via `aws cloudformation validate-template --template-url`
   - Cross-check parameter keys in the resolved params against the
     template's declared parameters
   - Error if required parameters (no default value) are missing
   - Warn if extra parameters are provided that the template doesn't expect
9. Calls `aws cloudformation create-stack` (or `update-stack` for existing
   stacks) with the S3-hosted HyperPod template
10. Waits for stack completion (20-30 minutes)
11. Extracts outputs to `env_vars.sh`

### Terraform Path

1. Same initial steps (1-5) as CloudFormation
2. Copies `custom.tfvars` to the terraform-modules directory
3. Calls `resolve_tf_vars()` to patch region, AZ, and instance overrides
4. Runs `terraform init`, `plan`, and `apply` (with user confirmation)
5. Runs `terraform_outputs.sh` to extract outputs to `env_vars.sh`

## Command Reference

```
Usage: deploy.sh --instance-type <ml.X.Y> --infra <cfn|tf> [OPTIONS]

Required:
  --instance-type <type>    SageMaker instance type (e.g. ml.g5.8xlarge)
  --infra <cfn|tf>          Infrastructure deployment method

Optional:
  --instance-count <N>      Number of accelerated instances (default: 4)
  --region <region>         AWS region (default: us-west-2)
  --az-id <az-id>           Availability zone ID (default: usw2-az2)
  --stack-name <name>       CFN stack name (default: hp-eks-slinky-stack)
  --help                    Show help
```

## Verification

Deployment is successful when:

- `deploy.sh` exits with code 0
- `env_vars.sh` exists and contains all 7 exported variables
- `kubectl get nodes` shows nodes in Ready state
- `kubectl cluster-info` returns the EKS API server endpoint

```bash
# Quick verification
source env_vars.sh
echo "Cluster: $EKS_CLUSTER_NAME"
echo "Region: $AWS_REGION"
kubectl get nodes -o wide
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Stack creation fails with capacity error | AZ doesn't have capacity for the instance type | Try a different `--az-id` |
| `ROLLBACK_COMPLETE` status | CFN template parameter issue | Check CloudFormation events: `aws cloudformation describe-stack-events --stack-name <name>` |
| Terraform plan fails | Missing terraform-modules directory | Ensure the full `awsome-distributed-ai` repo is cloned, not just the slinky-slurm subdirectory |
| `env_vars.sh` not created | Script failed before output extraction | Check the script output for errors and re-run |
| AZ ID validation warning | Specified AZ not in the region | Use one of the AZ IDs shown in the "Available AZs" message |
| Stack already exists | Previous deployment not cleaned up | Run `destroy.sh` first, or use a different `--stack-name` |
| Template validation fails | S3-hosted template unreachable or params mismatch | Check network/credentials; review the missing/extra parameter warnings in the output |

## References

- `deploy.sh` -- Main infrastructure deployment script
- `lib/deploy_helpers.sh` -- Helper functions (`resolve_instance_profile`,
  `resolve_cfn_params`, `resolve_tf_vars`, `validate_az_id`,
  `validate_cfn_template`, `check_command`)
- `params.json` -- CloudFormation parameters (40 params, g5 defaults)
- `custom.tfvars` -- Terraform variables (g5 defaults)
