---
name: build-slurm-image
description: Build the Slurm compute node container image via CodeBuild or local Docker, generate SSH keys, and render Helm values using setup.sh
---

# Build Slurm Image & Generate Values

## Overview

Use this skill to build the Slurmd DLC (Deep Learning Container) image, generate
SSH keys for login node access, and render the `slurm-values.yaml` Helm values
file from the template. This is **Phase 2** of the slinky-slurm deployment
workflow.

The `setup.sh` script handles three image build paths:

| Mode | Flag | When to Use |
|------|------|-------------|
| **CodeBuild** (default) | *(none)* | Production builds in AWS (creates a CodeBuild project to build the image) |
| **Local Docker** | `--local-build` | Development/testing on a local machine with Docker |
| **Skip Build** | `--skip-build` | Image already exists in ECR from a prior build |

After the image build, `setup.sh` always:
1. Generates an SSH key pair (`~/.ssh/id_ed25519_slurm`) if not present
2. Renders `slurm-values.yaml` from `slurm-values.yaml.template` with
   profile-specific values (GPU count, EFA count, GRES, replicas, etc.)

## Prerequisites

- AWS CLI configured with valid credentials
- `--instance-type` and `--infra` flags decided
- For **CodeBuild** (default):
  - `env_vars.sh` available (from `deploy.sh`) for AWS_ACCOUNT_ID, AWS_REGION
  - `zip` command available
  - `jq` (for CFN) or `terraform` (for TF)
- For **Local Docker**:
  - Docker Desktop running
  - Access to DLC ECR registry (`763104351884.dkr.ecr.us-east-1.amazonaws.com`)
- For **Skip Build**:
  - Image must already exist in ECR

See the `deployment-preflight` skill for full prerequisite validation.

## Steps

### Step 1: Choose build mode and run setup.sh

**CodeBuild (default) -- recommended for production:**

```bash
# Build via CodeBuild for g5 instances using CloudFormation
bash setup.sh --instance-type ml.g5.8xlarge --infra cfn

# Build via CodeBuild for p5 instances using Terraform
bash setup.sh --instance-type ml.p5.48xlarge --instance-count 2 --infra tf
```

**Local Docker -- for development/testing:**

```bash
# Build locally for g5 instances
bash setup.sh --instance-type ml.g5.8xlarge --infra cfn --local-build

# Build locally for p5 instances
bash setup.sh --instance-type ml.p5.48xlarge --instance-count 2 --infra tf --local-build
```

**Skip Build -- image already in ECR:**

```bash
# Use existing ECR image
bash setup.sh --instance-type ml.g5.8xlarge --infra cfn --skip-build

# With custom repo name and tag
bash setup.sh --instance-type ml.g5.8xlarge --infra cfn --skip-build \
    --repo-name my-slurmd --tag v1.0
```

### Step 2: Verify outputs

After `setup.sh` completes:

```bash
# Verify slurm-values.yaml was generated
cat slurm-values.yaml | head -20

# Check for unresolved template variables (should find none)
grep '${' slurm-values.yaml
# Expected: no output (all variables substituted)

# Verify SSH key exists
ls -la ~/.ssh/id_ed25519_slurm*
```

## What setup.sh Does Internally

### CodeBuild Path (Default)

1. Sources `lib/deploy_helpers.sh` and calls `resolve_helm_profile()`
2. Gets AWS account ID via `aws sts get-caller-identity`
3. Sources `env_vars.sh` if available (for AWS_ACCOUNT_ID, AWS_REGION)
4. Creates an S3 bucket for build context:
   `dlc-slurmd-codebuild-<account_id>-<region>`
5. Packages `dlc-slurmd.Dockerfile` + `buildspec.yml` into a zip file
6. Uploads zip to `s3://<bucket>/codebuild/slurmd-build-context.zip`
7. Deploys CodeBuild stack:
   - **CFN path**: `aws cloudformation create-stack` with
     `codebuild-stack.yaml`
   - **TF path**: `terraform apply` with `codebuild.tf`
8. Gets the CodeBuild project name from stack outputs
9. Starts the build: `aws codebuild start-build`
10. Polls build status every 15 seconds until SUCCEEDED/FAILED
11. On success, the image is in ECR at
    `<account>.dkr.ecr.<region>.amazonaws.com/dlc-slurmd:<tag>`

### Local Docker Path (`--local-build`)

1. Authenticates to the DLC ECR registry in `us-east-1`:
   ```bash
   aws ecr get-login-password --region us-east-1 | \
       docker login --username AWS \
       --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
   ```
2. Builds the image (platform-aware):
   - **macOS**: `docker buildx build --platform linux/amd64`
   - **Linux**: `docker build`
3. Creates ECR repository if it doesn't exist
4. Authenticates to the project's ECR registry
5. Tags and pushes the image

### Skip Build Path (`--skip-build`)

1. Verifies the image exists in ECR:
   ```bash
   aws ecr describe-images \
       --repository-name dlc-slurmd \
       --image-ids imageTag=25.11.1-ubuntu24.04
   ```
2. Fails with an error if the image is not found

### SSH Key Generation (All Paths)

- Checks for `~/.ssh/id_ed25519_slurm`
- If missing, generates a new ed25519 key pair with `ssh-keygen`
- The public key is embedded into `slurm-values.yaml` for login node access

### Values File Rendering (All Paths)

Calls `resolve_helm_profile()` which sets these variables based on
`--instance-type`:

| Variable | g5 Value | p5 Value |
|----------|----------|----------|
| `HELM_ACCEL_INSTANCE_TYPE` | `ml.g5.8xlarge` | `ml.p5.48xlarge` |
| `GPU_COUNT` | `1` | `8` |
| `EFA_COUNT` | `1` | `32` |
| `GPU_GRES` | `gpu:a10g:1` | `gpu:h100:8` |
| `REPLICAS` | `4` | `2` |
| `MGMT_INSTANCE_TYPE` | `ml.m5.4xlarge` | `ml.m5.4xlarge` |
| `PVC_NAME` | `fsx-claim` | `fsx-claim` |

Then uses `sed` to substitute 10 template variables in
`slurm-values.yaml.template`:
- `${image_repository}`, `${image_tag}`, `${ssh_key}`
- `${mgmt_instance_type}`, `${accel_instance_type}`
- `${gpu_count}`, `${efa_count}`, `${gpu_gres}`
- `${replicas}`, `${pvc_name}`

## Command Reference

```
Usage: setup.sh --instance-type <ml.X.Y> --infra <cfn|tf> [OPTIONS]

Required:
  --instance-type <type>    SageMaker instance type for GPU/EFA/GRES resolution
  --infra <cfn|tf>          Infrastructure method for CodeBuild stack

Optional:
  --instance-count <N>      Number of compute node replicas (default: varies by instance type)
  --repo-name <name>        ECR repository name (default: dlc-slurmd)
  --tag <tag>               Image tag (default: 25.11.1-ubuntu24.04)
  --region <region>         AWS region (default: AWS CLI configured or us-west-2)
  --local-build             Build image locally instead of CodeBuild
  --skip-build              Skip image build (use existing image in ECR)
  --help                    Show help
```

## Verification

Build is successful when:

- `setup.sh` exits with code 0
- `slurm-values.yaml` exists in the project directory
- No unresolved `${...}` variables in `slurm-values.yaml`
- SSH key pair exists at `~/.ssh/id_ed25519_slurm`
- ECR image exists (verify with `aws ecr describe-images`)

```bash
# Quick verification
ls -la slurm-values.yaml
grep -c '${' slurm-values.yaml  # Should return 0 / exit 1
ls ~/.ssh/id_ed25519_slurm

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr describe-images \
    --repository-name dlc-slurmd \
    --image-ids imageTag=25.11.1-ubuntu24.04 \
    --region "${AWS_REGION:-us-west-2}"
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| CodeBuild FAILED status | Dockerfile build error or DLC base image pull failure | Check CodeBuild logs: `aws codebuild batch-get-builds --ids <build-id>` |
| Docker login fails for DLC ECR | Region mismatch | DLC registry is always in `us-east-1`, not the deployment region |
| `docker buildx` fails on macOS | Docker Desktop not running or buildx not enabled | Start Docker Desktop; ensure buildx is available: `docker buildx version` |
| ECR image not found (`--skip-build`) | Wrong repo name, tag, or region | Verify with `aws ecr describe-images --repository-name <name>` |
| Template variables not substituted | `resolve_helm_profile` failed | Check that `--instance-type` is a valid instance type |
| S3 bucket creation fails | Bucket name already taken | The bucket name includes account ID and region; check IAM permissions |
| CodeBuild stack already exists | Prior run created it | Script handles this gracefully (skips creation) |

## References

- `setup.sh` -- Main image build and values generation script
- `lib/deploy_helpers.sh` -- `resolve_helm_profile()` function (lines 43-70)
- `dlc-slurmd.Dockerfile` -- Multi-stage Dockerfile for Slurm compute node
- `slurm-values.yaml.template` -- Helm values template with 10 variables
- `buildspec.yml` -- CodeBuild build specification
- `codebuild-stack.yaml` -- CloudFormation template for CodeBuild project
- `codebuild.tf` -- Terraform config for CodeBuild project
