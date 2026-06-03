---
id: dockerfile-update-codebuild-ci
name: Dockerfile Update and CodeBuild CI Pipeline
type: Feature
priority: P1
effort: Medium
impact: High
created: 2026-03-06
---

# Dockerfile Update and CodeBuild CI Pipeline

## Problem Statement

The `dlc-slurmd.Dockerfile` needs to be updated to reflect the latest changes
from the upstream reference at
`ai-on-eks-slurm/ai-on-eks/blueprints/training/slinky-slurm/dlc-slurmd.Dockerfile`.
Additionally, the current workflow requires building the container image locally
and pushing it to ECR manually. This should be replaced with an AWS CodeBuild
pipeline that automates the build-and-push process.

## Proposed Solution

### Part 1: Dockerfile Update

Update the slurmd base image version in `dlc-slurmd.Dockerfile`:

```diff
- FROM ghcr.io/slinkyproject/slurmd:25.05.0-ubuntu24.04
+ FROM ghcr.io/slinkyproject/slurmd:25.11.1-ubuntu24.04
```

This is the only difference between the current file and the upstream reference.
All other layers (CUDA, EFA, OpenMPI, NCCL, Python, OpenSSH) are identical.

### Part 2: CodeBuild CI Pipeline

Deploy an AWS CodeBuild project that:

1. Authenticates to the DLC ECR registry
   (`763104351884.dkr.ecr.<region>.amazonaws.com`) for the base image pull
2. Builds the `dlc-slurmd` image using `docker buildx`
3. Tags and pushes the image to the project's ECR repository
4. Triggers on relevant changes (Dockerfile modifications, manual dispatch)

Artifacts to create:

- `buildspec.yml` -- CodeBuild build specification
- CloudFormation or Terraform resource for the CodeBuild project, including:
  - IAM role with ECR push permissions and DLC ECR pull permissions
  - ECR repository (if not already created by the infra stack)
  - Build environment (privileged mode for Docker builds)
  - Source configuration (GitHub or S3)

### Current manual process being replaced

```bash
# Current: run locally
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  763104351884.dkr.ecr.us-east-1.amazonaws.com

docker buildx build -t dlc-slurmd:latest -f dlc-slurmd.Dockerfile .

# Tag and push manually
docker tag dlc-slurmd:latest <account>.dkr.ecr.<region>.amazonaws.com/dlc-slurmd:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/dlc-slurmd:latest
```

### Key References

- **Upstream Dockerfile:**
  `ai-on-eks-slurm/ai-on-eks/blueprints/training/slinky-slurm/dlc-slurmd.Dockerfile`
- **Existing file to update:** `dlc-slurmd.Dockerfile`

## Success Criteria

- [ ] TBD

## Notes

Created via feature-capture
