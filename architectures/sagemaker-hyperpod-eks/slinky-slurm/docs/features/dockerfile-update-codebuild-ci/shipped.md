---
id: dockerfile-update-codebuild-ci
status: shipped
started: 2026-03-09
completed: 2026-03-09
shipped: 2026-03-12
---

# Shipped: Dockerfile Update and CodeBuild CI Pipeline

## Summary

Bumped the `dlc-slurmd.Dockerfile` base image from Slinky `25.05.0` to
`25.11.1` and added a CodeBuild CI pipeline (both CloudFormation and
Terraform) for automated container image builds.

## What Changed

- **Dockerfile**: Bumped `ghcr.io/slinkyproject/slurmd` from `25.05.0`
  to `25.11.1-ubuntu24.04`. Multi-stage build copies CUDA, EFA,
  OpenMPI, NCCL, and Python from the AWS DLC PyTorch image.
- **buildspec.yml**: CodeBuild build spec that authenticates to DLC ECR
  (`763104351884`) and project ECR, builds the image, tags, and pushes.
- **codebuild-stack.yaml**: CloudFormation template creating ECR repo
  (lifecycle: keep 10, scan-on-push), IAM role (least-privilege ECR
  push/pull, CloudWatch Logs, S3 source), and CodeBuild project
  (`BUILD_GENERAL1_LARGE`, privileged mode).
- **codebuild.tf**: Terraform equivalent with identical resources, IAM
  policies, and defaults. Conditional ECR creation via
  `create_ecr_repository` variable.
- **setup.sh integration**: Deploys CodeBuild stack if not present,
  uploads build context to S3, triggers `start-build`, and polls for
  completion. Falls back to local build with `--local-build` flag.

## Files Modified

| File | Change |
|------|--------|
| `dlc-slurmd.Dockerfile` | Base image bump 25.05.0 -> 25.11.1 |
| `buildspec.yml` | New CodeBuild build specification |
| `codebuild-stack.yaml` | New CFN template for CodeBuild + ECR + IAM |
| `codebuild.tf` | New Terraform config for CodeBuild + ECR + IAM |
| `setup.sh` | CodeBuild stack deploy, S3 upload, build trigger |

## Quality Gates

### Security Audit: APPROVED

- Base image from trusted sources (AWS DLC ECR `763104351884`, Slinky
  official `ghcr.io/slinkyproject`)
- IAM role follows least-privilege: ECR push scoped to single repo,
  DLC pull scoped to official AWS account, CloudWatch Logs scoped to
  `/aws/codebuild/*`, S3 scoped to exact source path
- ECR credentials piped via `get-login-password`, never stored to disk
- No hardcoded secrets, tokens, or credentials
- ECR scan-on-push enabled, lifecycle policy retains last 10 images
- Privileged mode justified for Docker-in-Docker builds
- Advisory: root container justified for slurmd, SSH
  `StrictHostKeyChecking no` is standard HPC pattern, baked-in SSH
  key overridden by Helm at deploy time (non-blocking)

### QA Validation: APPROVED

- 162/162 bats tests passing (80 deploy + 49 install + 13 setup +
  20 destroy)
- All 5 shell scripts pass `bash -n` syntax check
- Dockerfile validated: correct base image version, multi-stage build,
  apt cleanup, sorted packages, no syntax errors
- buildspec.yml validated: correct YAML structure, ECR login, build/
  tag/push phases
- codebuild-stack.yaml validated: proper CFN structure, 3 resources,
  5 IAM policy statements, conditional ECR
- codebuild.tf validated: equivalent resources, valid HCL, CFN parity
- All 4 plan acceptance criteria checked

## Acceptance Criteria

All 4 implementation steps completed:

1. Bumped `dlc-slurmd.Dockerfile` from `25.05.0` to `25.11.1`
2. Created `buildspec.yml` with DLC ECR auth, build, tag, push phases
3. Created `codebuild-stack.yaml` with ECR + IAM + CodeBuild resources
4. Created `codebuild.tf` with equivalent Terraform configuration
