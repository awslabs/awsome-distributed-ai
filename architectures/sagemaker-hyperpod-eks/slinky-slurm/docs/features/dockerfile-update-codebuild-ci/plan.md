---
id: dockerfile-update-codebuild-ci
status: done
started: 2026-03-09
completed: 2026-03-09
---

# Dockerfile Update and CodeBuild CI Pipeline — Plan

## Steps

- [x] 1. Bump `dlc-slurmd.Dockerfile`: `25.05.0` -> `25.11.1`
- [x] 2. Create `buildspec.yml` — CodeBuild build specification
- [x] 3. Create `codebuild-stack.yaml` — CloudFormation template for
      CodeBuild project + IAM role + ECR repository
- [x] 4. Create `codebuild.tf` — Equivalent Terraform configuration

## What Changed

### Dockerfile

One-line change: bumped the Slinky slurmd base image from `25.05.0` to
`25.11.1` to match the upstream reference.

### CodeBuild Pipeline

Both CloudFormation and Terraform templates provided for the CodeBuild
infrastructure. Each creates:

- **ECR Repository** — `dlc-slurmd` with lifecycle policy (keep last 10)
- **IAM Role** — Permissions for ECR push, DLC ECR pull, CloudWatch Logs,
  S3 source access
- **CodeBuild Project** — `BUILD_GENERAL1_LARGE` with privileged mode
  (Docker builds), S3 source type

The `buildspec.yml` handles:
1. Authenticating to the DLC ECR registry (account `763104351884`)
2. Authenticating to the project ECR registry
3. Building the Docker image from `dlc-slurmd.Dockerfile`
4. Tagging and pushing to ECR

### Integration with setup.sh

`setup.sh` deploys the CodeBuild stack if not present, uploads the build
context to S3, triggers `start-build`, and polls for completion. Falls
back to local build with `--local-build` flag.
