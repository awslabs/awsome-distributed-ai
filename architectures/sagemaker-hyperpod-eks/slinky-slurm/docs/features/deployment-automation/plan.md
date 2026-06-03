---
id: deployment-automation
status: done
started: 2026-03-06
completed: 2026-03-09
---

# Deployment Automation Scripts — Plan

> **NOTE:** The `--node-type g5|p5` CLI described below was subsequently
> refactored to `--instance-type <ml.X.Y>` + `--instance-count <N>` with
> EC2 API auto-discovery. See `AGENTS.md` and the current script `--help`
> for the updated interface.

## Steps

- [x] 1. Create `deploy.sh` (Phase 0: Infrastructure deployment)
- [x] 1a. Extract testable functions into `lib/deploy_helpers.sh`
- [x] 1b. Unit test `deploy.sh` with bats-core (34 tests passing)
- [x] 2. Create `setup.sh` (Phase 1: Docker build + SSH keys + values template)
- [x] 3. Create `install.sh` (Phase 2: Helm installs + k8s Day-2 config)
- [x] 4. Create `destroy.sh` (Phase 3: Reverse teardown)
- [x] 5. Update README.md to reference all scripts
- [x] 6. Mark complete

## Completed: deploy.sh

`deploy.sh` handles infrastructure deployment via CloudFormation or Terraform
with automatic AZ resolution.

### Interface

```bash
./deploy.sh --node-type <g5|p5> --infra <cfn|tf> [OPTIONS]

Options:
  --region <region>       AWS region (default: us-west-2)
  --az-id <az-id>         AZ for instance groups + FSx (default: usw2-az2)
  --stack-name <name>     CFN stack name (default: hp-eks-slinky-stack)
  --help                  Show usage
```

### Features

- Resolves up to 5 non-opt-in AZs via `aws ec2 describe-availability-zones`
- Validates the specified `--az-id` exists in the resolved AZ list
- CFN path: `jq` substitution of `AvailabilityZoneIds`,
  `FsxAvailabilityZoneId`, `TargetAvailabilityZoneId`, and instance
  type/count in `InstanceGroupSettings1`
- TF path: copies tfvars, overrides `aws_region`, `availability_zone_id`,
  and instance type/count via `sed`, runs `terraform init/plan/apply`
- Extracts stack outputs to `env_vars.sh`

## Completed: setup.sh

`setup.sh` handles container image builds, SSH key generation, and Helm
values template substitution.

### Interface

```bash
./setup.sh --node-type <g5|p5> --infra <cfn|tf> [OPTIONS]

Required:
  --node-type <g5|p5>       Instance profile (sets GPU/EFA/GRES/replicas)
  --infra <cfn|tf>          Infrastructure method for CodeBuild stack

Optional:
  --repo-name <name>        ECR repository name (default: dlc-slurmd)
  --tag <tag>               Image tag (default: 25.11.1-ubuntu24.04)
  --region <region>         AWS region (default: AWS CLI configured or us-west-2)
  --local-build             Build image locally instead of CodeBuild
  --skip-build              Skip image build (use existing image in ECR)
  --help                    Show this help message
```

### Features

- Deploys CodeBuild infrastructure (CFN or TF) for building container images
- Starts CodeBuild build and waits for completion (default), or builds
  locally with `--local-build`, or skips with `--skip-build`
- Generates SSH ed25519 key pair for Slurm login access
- Resolves `slurm-values.yaml` from `slurm-values.yaml.template` using
  `resolve_helm_profile` from `lib/deploy_helpers.sh` (substitutes 10
  shell template variables based on node type)

## Completed: install.sh

`install.sh` orchestrates Helm installations and Kubernetes Day-2
configuration.

### Interface

```bash
./install.sh [OPTIONS]

Optional:
  --skip-setup              Use previously generated slurm-values.yaml
  --region <region>         AWS region (default: AWS CLI configured or us-west-2)
  --help                    Show this help message

Options passed through to setup.sh:
  --node-type <g5|p5>       Instance profile
  --infra <cfn|tf>          Infrastructure method for CodeBuild stack
  --repo-name <name>        ECR repository name
  --tag <tag>               Image tag
  --local-build             Build image locally instead of CodeBuild
  --skip-build              Skip image build (use existing image in ECR)
```

### Features

- Calls `setup.sh` first (unless `--skip-setup`)
- Installs MariaDB operator (Helm chart v25.10.4) and applies MariaDB CR
- Installs Slurm operator (OCI chart v1.0.1) with default values
- Installs Slurm cluster (OCI chart v1.0.1) with generated `slurm-values.yaml`
- Configures login service NLB via `slurm-login-service-patch.yaml`

## Completed: destroy.sh

`destroy.sh` tears down all resources in reverse order.

### Interface

```bash
./destroy.sh --infra <cfn|tf> [OPTIONS]

Required:
  --infra <cfn|tf>          Infrastructure method used for deployment

Optional:
  --region <region>         AWS region (default: AWS CLI configured or us-west-2)
  --stack-name <name>       HyperPod CFN stack name (default: hp-eks-slinky-stack)
  --help                    Show this help message
```

### Features

- Prompts for confirmation before proceeding
- Uninstalls Slurm cluster, Slurm operator, MariaDB in reverse order
- Deletes CodeBuild stack (CFN or TF)
- Deletes HyperPod infrastructure stack (CFN or TF)
- Cleans up local generated files (`slurm-values.yaml`,
  `slurm-login-service-patch.yaml`, `env_vars.sh`)

## Also completed in this feature

- Consolidated `g5/g5-params.json` and `p5/p5-params.json` into a single
  `params.json` at the project root (defaults to g5 settings; `deploy.sh`
  overrides instance type/count for p5 at deploy time)
- Consolidated `g5/g5-custom.tfvars` and `p5/p5-custom.tfvars` into a
  single `custom.tfvars` at the project root (same override pattern)
- Consolidated `g5/g5-values.yaml` and `p5/p5-values.yaml` into a single
  `slurm-values.yaml.template` with shell template variables
- Updated `params.json` to include all 40 parameters (23 previously missing)
  matching the reference configuration
- Default `AvailabilityZoneIds` set to `usw2-az1,usw2-az2,usw2-az3,usw2-az4`
- Bumped `KubernetesVersion` to `1.34` in both params.json and custom.tfvars
- Enabled all features: observability, logging, training operator, inference
  addon, task governance, GPU operator
- Updated README.md to reference `deploy.sh` as primary deployment method
  with manual commands preserved as collapsible fallback
- Created `buildspec.yml`, `codebuild-stack.yaml`, and `codebuild.tf` for
  CodeBuild-based container image builds
- Created `mariadb.yaml` MariaDB CR manifest for Slurm accounting
- Created `slurm-login-service-patch.yaml.template` for NLB configuration

## Completed: Unit Testing (bats-core)

45 unit tests covering all 7 extracted helper functions and script argument
parsing. Test infrastructure includes:

- `tests/test_deploy.bats` — Test file (45 tests across 9 categories)
- `tests/helpers/setup.bash` — Common setup/teardown with library guard
- `tests/helpers/mock_aws.bash` — AWS CLI mock with canned responses
- `tests/fixtures/params.json` — Independent fixture copy
- `tests/fixtures/custom.tfvars` — Independent fixture copy
- `tests/fixtures/slurm-values.yaml.template` — Helm values fixture
- `tests/install_bats_libs.sh` — Installs bats-assert + bats-support
- `.gitignore` updated to exclude `tests/bats/`

Bug fixed: `resolve_tf_vars` used GNU sed `0,/pattern/` syntax (first
occurrence only) which silently fails on macOS (BSD sed). Replaced with
portable awk approach.
