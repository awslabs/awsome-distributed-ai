# AGENTS.md — slinky-slurm

> Guidelines for AI coding agents operating in the `slinky-slurm` subdirectory of
> `awsome-distributed-ai`. This is an infrastructure-as-code project for deploying
> Slurm on Amazon SageMaker HyperPod EKS via the Slinky Project (SchedMD).

## Project Overview

This project contains Helm values, Kubernetes manifests, Dockerfiles, Terraform/CloudFormation
parameters, Slurm batch scripts, deployment automation scripts, and documentation. There is
**no application source code** (no Python/Go/TS modules). Sbatch scripts are organized by
workload type under `sbatch/` (e.g., `sbatch/fsdp/`) with hardware-profile prefixes (g5, p5).
Infrastructure config files (`params.json`, `custom.tfvars`) are consolidated at the
project root with `ml.g5.8xlarge` defaults; `deploy.sh` overrides values for any user-specified
instance type via `--instance-type` and `--instance-count`.

Key automation scripts:
- **`deploy.sh`** — Infrastructure deployment via CloudFormation or Terraform;
  supports `--training-plan <name>` for reserved capacity (auto-resolves ARN and AZ);
  CFN path is idempotent (create or update based on stack status)
- **`setup.sh`** — Container image build (CodeBuild/local), SSH keys, Helm values generation
- **`install.sh`** — cert-manager, AWS LB Controller (Pod Identity), subnet tagging,
  FSx PVC, MariaDB, Slurm operator, Slurm cluster Helm installs, NLB config;
  supports `--skip-cert-manager`, `--skip-lb-controller`, `--skip-ebs-csi` for
  pre-installed components and `--cluster-name`/`--vpc-id` for bring-your-own-cluster;
  uses `helm upgrade --install` for idempotent operations
- **`destroy.sh`** — Reverse teardown of all deployed resources (including LB Controller
  Pod Identity + IAM, cert-manager, and FSx PVC); warns when `EKS_CLUSTER_NAME` is
  unset and Pod Identity / addon cleanup is skipped; warns when `AWS_ACCOUNT_ID` is
  unavailable and IAM cleanup is skipped; preserves ECR repository and S3 build context
  bucket with manual cleanup commands printed at end of run
- **`lib/deploy_helpers.sh`** — Extracted testable functions sourced by `deploy.sh` and `setup.sh`
- **`params.json`** — CloudFormation parameters (40 params, g5 defaults)
- **`custom.tfvars`** — Terraform variables (g5 defaults)
- **`slurm-values.yaml.template`** — Consolidated Helm values with shell template variables

## Build / Validate / Test Commands

There is no traditional build system (no Makefile, package.json, or pyproject.toml).

### Docker Image Build

```bash
# Authenticate to ECR (required for DLC base image)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# Build the Slurm compute node image
docker buildx build -t dlc-slurmd:latest -f dlc-slurmd.Dockerfile .
```

### Helm Chart Validation

```bash
# Lint a values file against the upstream chart
helm lint <chart-path> -f slurm-values.yaml
helm template <release-name> <chart-path> -f slurm-values.yaml
```

### YAML Validation

```bash
# Validate Kubernetes manifests
kubectl apply --dry-run=client -f lustre-pvc-slurm.yaml
kubectl apply --dry-run=client -f lustre-storageclass.yaml
```

### Markdown Linting

The repo root has `.markdownlint.jsonc` with these rules:
- MD041 (first-line heading): disabled
- MD013 (line length): 100 chars, code blocks excluded
- MD033 (inline HTML): disabled

```bash
# From repo root
npx markdownlint-cli2 "architectures/sagemaker-hyperpod-eks/slinky-slurm/**/*.md"
```

### CI Static Analysis (PR workflow)

The GitHub Actions workflow `pr-review-and-slurm-test.yml` runs on PRs to `main`:
- `pylint` and `flake8` on any `.py` files
- `bash -n` syntax checking on `.sh` files
- Secrets scanning via grep patterns

### Tests

Bash scripts are tested using [bats-core](https://github.com/bats-core/bats-core) with
bats-assert and bats-support helper libraries. The repo root `conftest.py` provides legacy
pytest fixtures for Docker-based tests but should not be used as a reference for new tests.

```bash
# One-time setup: install bats-core
brew install bats-core            # macOS
# OR: sudo apt-get install -y bats  # Debian/Ubuntu
# OR: npm install -g bats           # cross-platform

# One-time setup: install bats helper libraries
bash tests/install_bats_libs.sh

# Run all bats tests
bats tests/

# Run a specific test file
bats tests/test_deploy.bats

# Verbose output (show test names)
bats --verbose-run tests/test_deploy.bats
```

Test structure:
- `tests/test_deploy.bats` — 72 unit tests for `deploy.sh` and `lib/deploy_helpers.sh`
- `tests/test_setup.bats` — 13 unit tests for `setup.sh` argument parsing, profile
  resolution, and template substitution
- `tests/test_install.bats` — 49 unit tests for `install.sh` argument parsing, version
  constants, install order, skip flags, existing cluster support, `helm upgrade --install`,
  IAM idempotency, `env_vars.sh` dependency validation, and EBS CSI/gp3 phases
- `tests/test_destroy.bats` — 20 unit tests for `destroy.sh` argument parsing, teardown
  order, IAM cleanup, EKS_CLUSTER_NAME warnings, and CodeBuild TF destroy
- `tests/fixtures/` — Independent copies of `params.json`, `custom.tfvars`, and
  `slurm-values.yaml.template` for test isolation
- `tests/helpers/setup.bash` — Common bats setup/teardown (loads helpers, creates temp dir,
  sources `lib/deploy_helpers.sh`, activates AWS CLI mock)
- `tests/helpers/mock_aws.bash` — AWS CLI mock function that intercepts `aws` calls with
  canned responses
- `tests/install_bats_libs.sh` — Clones bats-assert and bats-support into `tests/bats/`
  (gitignored, not committed)

When adding new bash scripts, follow this pattern:
1. Extract testable functions into `lib/<script>_helpers.sh`
2. Source the helpers file from the main script
3. Add fixture data to `tests/fixtures/`
4. Write tests in `tests/test_<script>.bats`
5. Load `helpers/setup` at the top of each `.bats` file

## Code Style Guidelines

### EditorConfig (repo-wide)

Defined in `/.editorconfig`:
- **Line endings:** LF (Unix)
- **Charset:** UTF-8
- **Trailing whitespace:** trimmed
- **Final newline:** required
- **YAML/JSON indent:** 2 spaces
- **Makefile indent:** tabs

### File Naming

- Lowercase with hyphens: `lustre-pvc-slurm.yaml`, `openzfs-storageclass.yaml`
- Instance-type prefix for profile-specific files: `g5-values.yaml`, `p5-values.yaml`
- Dockerfile uses PascalCase extension: `dlc-slurmd.Dockerfile`
- Model names in sbatch files may use underscores: `g5-llama2_7b-training.sbatch`

### YAML Conventions (Helm Values)

- **2-space indentation** throughout (Kubernetes/Helm standard)
- **helm-docs type annotations** above every configurable value:
  ```yaml
  # -- (string)
  # Set the image pull policy.
  imagePullPolicy: IfNotPresent
  ```
- **WARNING:** prefix for destructive/important caveats
- **NOTE:** prefix for informational notes
- **Ref:** prefix linking to upstream documentation
- **`@ignored`** tag to exclude values from helm-docs
- Empty defaults: `{}` for maps, `[]` for lists, `""` for strings
- Use YAML anchors (`&anchorName`) and aliases (`*anchorName`) for repeated config blocks
  (e.g., `commonAffinity`)
- Commented-out examples for optional/disabled values:
  ```yaml
  resources: {}
    # requests:
    #   cpu: 1
    #   memory: 1Gi
  ```

### Kubernetes Manifest YAML

- Resource names: lowercase with hyphens (`fsx-claim`, `openzfs-sc`)
- Standard key ordering: `apiVersion`, `kind`, `metadata`, `spec`
- Quoted string values for StorageClass parameters: `"0"`, `"true"`, `"LZ4"`

### JSON Conventions

- **2-space indentation**
- Standard JSON (no trailing commas)
- CloudFormation parameter keys use PascalCase per AWS convention

### Terraform Variables (.tfvars)

- **snake_case** for all variable names (standard Terraform convention)
- String values quoted, numbers and booleans unquoted
- One assignment per line

### Shell / Slurm Batch Scripts (.sbatch)

- Shebang: `#!/bin/bash`
- Error handling: `set -ex` at the top of every sbatch script
  - For automation scripts (`deploy.sh`, `setup.sh`, `install.sh`, `destroy.sh`),
    prefer `set -euo pipefail`
  - For prolog/epilog scripts, prefer `set -euo pipefail`
- Environment variables: `SCREAMING_SNAKE_CASE`
- Group variables by category with decorative section headers:
  ```bash
  ###########################
  ###### User Variables #####
  ###########################
  ```
- Quote paths and string values in exports: `export CUDA_HOME="/usr/local/cuda"`
- Use `${VAR}` brace syntax in compound expressions: `${EFA_PATH}/lib`
- Bash arrays with `declare -a` and 4-space indentation inside:
  ```bash
  declare -a TORCHRUN_ARGS=(
      --nproc_per_node=$GPUS_PER_NODE
      --nnodes=$SLURM_JOB_NUM_NODES
  )
  ```
- Expand arrays with proper quoting: `"${TORCHRUN_ARGS[@]}"`
- Right-align inline comments for visual consistency

### Dockerfile Conventions

- Multi-stage builds: name stages with `AS <name>` for `COPY --from=<name>`
- Use `ARG` for parameterized versions
- Group `ENV` declarations with `\` continuation by functional area
- APT pattern: `apt-get update && apt-get install -y --no-install-recommends ... && rm -rf /var/lib/apt/lists/* && apt-get clean` in a single `RUN` layer
- Sort package lists alphabetically, one per line
- Section comments above each logical block of `COPY`/`RUN` instructions
- Minimize layers: chain related commands with `&&`

### License Headers

- Helm values (SchedMD origin): SPDX format
  ```yaml
  # SPDX-FileCopyrightText: Copyright (C) SchedMD LLC.
  # SPDX-License-Identifier: Apache-2.0
  ```
- Sbatch scripts (Amazon origin):
  ```bash
  # Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  # SPDX-License-Identifier: MIT-0
  ```

### Markdown Documentation

- Use `###` for primary section headings
- Max line length: 100 characters (code blocks excluded)
- Inline HTML (`<u>`, etc.) is permitted (MD033 disabled)
- Backtick-wrap commands, file names, and environment variables
- Tables for structured component descriptions
- Link format: `[text](url)`

## Keeping g5 and p5 Sbatch Files in Sync

Sbatch scripts are organized under `sbatch/` by workload type (e.g., `sbatch/fsdp/`), with
hardware-profile prefixes (`g5-`, `p5-`). When modifying one profile's sbatch, check whether
the same change should be applied to the other. The two files within each workload directory
are structurally aligned (same section ordering, same environment variable blocks) so they
diff cleanly. The `params.json` and `custom.tfvars` files have been consolidated to the
project root (g5 defaults); `deploy.sh` overrides values for any instance type at deploy time.
Helm values files have been consolidated into `slurm-values.yaml.template` at the project root.

Known intentional differences between g5 and p5 sbatch scripts (GPU/EFA/GRES values are
auto-discovered from the instance type via `aws ec2 describe-instance-types`):

- Instance types (`ml.g5.8xlarge` vs `ml.p5.48xlarge`)
- GPU counts (1 vs 8) and EFA interface counts (1 vs 32)
- Compute node replicas (4 vs 2)
- GRES configuration (`gpu:a10g:1` vs `gpu:h100:8`)
- SBATCH directives (`--ntasks-per-node`, `--cpus-per-task`)
- EFA/NCCL environment variables specific to the network topology

All other structure and configuration patterns should remain consistent between the two profiles.
