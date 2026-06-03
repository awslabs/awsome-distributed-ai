---
id: agent-deployment-skills
name: Agent Deployment Skills & Validation Tests
type: Feature
priority: P1
effort: Large
impact: High
created: 2026-03-09
---

# Agent Deployment Skills & Validation Tests

## Problem Statement

The slinky-slurm project has four deployment scripts (`deploy.sh`, `setup.sh`,
`install.sh`, `destroy.sh`) and a testable helpers library
(`lib/deploy_helpers.sh`), but no OpenCode agent skills that teach future agents
how to orchestrate the full deployment workflow, perform preflight checks, or
validate that a deployment is healthy. The existing `bash-testing` skill covers
unit test patterns but nothing guides agents through the actual deployment
lifecycle on a HyperPod EKS cluster.

Additionally, the current test suite (45 bats tests in `test_deploy.bats`) has
strong coverage of `deploy.sh` and `lib/deploy_helpers.sh` but only basic
`--help` / missing-arg tests for `setup.sh`, `install.sh`, and `destroy.sh`.

## Proposed Solution

### 1. Migrate Existing Skill to Official SKILL.md Format

The existing `.opencode/skills/bash-testing.md` uses a flat file format. Migrate
it to the official OpenCode skills structure:

```
.opencode/skills/bash-testing.md  -->  .opencode/skills/bash-testing/SKILL.md
```

Add YAML frontmatter with `name` and `description` fields per the OpenCode
skills specification.

### 2. Create 5 New Agent Skills

Each skill follows the `.opencode/skills/<name>/SKILL.md` format with YAML
frontmatter and structured markdown body.

| # | Skill Name | Description | Script Referenced |
|---|-----------|-------------|-------------------|
| 1 | `deployment-preflight` | Validate all prerequisites before deployment | All scripts |
| 2 | `deploy-infrastructure` | Deploy HyperPod EKS infra via CFN or TF | `deploy.sh` |
| 3 | `build-slurm-image` | Build container image and generate Helm values | `setup.sh` |
| 4 | `deploy-slurm-cluster` | Install MariaDB, Slurm operator, and cluster | `install.sh` |
| 5 | `validate-deployment` | Post-deployment health checks and validation | N/A (kubectl/ssh) |

#### Skill 1: `deployment-preflight`

Covers prerequisite validation before any deployment step:

- **CLI tools**: `aws`, `kubectl`, `helm`, `jq`, `docker` (or CodeBuild),
  `terraform` (if TF path)
- **AWS credentials**: `aws sts get-caller-identity` succeeds
- **Environment variables**: Required vars are set (`AWS_REGION`,
  `EKS_CLUSTER_NAME`, etc.) depending on which script will be run
- **kubectl context**: Points to the correct HyperPod EKS cluster
- **AZ validation**: Selected AZ ID is valid for the region
- **ECR image exists** (when using `--skip-build`)
- Common failure modes and fixes

#### Skill 2: `deploy-infrastructure`

Guides agents through `deploy.sh`:

- Choosing between `--infra cfn` and `--infra tf`
- Selecting `--node-type g5` or `--node-type p5` and what each implies
- Specifying `--region` and `--az-id`
- What happens during the 20-30 minute deployment
- Sourcing `env_vars.sh` and updating kubeconfig after completion
- Verification: stack outputs, kubectl cluster-info
- CFN-specific steps and TF-specific steps

#### Skill 3: `build-slurm-image`

Guides agents through `setup.sh` with all three image build paths:

- **CodeBuild (default)**: CFN/TF stack deployment for CodeBuild project,
  S3 bucket creation, build context upload, build trigger, polling for
  completion, ECR image verification
- **Local Docker (`--local-build`)**: ECR DLC authentication, `docker buildx`
  on macOS vs `docker build` on Linux, ECR repo creation, tag and push
- **Skip build (`--skip-build`)**: Verifying image already exists in ECR
- SSH key generation (`~/.ssh/id_ed25519_slurm`)
- Helm values rendering from `slurm-values.yaml.template`
- g5 vs p5 profile differences (`resolve_helm_profile`)
- Verification: ECR image exists, `slurm-values.yaml` generated with no
  unresolved template variables

#### Skill 4: `deploy-slurm-cluster`

Guides agents through `install.sh`:

- Running with `--skip-setup` when `slurm-values.yaml` already exists
- MariaDB operator + instance deployment and readiness wait
- Slurm operator OCI chart installation (CRD cleanup for upgrades)
- Slurm cluster Helm installation with generated values
- NLB configuration: public IP detection, service patch rendering and apply
- NLB endpoint readiness wait
- Verification: all pods running, SSH connection command
- The full `install.sh` -> `setup.sh` call chain when not skipping setup

#### Skill 5: `validate-deployment`

Post-deployment health checks an agent should perform:

- **Pod health**: All pods in `slurm`, `slinky`, `mariadb` namespaces are
  Running/Ready
- **Slurm node registration**: `sinfo` shows expected number of nodes in
  `idle` state
- **Login node SSH**: Connect via NLB endpoint using generated SSH key
- **Test job**: Submit a g5 or p5 sbatch job and verify completion
- **Common failures**: Pods in CrashLoopBackOff, nodes in `drain` state,
  NLB not resolving, SSH connection refused
- **Logs to check**: `kubectl logs` for slurm-controller, slurmd pods

### 3. New Bats Test Files

Extend test coverage to the three under-tested scripts. All tests follow the
existing patterns defined in the `bash-testing` skill and AGENTS.md conventions.

#### `tests/test_setup.bats` (~10-12 tests)

| Category | Tests |
|----------|-------|
| Argument parsing | `--help` exits 0, missing `--node-type`, missing `--infra`, `--skip-build` accepted, `--local-build` accepted, unknown option rejected |
| `resolve_helm_profile` integration | g5 sets all 7 template variables, p5 sets all overrides, invalid type fails |
| Template substitution | g5 produces valid YAML with no unresolved `${vars}`, p5 has correct GPU/EFA counts |

#### `tests/test_install.bats` (~8-10 tests)

| Category | Tests |
|----------|-------|
| Argument parsing | `--help` exits 0, missing `--node-type`, missing `--infra`, `--skip-setup` accepted, unknown option rejected |
| Pass-through flags | `--local-build`, `--skip-build`, `--repo-name`, `--tag` are accepted |
| Flag validation | `--skip-setup` without existing `slurm-values.yaml` behavior |

#### `tests/test_destroy.bats` (~6-8 tests)

| Category | Tests |
|----------|-------|
| Argument parsing | `--help` exits 0, missing `--infra`, unknown option rejected |
| Optional flags | `--region`, `--stack-name` accepted |
| Non-interactive abort | Aborts when confirmation prompt cannot be answered |

### 4. Deliverables Summary

```
# Migrated skill
.opencode/skills/bash-testing/SKILL.md        (was .opencode/skills/bash-testing.md)

# New skills
.opencode/skills/deployment-preflight/SKILL.md
.opencode/skills/deploy-infrastructure/SKILL.md
.opencode/skills/build-slurm-image/SKILL.md
.opencode/skills/deploy-slurm-cluster/SKILL.md
.opencode/skills/validate-deployment/SKILL.md

# New tests
tests/test_setup.bats
tests/test_install.bats
tests/test_destroy.bats

# This feature doc
docs/features/agent-deployment-skills/idea.md
```

### 5. Skill Structure Convention

Each SKILL.md follows a consistent structure:

```markdown
---
name: <kebab-case-name>
description: <1-2 sentence description for agent discovery>
---

# <Skill Title>

## Overview
What this skill does and when an agent should use it.

## Prerequisites
What must be true before using this skill.

## Steps
Ordered instructions referencing the project scripts with exact flags.

## Verification
How to confirm each step succeeded.

## Troubleshooting
Common failure modes and fixes.

## References
Links to the relevant scripts and config files.
```

### 6. Implementation Order

1. Create `docs/features/agent-deployment-skills/idea.md` (this document)
2. Migrate `bash-testing.md` to `bash-testing/SKILL.md`
3. Create `deployment-preflight/SKILL.md`
4. Create `deploy-infrastructure/SKILL.md`
5. Create `build-slurm-image/SKILL.md`
6. Create `deploy-slurm-cluster/SKILL.md`
7. Create `validate-deployment/SKILL.md`
8. Create `tests/test_setup.bats`
9. Create `tests/test_install.bats`
10. Create `tests/test_destroy.bats`
11. Run `bats tests/` to verify all tests pass

## Success Criteria

- [ ] Existing `bash-testing` skill migrated to `SKILL.md` format with frontmatter
- [ ] All 5 new skills discoverable via OpenCode `skill` tool
- [ ] Skills contain enough detail for an agent to deploy end-to-end without
      prior knowledge of the project
- [ ] CodeBuild image build path fully documented in `build-slurm-image` skill
- [ ] New bats tests pass (`bats tests/`)
- [ ] Existing 45 tests in `test_deploy.bats` still pass
- [ ] Test coverage extended to `setup.sh`, `install.sh`, and `destroy.sh`
      argument parsing

## Notes

Created via feature-capture
