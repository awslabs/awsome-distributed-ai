---
id: deployment-automation
status: shipped
started: 2026-03-06
completed: 2026-03-09
shipped: 2026-03-12
---

# Shipped: Deployment Automation Scripts

## Summary

Created the 4 core deployment automation scripts (`deploy.sh`,
`setup.sh`, `install.sh`, `destroy.sh`), the extracted helper library
(`lib/deploy_helpers.sh`), and a comprehensive bats-core test suite
with 162 unit tests. This replaced the previous manual multi-step
deployment process with a scripted 3-phase workflow.

## What Changed

- **deploy.sh** (Phase 0): Infrastructure deployment via CloudFormation
  or Terraform with `--instance-type` auto-discovery via EC2 API, AZ
  resolution, idempotent create/update, and stack output extraction to
  `env_vars.sh`
- **setup.sh** (Phase 1): Container image builds via CodeBuild (or
  `--local-build`), SSH ed25519 key generation, and Helm values
  template substitution using `resolve_helm_profile()` for any GPU
  instance type
- **install.sh** (Phase 2): Orchestrated Helm installs for cert-manager,
  AWS LB Controller (Pod Identity + IAM), EBS CSI driver, FSx PVC,
  MariaDB operator + CR, Slurm operator, Slurm cluster, and NLB
  configuration. Supports `--skip-cert-manager`, `--skip-lb-controller`,
  `--skip-ebs-csi` for pre-installed components and `--cluster-name`/
  `--vpc-id` for bring-your-own-cluster
- **destroy.sh** (Phase 3): Reverse teardown with confirmation prompts,
  IAM role/policy cleanup, Pod Identity disassociation, Helm uninstalls,
  and local file cleanup
- **lib/deploy_helpers.sh**: 7 extracted testable functions
  (`resolve_instance_profile`, `resolve_helm_profile`,
  `resolve_training_plan`, `check_command`, `validate_az_id`,
  `resolve_cfn_params`, `validate_cfn_template`, `resolve_tf_vars`)
- **Test infrastructure**: bats-core suite with mock AWS CLI, fixture
  files, and 162 unit tests across 4 test files
- **Consolidation**: Merged per-profile config files (g5/p5 params,
  tfvars, values) into single root-level files with runtime overrides

## Files Modified

| File | Change |
|------|--------|
| `deploy.sh` | New — infrastructure deployment (CFN/TF) |
| `setup.sh` | New — image build, SSH keys, Helm values |
| `install.sh` | New — Helm installs and K8s Day-2 config |
| `destroy.sh` | New — reverse teardown |
| `lib/deploy_helpers.sh` | New — extracted helper functions |
| `params.json` | Consolidated from g5/p5, expanded to 40 params |
| `custom.tfvars` | Consolidated from g5/p5 with module toggles |
| `tests/test_deploy.bats` | New — 80 tests |
| `tests/test_setup.bats` | New — 13 tests |
| `tests/test_install.bats` | New — 49 tests |
| `tests/test_destroy.bats` | New — 20 tests |
| `tests/helpers/setup.bash` | New — common test setup/teardown |
| `tests/helpers/mock_aws.bash` | New — AWS CLI mock |
| `tests/fixtures/` | New — params.json, custom.tfvars, template |
| `tests/install_bats_libs.sh` | New — bats helper installer |
| `README.md` | Updated to reference automation scripts |

## Quality Gates

### Security Audit: APPROVED

- `set -euo pipefail` on all 4 scripts; no `eval`/`exec`
- AWS credentials never stored to disk; ECR creds piped correctly
- IAM roles properly scoped with idempotency checks (get before
  create, detach before delete)
- NLB access restricted to deployer's IP via `/32` CIDR
- SSH keys use ed25519, generated idempotently, private key local-only
- Teardown has confirmation prompts and correct reverse ordering
- `env_vars.sh` gitignored; generated files cleaned up by destroy.sh
- Advisory: add `trap` for temp file cleanup, use bash arrays for
  SETUP_ARGS, add IP format validation after checkip call, chmod 600
  on env_vars.sh (non-blocking)

### QA Validation: APPROVED

- 162/162 bats tests passing (80 deploy + 49 install + 13 setup +
  20 destroy)
- All 5 shell scripts pass `bash -n` syntax check
- All 4 scripts follow consistent patterns: `set -euo pipefail`,
  `usage()`, `while [[ $# -gt 0 ]]`, `echo "Error: ..."` format
- Test infrastructure complete: helpers, mock, 3 fixtures, installer
- Test coverage spans all 7 helper functions, all CLI arg paths,
  version constants, install ordering, skip flags, bring-your-own-
  cluster, helm upgrade --install idempotency, IAM lifecycle, and
  teardown ordering
- All 8 plan acceptance criteria checked (steps 1-6 + 1a, 1b)

## Acceptance Criteria

All 8 implementation steps completed:

1. Created `deploy.sh` (infrastructure deployment)
1a. Extracted testable functions into `lib/deploy_helpers.sh`
1b. Unit tested with bats-core (162 tests)
2. Created `setup.sh` (image build + SSH keys + Helm values)
3. Created `install.sh` (Helm installs + K8s Day-2)
4. Created `destroy.sh` (reverse teardown)
5. Updated README.md to reference all scripts
6. Marked complete
