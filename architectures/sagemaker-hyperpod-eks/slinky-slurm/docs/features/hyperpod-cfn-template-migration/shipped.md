---
id: hyperpod-cfn-template-migration
status: shipped
started: 2026-03-06
completed: 2026-03-06
shipped: 2026-03-12
---

# Shipped: HyperPod CloudFormation Template Migration

## Summary

Migrated the CloudFormation deployment path from the legacy
`awsome-distributed-ai` nested stack templates (curled from GitHub
at deploy time) to the official SageMaker HyperPod service team S3-hosted
templates at `aws-sagemaker-hyperpod-cluster-setup-<region>-prod`.

## What Changed

- **Template source**: `--template-body file://main-stack.yaml` replaced
  with `--template-url` pointing to the per-region S3-hosted HyperPod
  template
- **Parameters format**: Flat 17-parameter files replaced with a
  consolidated 40-parameter `params.json` using JSON array strings for
  instance groups (`InstanceGroupSettings1`)
- **Capabilities**: Added `CAPABILITY_AUTO_EXPAND` for nested stack
  transforms
- **Template validation**: New `validate_cfn_template()` function in
  `lib/deploy_helpers.sh` that validates the S3 template and
  cross-checks parameter keys before deployment
- **Idempotent deploys**: `deploy_cfn()` checks stack status to choose
  `create-stack` vs `update-stack`, handles "No updates" gracefully
- **README**: Updated deployment and cleanup instructions

## Files Modified

| File | Change |
|------|--------|
| `params.json` | New 40-param format with instance group JSON arrays |
| `tests/fixtures/params.json` | Test fixture mirror of params.json |
| `deploy.sh` | S3 template URL, `validate_cfn_template` call, idempotent create/update |
| `lib/deploy_helpers.sh` | Added `validate_cfn_template()`, updated `resolve_cfn_params()` |
| `tests/helpers/mock_aws.bash` | Mock for `cloudformation validate-template` |
| `tests/test_deploy.bats` | 8 new tests for template validation |
| `README.md` | Updated CFN deployment and cleanup sections |
| `.opencode/skills/deploy-infrastructure/SKILL.md` | Documented validation step |

## Quality Gates

### Security Audit: APPROVED

- No hardcoded credentials or secrets
- IAM capabilities justified for nested HyperPod template
- All variable expansions double-quoted, no injection vectors
- Template URL uses HTTPS with official SageMaker bucket pattern
- Temp files created via `mktemp`, cleaned up on error paths
- Advisory: consider adding `trap EXIT` cleanup and `--region`
  regex validation (non-blocking)

### QA Validation: APPROVED

- 162/162 bats tests passing (80 deploy + 49 install + 13 setup +
  20 destroy)
- All 5 shell scripts pass `bash -n` syntax check
- Both `params.json` files valid JSON with exactly 40 parameters
- All 18 acceptance criteria in plan.md checked
- `validate_cfn_template` correctly gates `create-stack` in deploy.sh

## Acceptance Criteria

All 6 implementation steps completed with 18/18 acceptance criteria met:

1. g5 params file with InstanceGroupSettings1 JSON arrays
2. p5 params file structurally symmetric with g5
3. README CFN deployment section updated (no curl, S3 template URL,
   CAPABILITY_AUTO_EXPAND)
4. README cleanup section updated
5. g5/p5 parameter key symmetry verified
6. Template validation integrated into deploy.sh
