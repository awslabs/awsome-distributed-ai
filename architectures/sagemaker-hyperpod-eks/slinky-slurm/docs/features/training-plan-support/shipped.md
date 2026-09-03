---
id: training-plan-support
status: shipped
started: 2026-03-12
completed: 2026-03-12
shipped: 2026-03-12
---

# Shipped: Training Plan Support for deploy.sh

## Summary

Added `--training-plan <name>` flag to `deploy.sh` that auto-resolves
a SageMaker Training Plan's ARN and availability zone, overrides the
deployment AZ to match the plan's reserved capacity, and injects the
`TrainingPlanArn` into the accelerated instance group for both
CloudFormation and Terraform deployment paths.

## What Changed

- **resolve_training_plan()**: New function in `lib/deploy_helpers.sh`
  that calls `aws sagemaker describe-training-plan`, extracts ARN and
  AZ, resolves AZ name to AZ ID via EC2 API, and handles error states
  (Failed -> error, Expired -> warning, no capacity -> error)
- **deploy.sh**: New `--training-plan` flag with AZ auto-override
  (logs WARNING when overriding `--az-id`), passes ARN as 6th arg to
  `resolve_cfn_params` and `resolve_tf_vars`
- **resolve_cfn_params()**: Extended with 6th `training_plan_arn` arg;
  injects `TrainingPlanArn` via jq `--arg` into accelerated instance
  group only (general group untouched)
- **resolve_tf_vars()**: Extended with 6th arg; injects
  `training_plan_arn` via awk `-v` into first instance group block
  only when non-empty
- **Mock AWS responses**: 6 SageMaker training plan mocks + 2 AZ
  name-to-ID resolution mocks
- **12 new unit tests**: Covering all error paths, ARN injection/
  omission, and scope isolation

## Files Modified

| File | Change |
|------|--------|
| `lib/deploy_helpers.sh` | Added `resolve_training_plan()`, extended `resolve_cfn_params()` and `resolve_tf_vars()` with 6th arg |
| `deploy.sh` | `--training-plan` flag, AZ override, ARN pass-through |
| `tests/test_deploy.bats` | 12 new training plan tests |
| `tests/helpers/mock_aws.bash` | 8 new mock patterns (SageMaker + EC2 AZ) |
| `AGENTS.md` | Training plan documentation |

## Quality Gates

### Security Audit: APPROVED

- ARN passed to jq via parameterized `--arg` (no string interpolation
  in filters) and to awk via `-v` (no code injection)
- AWS CLI arguments properly double-quoted (no shell injection)
- Failed plans error, expired plans warn, no-capacity plans error
- AZ override explicitly logged with WARNING and both old/new values
- TrainingPlanArn scoped to accelerated group only (verified by tests)
- Advisory: client-side plan name regex validation for defense-in-depth
  (non-blocking, AWS API validates server-side)

### QA Validation: APPROVED

- 162/162 bats tests passing (80 deploy + 49 install + 13 setup +
  20 destroy)
- All 5 shell scripts pass `bash -n` syntax check
- 12 training-plan-specific tests covering resolve, CFN injection,
  TF injection, CLI usage
- 8 mock patterns covering all error paths and AZ resolution targets
- deploy.sh integration verified: flag parsing, resolve call, AZ
  override with WARNING, ARN pass-through to both resolvers
- All 10 plan acceptance criteria checked

## Acceptance Criteria

All 10 implementation steps completed:

1. Added `resolve_training_plan()` to `lib/deploy_helpers.sh`
2. Added `--training-plan` flag to `deploy.sh` with AZ auto-override
3. Extended `resolve_cfn_params()` with 6th arg for training plan ARN
4. Extended `resolve_tf_vars()` with 6th arg for training plan ARN
5. Added 8 mock responses to `tests/helpers/mock_aws.bash`
6. Added 12 tests to `tests/test_deploy.bats`
7. Updated `deploy.sh` usage text
8. Updated AGENTS.md with training plan documentation
9. All 162 bats tests pass
10. Marked complete
