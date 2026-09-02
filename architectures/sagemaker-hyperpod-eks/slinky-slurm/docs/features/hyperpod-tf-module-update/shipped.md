---
id: hyperpod-tf-module-update
status: shipped
started: 2026-03-06
completed: 2026-03-06
shipped: 2026-03-12
---

# Shipped: HyperPod Terraform Module Syntax Update

## Summary

Rewrote `custom.tfvars` from the old Terraform module syntax (map-based
`instance_groups {}`) to the new HyperPod Terraform module syntax
(list-of-objects `instance_groups []`), added `aws_region` and per-group
`availability_zone_id`, and introduced 8 module toggle variables for
observability, task governance, inference/training operators, GuardDuty,
and FSx.

## What Changed

- **instance_groups format**: Map `{}` with group name as key replaced
  with list of objects `[]` with `name` field inside each object
- **availability_zone_id**: Moved from top-level variable to inside each
  instance group object
- **aws_region**: Added as new top-level variable
- **kubernetes_version**: Bumped from `1.32` to `1.34`
- **Module toggles**: 8 new boolean/string variables added for feature
  modules (observability, task governance, training/inference operators,
  GuardDuty cleanup, FSx)
- **resolve_tf_vars()**: Updated to patch the new syntax via sed/awk
  substitutions for region, AZ, instance type, count, and training plan
  ARN at deploy time
- **g5/p5 consolidation**: Single `custom.tfvars` with g5 defaults;
  `deploy.sh --instance-type` overrides at runtime for any instance type

## Files Modified

| File | Change |
|------|--------|
| `custom.tfvars` | Rewritten to new list-of-objects syntax with 8 module toggles |
| `tests/fixtures/custom.tfvars` | Test fixture mirror (byte-identical to production) |
| `lib/deploy_helpers.sh` | `resolve_tf_vars()` updated for new syntax |
| `deploy.sh` | `deploy_tf()` uses copy-resolve-init-plan-apply flow |

## Quality Gates

### Security Audit: APPROVED

- No hardcoded credentials, account IDs, or ARNs in tfvars
- Module toggle defaults follow least-privilege (only observability
  and GuardDuty cleanup enabled by default)
- Terraform execution has plan-before-apply with manual confirmation
- All variable substitutions use quoted expansions and safe awk `-v`
  bindings
- Advisory: add input format validation for `--instance-count` and
  `--az-id`, gitignore copied tfvars in terraform-modules (non-blocking)

### QA Validation: APPROVED

- 162/162 bats tests passing (80 deploy + 49 install + 13 setup +
  20 destroy)
- All 5 shell scripts pass `bash -n` syntax check
- Both `custom.tfvars` files use correct new syntax with all required
  variables (aws_region, per-group availability_zone_id, 8 toggles)
- Production and fixture files are byte-identical
- All 6 plan acceptance criteria checked
- resolve_tf_vars tests (50-60) validate region, AZ, instance type,
  count overrides, training plan ARN injection, and .bak cleanup

## Acceptance Criteria

All 6 implementation steps completed:

1. Analyzed old vs new tfvars syntax differences
2. Rewrote g5 custom.tfvars to new module syntax
3. Rewrote p5 custom.tfvars (consolidated into single file with g5
   defaults)
4. Verified g5/p5 symmetry (intentional differences: instance type
   and count only)
5. README Terraform section reviewed (no changes needed)
6. Marked complete
