---
id: training-plan-support
status: done
started: 2026-03-12
completed: 2026-03-12
---

# Training Plan Support for deploy.sh -- Plan

## Steps

- [x] 1. Add `resolve_training_plan()` to `lib/deploy_helpers.sh`
- [x] 2. Add `--training-plan` flag to `deploy.sh` with AZ auto-override
- [x] 3. Extend `resolve_cfn_params()` -- 6th arg for training plan ARN
- [x] 4. Extend `resolve_tf_vars()` -- 6th arg for training plan ARN
- [x] 5. Add mock responses to `tests/helpers/mock_aws.bash`
- [x] 6. Add 12 tests to `tests/test_deploy.bats`
- [x] 7. Update `deploy.sh` usage text
- [x] 8. Update AGENTS.md with training plan documentation
- [x] 9. Run `bats tests/` -- all 120 tests pass (68 deploy + 13 setup + 21 install + 18 destroy)
- [x] 10. Mark complete

> **NOTE:** End-to-end validation with a real training plan is deferred.
> All logic paths are covered by 12 unit tests with mocked AWS responses.
> The upstream CFN Lambda passthrough and TF `training_plan_arn = optional(string)`
> field have been traced and verified to support `TrainingPlanArn` without changes.

## Implementation Details

### 1. `lib/deploy_helpers.sh` -- `resolve_training_plan()`

New function added between `resolve_helm_profile` and `check_command`:

```
resolve_training_plan <plan_name> <region>
```

**Steps:**
1. Call `aws sagemaker describe-training-plan --training-plan-name <name>`
2. Extract `TrainingPlanArn` from response
3. Check `Status` -- error on `Failed`, warn on `Expired`
4. Extract `ReservedCapacitySummaries[0].AvailabilityZone` (AZ name)
5. Resolve AZ name -> AZ ID via `aws ec2 describe-availability-zones`
6. Set `TRAINING_PLAN_ARN` and `TRAINING_PLAN_AZ_ID`

**Error cases:**
- Plan doesn't exist -> error + return 1
- Status is `Failed` -> error + return 1
- Status is `Expired` -> warning (continues, user may know what they're doing)
- No `ReservedCapacitySummaries` -> error (no reserved capacity)
- AZ resolution fails -> error

### 2. `deploy.sh` -- Flag and AZ override

**New default:** `TRAINING_PLAN=""`

**New arg:** `--training-plan <name>` in case statement

**Integration point** (after `resolve_instance_profile`, before AZ
resolution):

```bash
TRAINING_PLAN_ARN=""
if [[ -n "${TRAINING_PLAN}" ]]; then
    resolve_training_plan "${TRAINING_PLAN}" "${AWS_REGION}"
    if [[ "${TRAINING_PLAN_AZ_ID}" != "${AZ_ID}" ]]; then
        echo "WARNING: Overriding --az-id from '${AZ_ID}' to"
        echo "  '${TRAINING_PLAN_AZ_ID}' to match training plan."
        AZ_ID="${TRAINING_PLAN_AZ_ID}"
    fi
fi
```

**Pass ARN to resolvers:** 6th arg to `resolve_cfn_params` and
`resolve_tf_vars` (empty string when no training plan).

### 3. `resolve_cfn_params()` -- Training plan injection

Add `--arg training_plan_arn` to jq. Extend accelerated group filter:

```jq
if $training_plan_arn != "" then
    .TrainingPlanArn = $training_plan_arn
else .
end
```

Only the `accelerated-instance-group-1` gets the ARN. The general
(management) group is untouched.

### 4. `resolve_tf_vars()` -- Training plan injection

Add awk pass to insert `training_plan_arn = "<arn>"` after
`lifecycle_script` in the first instance group block. Only runs when
the 6th arg is non-empty.

The TF module already has `training_plan_arn = optional(string)` in
its variable type definition at:
- `terraform-modules/hyperpod-eks-tf/variables.tf:472`
- `terraform-modules/hyperpod-eks-tf/modules/hyperpod_cluster/variables.tf:42`

### 5. Mock responses

Add to `tests/helpers/mock_aws.bash` (before existing AZ catch-all):

| Mock Pattern | Response |
|-------------|----------|
| `sagemaker describe-training-plan` + `test-plan` | Active plan in us-west-2a |
| `sagemaker describe-training-plan` + `az4-plan` | Active plan in us-west-2d (different AZ) |
| `sagemaker describe-training-plan` + `expired-plan` | Expired plan in us-west-2a |
| `sagemaker describe-training-plan` + `failed-plan` | Failed plan, empty capacity |
| `sagemaker describe-training-plan` + `no-capacity-plan` | Scheduled, empty capacity |
| `sagemaker describe-training-plan` + `*` (catch-all) | ResourceNotFound error |
| `ec2 describe-availability-zones --zone-names us-west-2a` | `usw2-az2` |
| `ec2 describe-availability-zones --zone-names us-west-2d` | `usw2-az4` |

### 6. Test plan (12 tests)

| # | Test Name | Validates |
|---|-----------|-----------|
| 1 | `resolve_training_plan: valid plan sets ARN and AZ ID` | Happy path |
| 2 | `resolve_training_plan: nonexistent plan returns 1` | Unknown plan |
| 3 | `resolve_training_plan: empty name returns 1` | Guard |
| 4 | `resolve_training_plan: failed plan returns 1` | Status=Failed |
| 5 | `resolve_training_plan: expired plan warns but succeeds` | Status=Expired |
| 6 | `resolve_training_plan: no reserved capacity returns 1` | Empty summaries |
| 7 | `resolve_cfn_params: injects TrainingPlanArn` | jq adds to accel group |
| 8 | `resolve_cfn_params: omits TrainingPlanArn when empty` | No ARN = no key |
| 9 | `resolve_cfn_params: TrainingPlanArn absent from general group` | Mgmt untouched |
| 10 | `resolve_tf_vars: injects training_plan_arn` | awk inserts line |
| 11 | `resolve_tf_vars: omits training_plan_arn when empty` | No insertion |
| 12 | `deploy.sh: --help mentions --training-plan flag` | Usage text |

Expected total: 108 + 12 = 120 tests.

## Data Flow

```
--training-plan my-plan --instance-type ml.p5.48xlarge --az-id usw2-az2

  1. resolve_training_plan("my-plan", "us-west-2")
     -> sagemaker describe-training-plan -> ARN + AZ name "us-west-2d"
     -> ec2 describe-availability-zones --zone-names -> AZ ID "usw2-az4"

  2. AZ_ID (usw2-az2) != TRAINING_PLAN_AZ_ID (usw2-az4)
     -> WARNING + override AZ_ID

  3a. CFN: resolve_cfn_params(..., ARN)
      -> jq adds TrainingPlanArn to accelerated group
      -> Lambda passes through to AWS::SageMaker::Cluster

  3b. TF: resolve_tf_vars(..., ARN)
      -> awk inserts training_plan_arn in first group
      -> awscc_sagemaker_cluster passes through
```

## Deliverables

```
# Modified
lib/deploy_helpers.sh           (resolve_training_plan, resolve_cfn_params, resolve_tf_vars)
deploy.sh                       (--training-plan flag, AZ override, pass ARN)
tests/test_deploy.bats          (12 new tests)
tests/helpers/mock_aws.bash     (sagemaker + ec2 --zone-names mocks)
AGENTS.md                       (training plan docs)

# New
docs/features/training-plan-support/idea.md
docs/features/training-plan-support/plan.md

# Updated
docs/features/DASHBOARD.md      (In Progress row)
```
