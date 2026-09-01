---
id: training-plan-support
name: Training Plan Support for deploy.sh
type: Feature
priority: P1
effort: Medium
impact: High
created: 2026-03-12
---

# Training Plan Support for deploy.sh

## Problem Statement

SageMaker Training Plans provide reserved GPU capacity (e.g., `ml.p5.48xlarge`)
at reduced rates via Flexible Training Plans. The `TrainingPlanArn` must be
included in the instance group specification when calling `CreateCluster`.
However, `deploy.sh` has no way to specify a training plan -- users must
manually edit the `InstanceGroupSettings1` JSON inside `params.json` (CFN path)
or add `training_plan_arn` to the instance group in `custom.tfvars` (TF path),
and must manually ensure `--az-id` matches the training plan's reserved capacity
availability zone. This is error-prone, undocumented, and not agent-friendly.

Key pain points:

1. **No CLI flag** -- users must hand-edit dense JSON strings or HCL blocks
2. **AZ mismatch risk** -- the cluster subnet must be in the training plan's AZ,
   but nothing validates or auto-resolves this
3. **No validation** -- no check that the plan exists, is active/scheduled, or
   targets `hyperpod-cluster` resources
4. **Undocumented** -- no guidance in README, AGENTS.md, or skill docs

## Proposed Solution

Add an optional `--training-plan <name>` flag to `deploy.sh` that:

1. **Validates** the plan exists via
   `aws sagemaker describe-training-plan --training-plan-name <name>`
2. **Auto-resolves** the full ARN from the response
3. **Resolves the AZ** from `ReservedCapacitySummaries[0].AvailabilityZone`
   (AZ name like `us-west-2a`) to an AZ ID (like `usw2-az2`) via
   `aws ec2 describe-availability-zones --zone-names`
4. **Overrides `--az-id`** with a warning if the user's value differs from
   the training plan's AZ (warn-and-continue, no prompt -- agent-friendly)
5. **Injects `TrainingPlanArn`** into only the accelerated instance group for
   both CFN and TF paths

### Upstream Data Flow Validation

The data flow through the upstream CFN templates has been traced and validated:

**CFN path:**
```
params.json InstanceGroupSettings1 (JSON string with TrainingPlanArn)
  -> main-stack-eks-based-template.yaml (pass-through)
  -> hyperpod-cluster-template.yaml (INSTANCE_GROUP_SETTINGS1 env var)
  -> Lambda combine_settings() -> enrich_instance_groups()
    (passes unknown keys through; only strips TargetAvailabilityZoneId
     and InstanceGroupType)
  -> generate_cluster_template_yaml()
    (dumps create_params as AWS::SageMaker::Cluster Properties)
  -> Nested CFN stack creates the cluster
```

**TF path:**
```
custom.tfvars instance_groups[0].training_plan_arn
  -> variables.tf: training_plan_arn = optional(string) (already defined)
  -> modules/hyperpod_cluster/main.tf: conditional merge()
    (includes training_plan_arn only when non-null)
  -> awscc_sagemaker_cluster resource
```

Both paths already support `TrainingPlanArn` as a passthrough -- the only
missing piece is the CLI flag and the `resolve_training_plan()` function
in `lib/deploy_helpers.sh`.

### API Details

**DescribeTrainingPlan response** (relevant fields):
```json
{
  "TrainingPlanArn": "arn:aws:sagemaker:<region>:<account>:training-plan/<name>",
  "TrainingPlanName": "<name>",
  "Status": "Active|Scheduled|Pending|Expired|Failed",
  "ReservedCapacitySummaries": [{
    "AvailabilityZone": "us-west-2a",
    "InstanceType": "ml.p5.48xlarge",
    "TotalInstanceCount": 2,
    "Status": "Active"
  }],
  "TargetResources": ["hyperpod-cluster"]
}
```

**AZ name -> AZ ID resolution:**
```bash
aws ec2 describe-availability-zones \
    --zone-names us-west-2a \
    --query 'AvailabilityZones[0].ZoneId' --output text
# -> usw2-az2
```

### Scope

- **In scope:** `deploy.sh`, `lib/deploy_helpers.sh`, `tests/test_deploy.bats`,
  `tests/helpers/mock_aws.bash`, README, AGENTS.md
- **Out of scope:** `setup.sh`, `install.sh`, `destroy.sh` (don't interact with
  the training plan), upstream Lambda/CFN template changes

### User Experience

```bash
# Deploy with a training plan (AZ auto-resolved from plan)
bash deploy.sh \
    --instance-type ml.p5.48xlarge \
    --instance-count 2 \
    --training-plan my-p5-plan \
    --infra cfn

# Output:
#   Resolving training plan 'my-p5-plan'...
#     ARN: arn:aws:sagemaker:us-west-2:123456789012:training-plan/my-p5-plan
#     Status: Active
#     AZ: us-west-2a (usw2-az2)
#
#   WARNING: Overriding --az-id from 'usw2-az2' to 'usw2-az4'
#     to match training plan 'my-p5-plan'.
#     The cluster subnet must be in the training plan's AZ.
```
