---
id: hyperpod-tf-module-update
name: HyperPod Terraform Module Syntax Update
type: Enhancement
priority: P1
effort: Medium
impact: High
created: 2026-03-06
---

# HyperPod Terraform Module Syntax Update

## Problem Statement

The Terraform deployment option (`g5/g5-custom.tfvars` and `p5/p5-custom.tfvars`)
uses outdated syntax that is incompatible with the latest SageMaker HyperPod EKS
Terraform modules. The tfvars files need to be updated to match the current module
interface located at
`architectures/sagemaker-hyperpod-eks/terraform-modules/hyperpod-eks-tf/custom.tfvars`.

## Proposed Solution

Update both `g5/g5-custom.tfvars` and `p5/p5-custom.tfvars` to align with the new
module syntax. Key changes identified:

1. **`instance_groups` type change:** Map (`{}`) to list of objects (`[]`), with
   the group name moved into a `name` field inside each object.
2. **`availability_zone_id` moved:** From a top-level variable into each instance
   group object.
3. **`aws_region` added:** New top-level variable (replaces implicit region config).
4. **New module toggles added:**
   - `create_observability_module`
   - `network_metric_level`
   - `logging_enabled`
   - `create_task_governance_module`
   - `create_hyperpod_training_operator_module`
   - `create_hyperpod_inference_operator_module`
   - `enable_guardduty_cleanup`
   - `create_new_fsx_filesystem`

### Old syntax (current)

```hcl
availability_zone_id = "usw2-az2"
instance_groups = {
    accelerated-instance-group-1 = {
        instance_type  = "ml.g5.8xlarge",
        instance_count = 4,
        ...
    }
}
```

### New syntax (target)

```hcl
aws_region = "us-west-2"
instance_groups = [
    {
        name                 = "accelerated-instance-group-1"
        instance_type        = "ml.g5.8xlarge",
        instance_count       = 4,
        availability_zone_id = "usw2-az2",
        ...
    }
]
create_observability_module               = true
create_task_governance_module             = true
create_hyperpod_training_operator_module  = true
create_hyperpod_inference_operator_module = true
enable_guardduty_cleanup                  = true
create_new_fsx_filesystem                 = true
```

### Key References

- **New module tfvars:**
  `architectures/sagemaker-hyperpod-eks/terraform-modules/hyperpod-eks-tf/custom.tfvars`
- **Existing files to update:**
  `g5/g5-custom.tfvars`, `p5/p5-custom.tfvars`

## Success Criteria

- [ ] TBD

## Notes

Created via feature-capture
