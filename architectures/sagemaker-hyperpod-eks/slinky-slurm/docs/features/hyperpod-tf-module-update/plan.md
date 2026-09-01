---
id: hyperpod-tf-module-update
status: done
started: 2026-03-06
completed: 2026-03-06
---

# HyperPod Terraform Module Syntax Update — Plan

## Steps

- [x] 1. Analyze old vs new tfvars syntax differences
- [x] 2. Rewrite `g5/g5-custom.tfvars` to new module syntax
- [x] 3. Rewrite `p5/p5-custom.tfvars` to new module syntax (mirror g5 with p5 values)
- [x] 4. Verify g5/p5 symmetry
- [x] 5. Update `README.md` Terraform section if needed (no changes needed)
- [x] 6. Mark complete

## Analysis

### Changes required

| Change | Old | New |
|--------|-----|-----|
| `instance_groups` type | map `{}` | list of objects `[]` |
| Group name | map key | `name` field inside object |
| `availability_zone_id` | top-level var | moved into each instance group object |
| `aws_region` | not present | new top-level var |
| Module toggles | not present | 8 new boolean/string vars |
| `kubernetes_version` | `"1.32"` | `"1.34"` (match reference) |

### New module toggle variables

```hcl
create_observability_module               = true
network_metric_level                      = "ADVANCED"
logging_enabled                           = true
create_task_governance_module             = true
create_hyperpod_training_operator_module  = true
create_hyperpod_inference_operator_module = true
enable_guardduty_cleanup                  = true
create_new_fsx_filesystem                 = true
```

### Intentional g5/p5 differences

- `instance_type`: `ml.g5.8xlarge` vs `ml.p5.48xlarge`
- `instance_count`: `4` vs `2`

All other values (cluster names, toggles, k8s version, AZ, etc.) remain identical.

## Open Questions

- [x] Should `kubernetes_version` be bumped to `1.34`?
  - **Yes** — aligning with the reference module's tested version (bumped from
    initial `1.33` to `1.34` after cross-referencing the CFN params reference).
- [x] Should we keep the general-instance-group-2 (m5.2xlarge)?
  - **Yes** — it hosts non-compute components (controller, login, etc.).
