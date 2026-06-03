---
id: slinky-helm-values-update
status: done
started: 2026-03-09
completed: 2026-03-09
---

# Slinky Slurm Helm Values Syntax Update — Plan

## Steps

- [x] 1. Create `slurm-values.yaml.template` — single consolidated template
      with shell variables for all instance-type-specific values
- [x] 2. Create `mariadb.yaml` — MariaDB CR manifest for Slurm accounting
- [x] 3. Create `slurm-login-service-patch.yaml.template` — NLB service patch
- [x] 4. Delete `g5/g5-values.yaml` and `p5/p5-values.yaml`
- [x] 5. Add `resolve_helm_profile` function to `lib/deploy_helpers.sh`
- [x] 6. Add bats tests for template substitution and helm profile resolution

## What Changed

Replaced two 1000+ line Helm values files (`g5/g5-values.yaml` and
`p5/p5-values.yaml`) with a single 145-line `slurm-values.yaml.template`
using shell template variables. Chart version bumped from v0.3.0 to v1.0.1.

### Template Variables

| Variable | g5 | p5 |
|----------|----|----|
| `${accel_instance_type}` | `ml.g5.8xlarge` | `ml.p5.48xlarge` |
| `${gpu_count}` | 1 | 8 |
| `${efa_count}` | 1 | 32 |
| `${gpu_gres}` | `gpu:a10g:1` | `gpu:h100:8` |
| `${replicas}` | 4 | 2 |

### Key API Changes (v0.3.0 -> v1.0.1)

- `login:` -> `loginsets:` (map-based with named login sets)
- `compute:` -> `nodesets:` (map-based with named node sets)
- `accounting:` uses `podSpec:` for affinity
- Added `vendor.nvidia.dcgm` and `configFiles.gres.conf`
- Removed verbose debug/override/auth sections (~900 lines)

### New Manifests

- `mariadb.yaml` — MariaDB CR using mariadb-operator CRD
- `slurm-login-service-patch.yaml.template` — NLB annotations with
  `${ip_address}` source range restriction
