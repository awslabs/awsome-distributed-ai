---
id: slinky-helm-values-update
status: shipped
started: 2026-03-09
completed: 2026-03-09
shipped: 2026-03-12
---

# Shipped: Slinky Slurm Helm Values Syntax Update

## Summary

Consolidated two 1000+ line Helm values files (`g5-values.yaml`,
`p5-values.yaml`) into a single 153-line `slurm-values.yaml.template`
with shell template variables. Bumped chart version from v0.3.0 to
v1.0.1, added a MariaDB CR manifest for Slurm accounting, and added
an NLB service patch template for SSH access.

## What Changed

- **Helm values consolidation**: Replaced static per-profile files with
  a single template using 10 shell variables (`${accel_instance_type}`,
  `${gpu_count}`, `${efa_count}`, `${gpu_gres}`, `${replicas}`,
  `${image_repository}`, `${image_tag}`, `${ssh_key}`,
  `${mgmt_instance_type}`, `${pvc_name}`)
- **API syntax update (v0.3.0 -> v1.0.1)**: `login:` -> `loginsets:`,
  `compute:` -> `nodesets:`, added `vendor.nvidia.dcgm` and
  `configFiles.gres.conf`, removed ~900 lines of verbose debug/
  override/auth sections
- **mariadb.yaml**: MariaDB CR using mariadb-operator CRD with
  auto-generated passwords via Kubernetes Secrets
- **slurm-login-service-patch.yaml.template**: NLB annotations with
  `${ip_address}/32` source range restriction for SSH access
- **resolve_helm_profile()**: New function in `lib/deploy_helpers.sh`
  that queries EC2 API for GPU count, model, EFA interfaces and builds
  GRES strings for any SageMaker instance type
- **Deleted**: `g5/g5-values.yaml`, `p5/p5-values.yaml`

## Files Modified

| File | Change |
|------|--------|
| `slurm-values.yaml.template` | New consolidated template (153 lines) |
| `tests/fixtures/slurm-values.yaml.template` | Test fixture (byte-identical) |
| `mariadb.yaml` | New MariaDB CR manifest |
| `slurm-login-service-patch.yaml.template` | New NLB service patch |
| `lib/deploy_helpers.sh` | Added `resolve_helm_profile()` |
| `setup.sh` | Template substitution via sed |

## Quality Gates

### Security Audit: APPROVED

- No hardcoded credentials; MariaDB passwords auto-generated via
  Kubernetes Secrets (`rootPasswordSecretKeyRef`, `passwordSecretKeyRef`
  with `generate: true`)
- NLB IP-restricted to deployer's public IP (`${ip_address}/32`);
  SSH key-only auth (no password), ed25519 algorithm
- Container images from trusted registries (Slinky official, AWS DLC
  ECR, customer ECR)
- Template variables derived from validated AWS API responses,
  eliminating injection risk
- Rendered files (`slurm-values.yaml`, `slurm-login-service-patch.yaml`)
  gitignored
- Advisory: race window between Helm install and NLB patch (service
  briefly exposed without IP restriction), MariaDB `bind-address=*`
  more permissive than needed, no CPU/memory limits on management
  pods (non-blocking)

### QA Validation: APPROVED

- 162/162 bats tests passing (80 deploy + 49 install + 13 setup +
  20 destroy)
- All 5 shell scripts pass `bash -n` syntax check
- Template uses v1.0.1 API syntax (loginsets, nodesets), all 10
  variables present
- Production and fixture templates are byte-identical
- MariaDB manifest valid CRD structure with auto-generated secrets
- NLB patch has correct AWS LB Controller annotations and IP
  restriction
- Template substitution tests verify g5/p5 produce valid YAML with
  no unresolved variables
- resolve_helm_profile tests cover g5/p5/g6, Neuron/CPU rejection
- All 6 plan acceptance criteria checked

## Acceptance Criteria

All 6 implementation steps completed:

1. Created `slurm-values.yaml.template` with shell template variables
2. Created `mariadb.yaml` MariaDB CR for Slurm accounting
3. Created `slurm-login-service-patch.yaml.template` NLB patch
4. Deleted old `g5/g5-values.yaml` and `p5/p5-values.yaml`
5. Added `resolve_helm_profile` to `lib/deploy_helpers.sh`
6. Added bats tests for template substitution and profile resolution
