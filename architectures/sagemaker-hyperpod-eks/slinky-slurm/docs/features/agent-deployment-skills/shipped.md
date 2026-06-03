---
id: agent-deployment-skills
status: shipped
started: 2026-03-09
completed: 2026-03-09
shipped: 2026-03-12
---

# Shipped: Agent Deployment Skills & Validation Tests

## Summary

Created 6 OpenCode agent deployment skills and expanded the bats-core
test suite to 162 unit tests across 4 test files. The skills enable
AI agents to execute the full slinky-slurm deployment workflow from
preflight validation through post-deployment health checks.

## What Changed

- **bash-testing skill**: Migrated from flat `.md` file to official
  `SKILL.md` format with YAML frontmatter
- **deployment-preflight skill**: Validates CLI tools, AWS credentials,
  region, AZ, and per-phase prerequisites (deploy/setup/install)
- **deploy-infrastructure skill**: Guides `deploy.sh` usage for both
  CFN and TF paths with internal workflow documentation, verification
  steps, and troubleshooting table
- **build-slurm-image skill**: Guides `setup.sh` with all 3 build
  paths (CodeBuild, local Docker, skip), SSH key generation, and
  template substitution
- **deploy-slurm-cluster skill**: Guides `install.sh` with phases
  A-E (setup, MariaDB, Slurm operator, Slurm cluster, NLB config),
  skip flags, and bring-your-own-cluster support
- **validate-deployment skill**: 7-step post-deployment health check
  covering pods, nodes, NLB, SSH, Slurm services, and test jobs
- **Test expansion**: Added `test_setup.bats` (13 tests),
  `test_install.bats` (49 tests), `test_destroy.bats` (20 tests),
  expanded `test_deploy.bats` to 80 tests

## Files Modified

| File | Change |
|------|--------|
| `.opencode/skills/bash-testing/SKILL.md` | Migrated from flat file |
| `.opencode/skills/deployment-preflight/SKILL.md` | New skill |
| `.opencode/skills/deploy-infrastructure/SKILL.md` | New skill |
| `.opencode/skills/build-slurm-image/SKILL.md` | New skill |
| `.opencode/skills/deploy-slurm-cluster/SKILL.md` | New skill |
| `.opencode/skills/validate-deployment/SKILL.md` | New skill |
| `tests/test_deploy.bats` | Expanded to 80 tests |
| `tests/test_setup.bats` | New — 13 tests |
| `tests/test_install.bats` | New — 49 tests |
| `tests/test_destroy.bats` | New — 20 tests |
| `tests/helpers/setup.bash` | Common test setup/teardown |
| `tests/helpers/mock_aws.bash` | AWS CLI mock with canned responses |
| `tests/fixtures/` | params.json, custom.tfvars, template fixture |
| `tests/install_bats_libs.sh` | bats helper library installer |

## Quality Gates

### Security Audit: APPROVED

- No real AWS credentials, account IDs, or infrastructure fingerprinting
  data in any skill or test file
- All account IDs are placeholder `123456789012` or public DLC ECR
  `763104351884`
- No unguarded destructive commands in skills; `destroy.sh` requires
  confirmation
- IAM roles in skills are properly scoped; NLB is IP-restricted
- Mock data is clearly fake with fail-safe catch-all for unmocked calls
- Test isolation via `mktemp` with teardown cleanup
- Advisory: add explicit SSH private key commit prevention note to
  build-slurm-image skill (non-blocking)

### QA Validation: APPROVED

- 162/162 bats tests passing (80 deploy + 49 install + 13 setup +
  20 destroy)
- All 5 shell scripts pass `bash -n` syntax check
- All 6 skills have proper YAML frontmatter, Overview, and Steps
  sections
- Old flat skill file (`bash-testing.md`) correctly removed
- Test infrastructure complete: helpers, mock, 3 fixtures, installer
- All 11 plan acceptance criteria checked

## Acceptance Criteria

All 11 implementation steps completed:

1. Created feature idea.md
2. Migrated `bash-testing` skill to SKILL.md format
2a. Created `.opencode/skills/bash-testing/SKILL.md`
2b. Removed old flat file
3. Created `deployment-preflight` skill
4. Created `deploy-infrastructure` skill
5. Created `build-slurm-image` skill
6. Created `deploy-slurm-cluster` skill
7. Created `validate-deployment` skill
8. Created `tests/test_setup.bats` (13 tests)
9. Created `tests/test_install.bats` (49 tests)
10. Created `tests/test_destroy.bats` (20 tests)
11. All 162 bats tests pass
