# Agent Instructions for awsome-distributed-ai

> These instructions apply to **all** AI-assisted contributions to
> `awslabs/awsome-distributed-ai`. They complement
> [CONTRIBUTING.md](./CONTRIBUTING.md), which remains the authority on
> contribution format and process.

## 1. Contribution Policy

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo awslabs/awsome-distributed-ai --comments
gh pr list --repo awslabs/awsome-distributed-ai --state open --search "<issue_number> in:body"
gh pr list --repo awslabs/awsome-distributed-ai --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change,
link fix, etc.). Mechanical cleanups are acceptable only when bundled with
substantive work.

### Accountability

- A human submitter must understand and defend the change end-to-end, review
  every changed line, and run the relevant verification.
- This repository ships **runnable reference material, not a library**: the
  test for a change is that the example actually works on the target
  hardware. CONTRIBUTING.md already requires test cases to be verified at
  their stated scale ("if you say 256 A100, test on that scale").
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - How the change was verified — for functional changes, the cluster
      type, instance types, and scale used, with relevant command output or
      logs under `## Test Results`. For changes with no functional impact,
      write `docs-only` there instead. This is CI-enforced by the
      `require-e2e-evidence` check.

### Fail-closed behavior

If the work is duplicate or trivial busywork, or you cannot verify a
functional change on representative hardware, **do not proceed**. Return a
short explanation of what is missing.

## 2. Repository Layout — where things go

| Directory | Contents |
| --- | --- |
| `architectures/` | Cluster-level reference architectures, one per orchestrator (ParallelCluster, PCS, HyperPod Slurm/EKS, EKS) plus shared building blocks (`common/`, `vpc_network/`, `ldap_server/`, `accounting-database/`) |
| `examples/training/` | Framework-centric training examples (FSDP, Megatron-LM, NeMo, verl, …) |
| `examples/use-cases/` | Model- or application-specific examples (fine-tunes, domain pipelines, world models, VLA models, …) |
| `examples/inference/` | Serving examples, organized by engine (vLLM, SGLang, NVIDIA Dynamo, …) |
| `micro-benchmarks/` | Low-level performance benchmarks (NCCL tests, expert-parallelism kernels, NVSHMEM, …) |
| `validation_and_observability/` | Cluster health checks, exporters, monitoring stacks |
| `ami_and_containers/` | Machine image build assets (Packer/Ansible) |
| `docs/` | Cross-cutting prose documentation (e.g. the [EFA cheatsheet](./docs/efa-cheatsheet.md)) |

Placement rules for new content:

- A **training framework** example goes to `examples/training/<framework>/`.
- A **specific model or application** (even if it trains) goes to
  `examples/use-cases/<name>/`.
- Anything whose purpose is **serving** goes to
  `examples/inference/<engine>/<name>/`.
- Kernel- or transport-level performance measurement goes to
  `micro-benchmarks/`.
- Every example is **self-contained**: its own README (prerequisites,
  copy-pasteable run instructions, known issues), pinned dependency
  versions (no `latest` tags), and everything needed to run it. See
  "Contributions format" in CONTRIBUTING.md.

## 3. Development Workflow

There is no repo-wide build. Verification is per-asset:

```bash
# Docker-build smoke tests (fixtures come from the root conftest.py —
# run pytest from the repo root or pass the example directory):
python3 -m pytest examples/training/megatron-lm/ -v
# Keep the built image for inspection instead of removing it:
python3 -m pytest examples/training/megatron-bridge/ -v --keep-artifacts
```

- Markdown follows the root [`.markdownlint.jsonc`](./.markdownlint.jsonc)
  (`npx markdownlint-cli2 "<changed files>"`).
- CloudFormation/Terraform changes: validate with the tooling documented in
  the area you are editing (e.g. `terraform validate`, per-area lint
  scripts).
- Functional changes to an example are verified by running it on the target
  hardware — a successful `docker build` alone is not verification.

## 4. Domain-Specific Guides

Read the relevant guide before modifying these areas. If a guide conflicts
with the requested change, stop and explain rather than proceeding.

- HyperPod EKS Terraform modules:
  [`architectures/sagemaker-hyperpod-eks/terraform-modules/AGENTS.md`](./architectures/sagemaker-hyperpod-eks/terraform-modules/AGENTS.md)
- Slinky Slurm on HyperPod EKS:
  [`architectures/sagemaker-hyperpod-eks/slinky-slurm/AGENTS.md`](./architectures/sagemaker-hyperpod-eks/slinky-slurm/AGENTS.md)
- HyperPod lifecycle scripts (`LifecycleScripts/base-config`) are owned by
  `@awslabs/hyperpod-lcs-dev` (see [CODEOWNERS](./.github/CODEOWNERS)) —
  changes there require that team's review.
