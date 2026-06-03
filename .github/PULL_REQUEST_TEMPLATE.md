## Purpose

<!-- Link related issues using "Fixes #123" or "Relates to #123" -->

## Changes

<!-- Summarize the changes made in this PR -->

-

## Test Plan

<!-- Describe how you tested these changes -->

**Environment:**
- AWS Service:
- Instance type:
- Number of nodes:

**Test commands:**
```bash

```

## Test Results

<!-- Share relevant metrics, logs, or screenshots -->

## Directory Structure

<!-- If adding or updating a test case, ensure it follows the expected layout below. -->

```
examples/
└── <framework>/                # e.g. pytorch, megatron, jax
    └── <library>/              # e.g. picotron, FSDP, megatron-lm
        └── <model>/            # e.g. SmolLM-1.7B (may be omitted for single-model cases)
            ├── Dockerfile      # Container / environment setup
            ├── README.md       # Overview, prerequisites, usage
            ├── slurm/          # Slurm-specific launch scripts
            ├── kubernetes/     # Kubernetes manifests
            └── hyperpod-eks/   # HyperPod EKS instructions
```

- Top-level files (`Dockerfile`, `README.md`, training scripts, configs) cover general setup.
- Subdirectories (`slurm/`, `kubernetes/`, `hyperpod-eks/`) contain service-specific launch instructions.
- Not all service subdirectories are required — include only the ones relevant to your test case.

## Checklist

- [ ] I have read the [contributing guidelines](https://github.com/awslabs/awsome-distributed-ai/blob/main/CONTRIBUTING.md).
- [ ] I am working against the latest `main` branch.
- [ ] I have searched existing open and recently merged PRs to confirm this is not a duplicate.
- [ ] The contribution is self-contained with documentation and scripts.
- [ ] External dependencies are pinned to a specific version or tag (no `latest`).
- [ ] A README is included or updated with prerequisites, instructions, and known issues.
- [ ] New test cases follow the [expected directory structure](#directory-structure).
