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

<!--
REQUIRED for functional changes: paste evidence of an end-to-end run on a
real AWS environment (job output, benchmark numbers, relevant logs).

For changes with no functional impact, replace this section's content with
the single word: docs-only
-->

## Directory Structure

<!-- If adding or updating an example, ensure it follows the expected layout below. -->

```
examples/
└── <category>/                 # training, use-cases, or inference
    └── <name>/                 # e.g. fsdp, cosmos3, sglang/<sample>
        ├── Dockerfile          # Container / environment setup
        ├── README.md           # Overview, prerequisites, usage
        ├── slurm/              # Slurm-specific launch scripts
        ├── kubernetes/         # Kubernetes manifests
        └── hyperpod-eks/       # HyperPod EKS instructions
```

- `training/` for framework-centric training examples, `use-cases/` for
  model- or application-specific examples, `inference/` for serving
  examples organized by engine (see [AGENTS.md](https://github.com/awslabs/awsome-distributed-ai/blob/main/AGENTS.md) for placement rules).

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
- [ ] Test Results above contain e2e evidence from a real AWS run (or this PR is marked `docs-only`).
