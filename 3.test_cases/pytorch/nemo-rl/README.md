<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# NeMo-RL test cases

[NeMo-RL](https://github.com/NVIDIA-NeMo/RL) is NVIDIA's scalable post-training library (GRPO,
DPO, SFT) for models from 1 GPU to thousands, with Megatron-core and DTensor training backends.
The samples in this directory deploy NeMo-RL on AWS with high-performance EFA networking and
expert-parallel MoE all-to-all.

## Available test cases

| Test case | Orchestrator | Description |
| --- | --- | --- |
| [`deepep-v2-efa`](./deepep-v2-efa) | Kubernetes (Ray cluster, 2-node) | GRPO post-training with MoE expert-parallel dispatch/combine via **DeepEP V2's NCCL backend** (`ElasticBuffer`) over **AWS EFA**, using aws-ofi-nccl with **GIN** (GPU-Initiated Networking) CPU-proxy. Image built NGC-from-scratch from public sources; the recipe runs image build → static verify → cross-node rollout probe → N-step Megatron MoE training gate. Mechanism chain measured on 2× p5.48xlarge (H100); this folder's image assembly is build-staged with an opt-in draft-PR layer for the full rollout path (honest measured-vs-staged breakdown in its README). |

For RL post-training with a different stack (SLIME + SGLang) on the same HyperPod-EKS Ray-cluster
pattern, see [`3.test_cases/pytorch/slime`](../slime). For kernel-level expert-parallelism
dispatch/combine benchmarks over EFA — including a DeepEP V2 benchmark on the same NCCL-GIN
substrate this test case uses — see
[`micro-benchmarks/expert-parallelism`](../../../micro-benchmarks/expert-parallelism).
