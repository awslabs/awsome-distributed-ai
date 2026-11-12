<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# TensorRT-LLM test cases

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) is NVIDIA's open-source
inference engine for large language models on NVIDIA GPUs. The samples in this
directory deploy TensorRT-LLM on AWS with high-performance EFA networking and
expert-parallel MoE all-to-all.

## Available test cases

| Test case | Orchestrator | Description |
| --- | --- | --- |
| [`nccl-ep-efa`](./nccl-ep-efa) | Kubernetes (2-node) | Wide-EP MoE dispatch/combine via TRT-LLM's **`NcclEP`** backend (`nccl.ep` / `libnccl_ep` — NOT the `deep_ep` package) over **AWS EFA**, using aws-ofi-nccl with **GIN** (GPU-Initiated Networking) CPU-proxy. Image built NGC-from-scratch from public sources; the recipe runs image build → transport smoke test → served `/v1/chat/completions` → concurrency benchmark. Validated on `p5en.48xlarge` (H200). |

For kernel-level expert-parallelism dispatch/combine benchmarks over EFA —
including a DeepEP V2 benchmark on the same NCCL-GIN substrate this test case
uses — see
[`micro-benchmarks/expert-parallelism`](../../../micro-benchmarks/expert-parallelism).
