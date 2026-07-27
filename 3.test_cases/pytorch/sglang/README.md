<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# SGLang test cases

[SGLang](https://github.com/sgl-project/sglang) is a fast, OpenAI-API-compatible
serving engine for large language and vision-language models. The samples in this
directory deploy SGLang on AWS for inference workloads, with high-performance EFA
networking and expert-parallel MoE all-to-all
([DeepEP](https://github.com/deepseek-ai/DeepEP)).

## Available test cases

| Test case | Orchestrator | Description |
| --- | --- | --- |
| [`dsr1-deepep-efa`](./dsr1-deepep-efa) | EC2 (Docker) | DeepSeek-R1 colocated 2-node inference (TP16/EP16) on `p5`/`p5en.48xlarge` with SGLang 0.5.13.post1 and **DeepEP** MoE all-to-all over NVSHMEM-libfabric/EFA. Also selects SGLang's ordinary all-to-all and pure TP for comparison. |

For the kernel-level DeepEP-on-EFA dispatch/combine benchmarks (Slurm and EKS
launchers, 1–32 nodes) see
[`micro-benchmarks/expert-parallelism/deepep-benchmark`](../../../micro-benchmarks/expert-parallelism/deepep-benchmark).
For the same model and fabric served by vLLM in a PD-disaggregated topology see
[`3.test_cases/pytorch/vllm/dsr1-deepep-efa`](../vllm/dsr1-deepep-efa).
