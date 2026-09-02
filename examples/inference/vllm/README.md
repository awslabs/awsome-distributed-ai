<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# vLLM test cases

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput,
OpenAI-API-compatible LLM serving engine. The samples in this directory
deploy vLLM on AWS for inference workloads, with options for high-
performance EFA networking, expert-parallel all-to-all
([UCCL-EP](https://github.com/uccl-project/uccl)), and disaggregated
prefill/decode KV-cache transfer
([NIXL](https://github.com/ai-dynamo/nixl)).

## Available test cases

| Test case | Orchestrator | Description |
| --- | --- | --- |
| [`dsv3-uccl-nixl`](./dsv3-uccl-nixl) | EKS / HyperPod EKS | DeepSeek-V3 disaggregated inference (1P+ND) on `p5en.48xlarge` with vLLM 0.21.0, UCCL-EP, and NIXL over EFA. |
| [`deepep-v2-GDAKI-efa`](./deepep-v2-GDAKI-efa) | EKS | Mixture-of-Experts (`Qwen3-30B-A3B-FP8`) with DeepEP-V2 expert-parallel all-to-all over EFA via the NCCL-GIN **GDAKI** (GPU-initiated) transport, on `p5en.48xlarge`. Eager + non-eager, DP16/EP16 and DP32/EP32, plus a same-node-set GDAKI-vs-CPU-proxy transport A/B. |
