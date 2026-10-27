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
| [`kimi-k2-multi-node`](./kimi-k2-multi-node) | EKS | Kimi K2 (FP8) on two `p5en.48xlarge` with vLLM 0.32: pipeline parallelism (PP2×TP8) or tensor parallelism (TP16) across the instances, or one instance (TP8). |

## Serving one model across instances

When a model needs more than one instance, split it by layers across the instances (pipeline parallelism) and keep
tensor parallelism inside each instance. Serving Kimi K2 on two `p5en.48xlarge`
([`kimi-k2-multi-node`](./kimi-k2-multi-node)), pipeline parallelism 2 × tensor parallelism 8 took 2% and 5% more
time per output token than one instance at 1 and 8 requests in flight, while tensor parallelism 16 across both took
68% and 46% more, because every layer's all-reduces cross EFA. Both two-instance layouts served more at 32 requests
in flight, 2.9 and 2.0 times one instance's output, because one instance has KV cache for only 12,640 tokens next to
the weights. vLLM's [parallelism guide](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) gives the same
rule: tensor parallelism up to the GPUs in a node, pipeline parallelism across nodes.
