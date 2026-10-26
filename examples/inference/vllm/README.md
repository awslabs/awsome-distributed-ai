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
| [`multi-instance-layouts`](./multi-instance-layouts) | EKS | Kimi K2 Instruct-0905 (FP8) on `p5en.48xlarge` with the AWS vLLM Deep Learning Container 0.32: TP 8 on one instance against PP 2 x TP 8 and TP 16 across two, each with a near and a far second instance by EC2 network topology. |
