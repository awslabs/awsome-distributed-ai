<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# llm-d test cases

[llm-d](https://github.com/llm-d/llm-d) deployments on AWS. llm-d puts a Kubernetes-native, inference-aware router (the Endpoint Picker, EPP, from the [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension)) in front of ordinary model-server pods. Each sub-directory is a self-contained sample.

| Test case | Hardware | Topology | Images |
| --- | --- | --- | --- |
| [`dsv4flash-agg`](./dsv4flash-agg) | 1× B200 (`p6-b200.48xlarge`, 8 GPU) | DeepSeek-V4-Flash-0731 as 4× aggregated SGLang tp=2 replicas behind the llm-d router (v1.0.0 optimized-baseline, Standalone Mode: EPP + Envoy sidecar, no Gateway controller). Replicas are discovered through an `InferencePool`; prefix-cache-aware routing | `lmsysorg/sglang:v0.5.22`, `ghcr.io/llm-d/llm-d-router-endpoint-picker:v1.0.0` (pinned by the chart) |
