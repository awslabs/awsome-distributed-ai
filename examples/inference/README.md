# Inference Examples

Framework-centric inference engine examples, organized by serving engine.

| Engine | Example | Description |
|---|---|---|
| [`vllm`](./vllm) | [`dsv3-uccl-nixl`](./vllm/dsv3-uccl-nixl) | DeepSeek-V3 disaggregated (prefill/decode) inference with vLLM, UCCL-EP, and NIXL on EKS |
| [`sglang`](./sglang) | [`qwen3.5-27b-b300-intra-pd`](./sglang/qwen3.5-27b-b300-intra-pd) | Qwen3.5-27B with intra-node prefill/decode disaggregation on a single B300 node |
| [`sglang`](./sglang) | [`kimi2.6-h200-1p1d`](./sglang/kimi2.6-h200-1p1d) | Kimi2.6 with node-level 1-prefill / 1-decode disaggregation across two H200 nodes |
| [`sglang`](./sglang) | [`dsv4pro-b300-single-node`](./sglang/dsv4pro-b300-single-node) | DeepSeek V4 Pro unified (non-PD) serving on a single B300 node |
| [`sglang`](./sglang) | [`dsv4flash-b300-intra-3p1d`](./sglang/dsv4flash-b300-intra-3p1d) | DeepSeek-V4-Flash with intra-node 3-prefill / 1-decode disaggregation (tp=2 each) on a single B300 node |
| [`sglang`](./sglang) | [`glm5.2-b300-tp2-dp4`](./sglang/glm5.2-b300-tp2-dp4) | GLM-5.2 (NVFP4) as 4× independent tp=2 engines behind an SGLang router on a single B300 node |

More engines (TRT-LLM, NIM, Dynamo, Ray Serve, …) are planned, including
content to be merged from [`aws-samples/awsome-inference`](https://github.com/aws-samples/awsome-inference)
(see issue #1056).
