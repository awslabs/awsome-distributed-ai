# Inference Examples

Framework-centric inference engine examples, organized by serving engine.

| Engine | Example | Description |
|---|---|---|
| [`vllm`](./vllm) | [`dsv3-uccl-nixl`](./vllm/dsv3-uccl-nixl) | DeepSeek-V3 disaggregated (prefill/decode) inference with vLLM, UCCL-EP, and NIXL on EKS |
| [`vllm`](./vllm) | [`cosmos-reason`](./vllm/cosmos-reason) | NVIDIA Cosmos Reason physical-reasoning VLM served by vLLM on EKS / SageMaker HyperPod EKS |
| [`sglang`](./sglang) | [`dsr1-deepep-efa`](./sglang/dsr1-deepep-efa) | DeepSeek-R1 on p5/p5en (EC2 + Docker) with DeepEP MoE all-to-all over NVSHMEM-libfabric/EFA — colocated and 2P2D PD-disaggregated |
| [`nvidia-dynamo`](./nvidia-dynamo) | [`nvidia-dynamo`](./nvidia-dynamo) | NVIDIA Dynamo (SGLang backend) on SageMaker HyperPod EKS — aggregated and PD-disaggregated serving via the Dynamo operator |
| [`nvidia-dynamo`](./nvidia-dynamo) | [`deepep-v2-efa`](./nvidia-dynamo/deepep-v2-efa) | `Qwen3-30B-A3B-FP8` MoE served by NVIDIA Dynamo (`dynamo.frontend` + `dynamo.vllm`) with DeepEP-V2 expert-parallel all-to-all over EFA via the NCCL-GIN CPU-proxy on `p5en.48xlarge` EKS — the vLLM `deepep-v2-efa` substrate plus a Dynamo serving front; DP16/EP16 (2 nodes) and DP32/EP32 (4 nodes) |
| [`sglang`](./sglang) | [`qwen3.5-27b-b300-intra-pd`](./sglang/qwen3.5-27b-b300-intra-pd) | Qwen3.5-27B with intra-node prefill/decode disaggregation on a single B300 node |
| [`sglang`](./sglang) | [`kimi2.6-h200-1p1d`](./sglang/kimi2.6-h200-1p1d) | Kimi2.6 with node-level 1-prefill / 1-decode disaggregation across two H200 nodes |
| [`sglang`](./sglang) | [`dsv4pro-b300-single-node`](./sglang/dsv4pro-b300-single-node) | DeepSeek V4 Pro unified (non-PD) serving on a single B300 node |
| [`sglang`](./sglang) | [`glm5.2-b300-tp2-dp4`](./sglang/glm5.2-b300-tp2-dp4) | GLM-5.2 (NVFP4) as 4× independent tp=2 engines behind an SGLang router on a single B300 node |
| [`sglang`](./sglang) | [`dsv41flash-pd-autodiscovery`](./sglang/dsv41flash-pd-autodiscovery) | DeepSeek-V4.1-Flash on B200 (accepts B300 node labels, verified on B200 so far): pod-per-engine prefill/decode disaggregation with independent scaling and router service discovery, over either NIXL/EFA RDMA or same-node NVLink |

More engines (TRT-LLM, NIM, Ray Serve, …) are planned, including
content to be merged from [`aws-samples/awsome-inference`](https://github.com/aws-samples/awsome-inference)
(see issue #1056).
