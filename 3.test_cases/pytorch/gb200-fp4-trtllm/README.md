# FP4 (NVFP4) Inference on GB200 with TensorRT-LLM

NVFP4 serving on **P6e-GB200** using TensorRT-LLM, with a throughput **reconciliation** test against the MLPerf envelope and an **accuracy gate**. Single node (4 GB200 GPUs of one `p6e-gb200.36xlarge`), arm64 (Grace).

> **GB200, not B300.** Grace/aarch64, `sm_100`. Distinct from any vLLM/UCCL disagg work on other hardware.

## Two paths

| Path | Steps |
|---|---|
| **Fast** | Pull a pre-quantized NVFP4 checkpoint (`nvidia/Llama-3.1-8B-Instruct-NVFP4`, or DeepSeek-R1-NVFP4 at scale), serve with `trtllm-serve`, benchmark the OpenAI endpoint. |
| **BYO** | One ModelOpt step: `huggingface_example.sh --model $HF --quant nvfp4 --tp 4` then `export_hf_checkpoint`; 128–512 calibration samples (minutes). |

## Two gates (do not trust throughput without them)

1. **Reconciliation** — measure per-GPU output tok/s (e.g. Llama 3.1 405B, server-like load) and compare to the MLPerf v5.x GB200 envelope (~3.4× vs H200 relative; ~170 tok/s/GPU offline on 405B absolute). A large miss means the backend is **not** hitting FP4 Tensor Cores (silent upcast) — the same failure mode the `transformer-engine-gemm` roofline tool catches.
2. **Accuracy** — `lm-eval-harness` on MMLU-PRO / GPQA vs the BF16/FP8 baseline; require **≤ 1% drop** (the published PTQ envelope).
3. **Sanity** — confirm the served checkpoint's `quantization_config` is `{quant_method: modelopt, ... nvfp4}` before believing any number.

## What's here

| File | Purpose |
|---|---|
| `gb200-fp4-trtllm.Dockerfile` | arm64 from TensorRT-LLM `release:1.2.0`; ModelOpt |
| `quantize.sh` | BYO NVFP4 quantization via ModelOpt |
| `serve.sh` | `trtllm-serve` of an NVFP4 checkpoint |
| `bench.sh` | rate-sweep + tok/s reconciliation against the MLPerf envelope |
| `accuracy_gate.sh` | lm-eval MMLU-PRO/GPQA ≤ 1% drop check |
| `slurm/serve.sbatch`, `kubernetes/serve-job.yaml` | per-orchestrator (vLLM family is k8s-only; the slurm variant is deliberate) |

## Run

```bash
# Fast path
MODEL=nvidia/Llama-3.1-8B-Instruct-NVFP4 ./serve.sh & ./bench.sh && ./accuracy_gate.sh
# Kubernetes
kubectl apply -f kubernetes/serve-job.yaml
```

## Testability

**Runnable** on one `p6e-gb200.36xlarge`. The reconciliation and accuracy gates are the deliverable — record measured tok/s and the MMLU-PRO/GPQA deltas in the PR.

## Version pins

TensorRT-LLM 1.2.0 (MLPerf used 0.18.0.dev) · NVIDIA ModelOpt 0.23–0.44 · CUDA 12.8 (sm_100) / 12.9 (sm_103) · arm64 (Grace).
