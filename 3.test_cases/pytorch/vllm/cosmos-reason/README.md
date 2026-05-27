# NVIDIA Cosmos Reason — vLLM Inference on Amazon EKS / HyperPod

Online inference reference for NVIDIA's [Cosmos Reason](https://www.nvidia.com/en-us/ai/cosmos/)
physical-reasoning Vision-Language Model, served by [vLLM](https://github.com/vllm-project/vllm)
on Amazon EKS or SageMaker HyperPod EKS.

Cosmos Reason is the physical-reasoning member of NVIDIA's Cosmos World Foundation Model
platform. It generates chain-of-thought reasoning traces about safety, causality, object
interactions, and spatiotemporal dynamics in videos and images — the building block for
auto-labeling Synthetic Data Generation (SDG) pipelines
([as adopted by Uber per NVIDIA SIGGRAPH 2025](https://blogs.nvidia.com/blog/nemotron-cosmos-reasoning-enterprise-physical-ai/)),
Reasoning VLA models, and video Q&A workloads.

## Production Use Cases

| Use Case | Pattern | Example |
|----------|---------|---------|
| AV training data labeling | SDG critic loop — auto-label scenes with structured metadata (objects, hazards, weather) | [AV data captioning (NVIDIA SIGGRAPH 2025)](https://blogs.nvidia.com/blog/nemotron-cosmos-reasoning-enterprise-physical-ai/) |
| Video scene understanding | Short clip Q&A with chain-of-thought physical reasoning | Dashcam review, warehouse safety monitoring |
| Image VQA with physical reasoning | Single-image spatial/causal analysis | Content moderation, quality inspection |
| SDG critic / verifier | Judge whether Cosmos Predict outputs are physically plausible before entering training sets | Synthetic data filtering |
| RL reward model | Score trajectories for physical coherence as a reward signal in RLHF or model-based RL | Robotics policy training, AV planning |
| Offline RL annotation | Label trajectory data with reasoning traces for offline RL training | Decision Transformer reward labels |

## Two Paths
This test case provides **two parallel deployment paths**, both `kubectl`-only:

```
                      Cosmos Reason on EKS
                              |
              +---------------+---------------+
              |                               |
         kubernetes/                    hyperpod-eks/
   (vanilla EKS Deployment +     (HyperPod Inference Operator
    Service + HPA, upstream       InferenceEndpointConfig CRD,
    vllm/vllm-openai image)       AWS-managed vLLM DLC image)
              |                               |
              +---------------+---------------+
                              |
                       OpenAI-compatible
                       /v1/chat/completions
                              |
                          examples/
                  image VQA / video VQA / SDG auto-label
```

### Why two paths?

| Path | Purpose |
|------|---------|
| `kubernetes/` | Plain EKS users who already have a vLLM deployment. No HyperPod required. Vendor `vllm/vllm-openai` image. |
| `hyperpod-eks/` | HyperPod EKS clusters using the [Inference Operator](https://aws.amazon.com/blogs/architecture/unlock-efficient-model-deployment-simplified-inference-operator-setup-on-amazon-sagemaker-hyperpod/) — auto KEDA + Karpenter scale-to-zero, managed KV cache, intelligent routing. AWS-managed `vllm` Deep Learning Container. |

Both paths serve the **same** model with the **same** vLLM CLI args. Pick the path that
matches your platform.

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| EKS cluster | Kubernetes ≥ 1.28, GPU-capable |
| GPU node | One of: g5.* (A10G 24 GB), g6.12xlarge (4× L4 24 GB), g6e.* (L40S 48 GB), p4d/p5/p5e (H100/H200) |
| NVIDIA device plugin | `nvidia-device-plugin` DaemonSet running on GPU nodes |
| Hugging Face token | Required — Cosmos Reason models are gated on HF (NVIDIA Open Model License acceptance). [Request access](https://huggingface.co/nvidia/Cosmos-Reason1-7B) on the model card first. |
| For `hyperpod-eks/` only | HyperPod Inference Operator installed (`hyperpod-inference-operator` Helm chart, image `v3.1`). Comes with `sagemaker-hyperpod-cli` v3.7.1+ or one-click EKS add-on. |

## Models

This test case is parameterized via `${MODEL_ID}` in `env_vars.example`. Supported models:

| Model | Backbone | Min GPU memory | Recommended `--max-model-len` | Reasoning parser |
|-------|----------|---------------|------------------------------|------------------|
| **`nvidia/Cosmos-Reason1-7B`** (default for sm_8x GPUs) | Qwen2.5-VL | 24 GB (A10G/L4 OK) | 8192 (24 GB GPUs) / 32768 (40+ GB) | None — `<think>` is inline in `content` |
| **`nvidia/Cosmos-Reason2-2B`** | Qwen3-VL | 24 GB | 16384 | `--reasoning-parser qwen3` (separates `<think>` into `reasoning_content`) |
| **`nvidia/Cosmos-Reason2-8B`** (default for sm_9x GPUs) | Qwen3-VL | 32 GB (L40S/H100/H200) | 16384 | `--reasoning-parser qwen3` |

> [!NOTE]
> NVIDIA validates Cosmos Reason on H100 / GB200 / DGX Spark / Jetson AGX Thor.
> A10G (sm_86) and L4 (sm_89) are not on NVIDIA's official validated list but work in
> practice; this test case has been empirically validated on g5.8xlarge (1× A10G 24 GB).
> See **Test Results** in this directory's git history for benchmark data.

## Quick Start (vanilla EKS, vendor image)

```bash
# 1. Set environment
cp env_vars.example env_vars
# Edit env_vars — at minimum set HF_TOKEN
source env_vars

# 2. Validate required variables are set
for v in INSTANCE_TYPE MODEL_ID NAMESPACE TENSOR_PARALLEL_SIZE \
         MAX_MODEL_LEN GPU_MEMORY_UTILIZATION DTYPE INVOCATION_PORT \
         VLLM_IMAGE_VANILLA HF_TOKEN; do
  [ -z "${!v}" ] && echo "ERROR: \$$v is unset" && exit 1
done

# 3. Create HF token Secret
kubectl create secret generic hf-token \
  --from-literal=token=${HF_TOKEN}

# 4. Dry-run to confirm manifest renders correctly
envsubst < kubernetes/deployment.yaml | kubectl apply --dry-run=client -f -

# 5. Deploy
envsubst < kubernetes/deployment.yaml | kubectl apply -f -

# 6. Wait for /health (3-5 min for first launch — HF download + CUDA graph)
kubectl wait --for=condition=Ready pod -l app=cosmos-reason --timeout=10m

# 7. Port-forward and test
kubectl port-forward deploy/cosmos-reason 8000:8000 &
python3 examples/image_vqa.py --image examples/sample.jpg
```

> [!NOTE]
> For Reason2 models, omit `--system-prompt` (or pass `--system-prompt ""`). Reason2
> separates reasoning into the `reasoning_content` field natively via `--reasoning-parser qwen3`,
> so the default `<think>/<answer>` system prompt is unnecessary and can cause the model to
> produce minimal answers instead of a full description.

## Quick Start (HyperPod Inference Operator)

```bash
# 1. Set environment (same env_vars as above)
source env_vars

# 2. Create HF token Secret in the same namespace as the InferenceEndpointConfig
kubectl create secret generic hf-token \
  --from-literal=token=${HF_TOKEN}

# 3. Deploy via the operator
envsubst < hyperpod-eks/endpoint.yaml | kubectl apply -f -

# 4. Wait for the operator to mark the endpoint Ready
kubectl get inferenceendpointconfigs -w

# 5. Invoke through the operator-managed ALB (or via SageMaker SDK)
python3 examples/image_vqa.py \
  --endpoint $(kubectl get inferenceendpointconfig cosmos-reason -o jsonpath='{.status.endpointUrl}') \
  --image examples/sample.jpg
```

## Use Cases (verified examples in `examples/`)

| Script | What it does |
|--------|--------------|
| `examples/image_vqa.py` | Single-image visual question answering. Pattern: drive-recorder review, content moderation. |
| `examples/video_qa.py` | Short video clip Q&A. Pattern: AV scene understanding (Uber pattern). |
| `examples/auto_label.py` | Batch auto-labeling with `<think>...</think><answer>...</answer>` schema. Pattern: SDG critic loop, training data curation. |

## Configuration Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_ID` | `nvidia/Cosmos-Reason1-7B` | HF model ID. Override to `Cosmos-Reason2-8B` on L40S/H100. |
| `IMAGE_TAG` (kubernetes) | `vllm/vllm-openai:v0.11.0` | Upstream vLLM container. Pin to a specific version, never `:latest`. |
| `IMAGE_TAG` (hyperpod-eks) | `vllm:0.17-gpu-py312` (AWS DLC) | AWS-managed vLLM DLC. ECR path: `763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/vllm:0.17-gpu-py312` |
| `INSTANCE_TYPE` | `ml.g5.8xlarge` | A10G 24 GB. Other validated: `ml.g6.12xlarge` (4×L4), `ml.g6e.4xlarge` (1×L40S 48 GB). |
| `MAX_MODEL_LEN` | `8192` | Reduce for tight VRAM, increase to 32768 on 40 GB+ GPUs. Cosmos Reason native context is 256K — must reduce for non-H100 hardware. |
| `GPU_MEMORY_UTILIZATION` | `0.92` | vLLM target memory headroom. Reduce to `0.85` if OOM during CUDA graph capture. |
| `TENSOR_PARALLEL_SIZE` | `1` | Single GPU for 7B/2B. Set to `4` for 8B on g6.12xlarge (4× L4 24 GB). |
| `NAMESPACE` | `default` | Kubernetes namespace |
| `HF_TOKEN` | (none — required) | Hugging Face token with model access. Stored as a `Secret`. |

## Cleanup

```bash
# Vanilla EKS path
kubectl delete -f kubernetes/

# HyperPod Inference Operator path
kubectl delete -f hyperpod-eks/

# Both paths
kubectl delete secret hf-token
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `GatedRepoError: 401` on first deploy | No `HF_TOKEN` provided | Set `HF_TOKEN` env var, recreate `hf-token` Secret |
| `GatedRepoError: 403` after providing token | Token valid but account not on access list | Visit the model card on HF and click "Request Access". NVIDIA-gated models require accepting the NVIDIA Open Model License. |
| Pod stuck in `ContainerCreating` for 5+ min | Image pull (12 GB vendor / 23 GB AWS DLC) | Normal on first deploy. Check `kubectl describe pod` for "Pulling image". |
| Pod `Running` but `/health` returns 404 | vLLM still loading model + compiling CUDA graphs | Wait. First-launch is 3-8 min on 7B-class models. With `--enforce-eager` skips CUDA graphs (faster startup, slower inference). |
| `OutOfMemoryError` during CUDA graph capture | `--gpu-memory-utilization` too high or `--max-model-len` too long | Drop `--gpu-memory-utilization` to `0.85`, drop `--max-model-len` to `4096`. |
| Flash-Attention `headdim not multiple of 32` error (Reason2 / Qwen3-VL only) | vLLM internal fork of FA rejects Qwen3-VL ViT head dims | Do NOT set `VLLM_ATTENTION_BACKEND=FLASH_ATTN`. Let vLLM auto-pick. Issue [#27562](https://github.com/vllm-project/vllm/issues/27562) closed Apr 2026; v0.21.0+ is fixed. |
| Reason2 returns `<think>` text inline in `content` (not separate `reasoning_content`) | Missing `--reasoning-parser qwen3` | Add `--reasoning-parser qwen3` to vLLM args. Required for Reason2; not applicable to Reason1. |
| Latency low / TTFT high | `--enforce-eager` skips CUDA graphs | Remove `--enforce-eager` to enable graph compilation. Adds ~2 min to startup but ~30% throughput improvement. |

## References

- NVIDIA Cosmos: https://www.nvidia.com/en-us/ai/cosmos/
- Cosmos Reason1-7B model card: https://huggingface.co/nvidia/Cosmos-Reason1-7B
- Cosmos Reason2-8B model card: https://huggingface.co/nvidia/Cosmos-Reason2-8B
- Cosmos Reason2 repo (NVIDIA): https://github.com/nvidia-cosmos/cosmos-reason2
- vLLM: https://github.com/vllm-project/vllm
- vLLM Qwen3-VL recipe: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html
- AWS vLLM DLC repo: https://github.com/aws/deep-learning-containers
- HyperPod Inference Operator setup blog: https://aws.amazon.com/blogs/architecture/unlock-efficient-model-deployment-simplified-inference-operator-setup-on-amazon-sagemaker-hyperpod/
- HyperPod Inference Operator best practices: https://aws.amazon.com/blogs/machine-learning/best-practices-to-run-inference-on-amazon-sagemaker-hyperpod/
