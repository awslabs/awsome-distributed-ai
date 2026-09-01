# OpenRLHF GRPO on Amazon EKS (HyperPod) — Multilingual Language Compliance

This recipe trains [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
(a 20B MoE model) with **GRPO** (Group Relative Policy Optimization) using
[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) on Amazon EKS with HyperPod.

The task is **multilingual language compliance**: the model must reason AND answer
in the same language as the user's question (English, French, German, Spanish,
Italian). A custom reward function scores language adherence.

## Architecture

OpenRLHF uses a **Non-Hybrid Engine** architecture: vLLM inference and DeepSpeed
training run on **separate** GPU nodes. This avoids the CPU memory pressure from
vLLM's sleep/wake backup mechanism that causes OOM when colocated with DeepSpeed
`adam_offload` on memory-constrained nodes.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    6× g5.12xlarge (4× A10G 24GB each)              │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────────────────────────────────┐ │
│  │   Node 1     │  │              Nodes 2-5                      │ │
│  │              │  │                                              │ │
│  │  Ray Head    │  │  GPU Workers (160Gi, 4 GPU, 1 EFA each)     │ │
│  │  (8Gi, 0 GPU)│  │                                              │ │
│  │      +       │  │  ┌────────────┐  ┌────────────────────────┐ │ │
│  │  GPU Worker  │  │  │  Worker 1  │  │  Workers 2-4           │ │ │
│  │  (160Gi,     │  │  │  vLLM TP=4 │  │  DeepSpeed ZeRO-3     │ │ │
│  │   4 GPU,     │  │  │  (inference)│  │  (training, 12 GPUs)  │ │ │
│  │   1 EFA)     │  │  └────────────┘  └────────────────────────┘ │ │
│  └──────────────┘  └──────────────────────────────────────────────┘ │
│                                                                     │
│  Total: 20 GPUs visible to Ray (head has num-gpus=0)               │
│  vLLM: 1 engine × TP=4 = 4 GPUs (dedicated inference node)        │
│  Training: 4 nodes × 4 GPUs = 16 GPUs (DeepSpeed ZeRO-3)          │
└─────────────────────────────────────────────────────────────────────┘
```

### Comparison with veRL

| Component | OpenRLHF | veRL |
|-----------|----------|------|
| Training framework | DeepSpeed ZeRO-3 | FSDP2 |
| Memory offload | `--adam_offload` (optimizer to CPU) | `offload_policy=True` (params + optimizer) |
| Inference engine | vLLM (dedicated node) | vLLM (inline, shared GPUs) |
| GPU layout | Separate vLLM + training nodes | All GPUs shared |
| Checkpoint format | HuggingFace directly (`--save_hf_ckpt`) | FSDP shards → merge step |
| Reward function API | `reward_func(queries, prompts, labels)` | `compute_score(data_source, solution_str, ground_truth)` |
| Orchestration | Ray | Ray |

## Training Results

Trained for 60+ steps on 6× g5.12xlarge (4× A10G 24GB per node). HuggingFace
checkpoints saved at steps 20 and 40. Selected metrics:

| Step | Reward | Policy Loss | Return | Gen Len | LR |
|------|--------|-------------|--------|---------|----|
| 1 | 5.69 | 2.66e7 | 0 | 256 | 0 |
| 5 | 5.59 | 2.22e6 | -0.09 | 253 | 9.99e-7 |
| 10 | 5.50 | 0 | 0 | 256 | 9.77e-7 |
| 15 | 5.78 | -0.07 | 0.09 | 250 | 9.25e-7 |
| 20 | 5.97 | -0.16 | 0.22 | 256 | 8.46e-7 |
| 30 | 5.78 | -0.07 | 0.09 | 256 | 7.03e-7 |
| 40 | 5.59 | -0.03 | 0.04 | 256 | 4.20e-7 |
| 50 | 5.69 | -0.07 | 0.09 | 252 | 2.54e-7 |
| 60 | 5.69 | 1.63e3 | 0.04 | 253 | 1.03e-7 |

- **~2.3 min/step** (60 steps in ~2h38m)
- **Reward range**: 4.88–5.97 (max 6.0 = perfect on all criteria)
- **No OOM** during training — 80GB/node adam_offload with 160Gi pod limit
- **Checkpoint save** succeeded at steps 20 and 40 (39GB HF format each)
- **vLLM throughput**: ~25s per batch of 16 prompts (TP=4 on dedicated node)

## Hardware Requirements

Tested on:
- **6× ml.g5.12xlarge** (HyperPod): 4× NVIDIA A10G 24GB per node, 1× EFA
- Ray Head: 8Gi memory, 4 CPU, `num-gpus=0` (co-located with one worker)
- 5 GPU Workers: 160Gi memory, 16 CPU, 4 GPU, 1 EFA each
- 1 worker runs vLLM inference (TP=4), 4 workers run DeepSpeed training (16 GPUs)
- FSx for Lustre: 1.2 TB shared storage

Also supports p5en.48xlarge (8× H100 80GB) — uncomment the p5en block in
`env_vars.example`.

### Memory Budget (g5.12xlarge)

| Component | Per Node | Notes |
|-----------|----------|-------|
| Node allocatable | 172Gi | After system pods (~4Gi) |
| Worker pod limit | 160Gi | Leaves 12Gi for head co-location |
| adam_offload | ~80GB | 320GB / 16 GPUs × 4 GPUs/node |
| Model partitions | ~10GB | ZeRO-3 sharded params + gradients |
| Overhead | ~10GB | Python, Ray, NCCL buffers |
| **Available for checkpoints** | **~60GB** | DS checkpoint writes + HF gather |

> **Why Non-Hybrid?** The Hybrid Engine (`--colocate_all_models`) backs up vLLM
> model weights to CPU RAM during sleep. For a 20B model, this adds ~40GB to each
> node's CPU memory. Combined with `adam_offload` (~107GB/node at 12 GPUs), total
> exceeds the 160Gi pod limit. The Non-Hybrid Engine avoids this entirely by
> running vLLM on a dedicated node.

## Prerequisites

1. An EKS cluster with HyperPod managed node groups (g5.12xlarge or p5en.48xlarge)
2. KubeRay operator installed
3. FSx for Lustre persistent volume (`fsx-claim`)
4. `kubectl` configured for the cluster
5. Docker + ECR access for building/pushing the image
6. The SFT-merged model at `/fsx/models/gpt-oss-20b-sft-merged/` (GRPO starts from SFT, not base)

## Quick Start

### 1. Configure Environment

```bash
cd hyperpod-eks/rlvr
cp setup/env_vars.example setup/env_vars
# Edit setup/env_vars with your cluster details, HF token, etc.
source setup/env_vars
```

### 2. Build and Push Docker Image

```bash
# Authenticate to ECR
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${REGISTRY}

# Create repository (first time only)
aws ecr create-repository --repository-name ${IMAGE} --region ${AWS_REGION} || true

# Build image
docker build -t ${REGISTRY}${IMAGE}:${TAG} -f ../../Dockerfile ../..

# Push to ECR
docker push ${REGISTRY}${IMAGE}:${TAG}
```

### 3. Deploy Ray Cluster

```bash
# Substitute environment variables into the manifest
envsubst < setup/raycluster.yaml | kubectl apply -f -

# Wait for all 6 pods (1 head + 5 workers) to be ready
kubectl get pods -w -l ray.io/is-ray-node=yes

# Port-forward Ray dashboard
kubectl port-forward svc/rayml-efa-head-svc 8265:8265 &
```

### 4. Prepare Data

```bash
bash setup/load_data_gptoss.sh
```

This downloads HuggingFaceH4/Multilingual-Thinking, formats it as JSONL with
chat-template messages + language labels, and saves to FSx.

### 5. Copy Reward Function to FSx

The reward function must be accessible on all Ray workers via shared storage:

```bash
HEAD_POD=$(kubectl get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
kubectl exec ${HEAD_POD} -- mkdir -p /fsx/openrlhf/reward
kubectl cp recipe/language_reward.py ${HEAD_POD}:/fsx/openrlhf/reward/language_reward.py
```

### 6. Launch GRPO Training

```bash
bash recipe/run_gptoss_grpo.sh
```

Monitor progress:
```bash
# Ray dashboard (after port-forward)
open http://localhost:8265

# Ray job logs
ray job logs <JOB_ID> --address http://localhost:8265 --follow
```

### 7. Evaluate

OpenRLHF saves checkpoints in HuggingFace format directly — **no conversion
step needed** (unlike veRL which requires FSDP→HF merging).

```bash
# SSH into a worker pod with GPUs
kubectl exec -it <worker-pod> -- bash

# Run evaluation (uses the step 40 HF checkpoint by default)
bash recipe/evaluate_gptoss.sh
```

## File Structure

```
openrlhf/
├── Dockerfile                              # OpenRLHF v0.9.0 + vLLM 0.11.0 + EFA
├── buildspec.yml                           # AWS CodeBuild spec for ECR push
└── hyperpod-eks/
    └── rlvr/
        ├── setup/
        │   ├── env_vars.example            # Cluster config (g5 + p5en templates)
        │   ├── raycluster.yaml             # KubeRay manifest (6-node layout)
        │   └── load_data_gptoss.sh         # Data preparation script
        └── recipe/
            ├── run_gptoss_grpo.sh          # Training launcher (ray job submit)
            ├── language_reward.py           # Custom reward function
            ├── evaluate_gptoss.py          # 50-question vLLM evaluation
            └── evaluate_gptoss.sh          # Evaluation wrapper
```

## Memory Optimization for g5.12xlarge (24GB GPUs)

Running a 20B MoE model on 24GB GPUs is tight. Key optimizations:

| Optimization | Flag | Effect |
|-------------|------|--------|
| ZeRO Stage 3 | `--zero_stage 3` | Shard params + grads + optimizer across all GPUs |
| Adam offload | `--adam_offload` | Move optimizer states to CPU (~80GB/node) |
| Gradient checkpointing | `--gradient_checkpointing` | Trade compute for memory |
| Non-Hybrid Engine | (default, no `--colocate_all_models`) | vLLM on dedicated node, no CPU memory competition |
| No CUDA graphs | `--enforce_eager` | Avoid graph workspace memory |
| No KL penalty | `--init_kl_coef 0` | Skip ref model forward pass |
| Dedicated vLLM | `--vllm_gpu_memory_utilization 0.8` | High utilization on inference-only node |
| No critic | (GRPO is critic-free) | No second model in memory |
| bf16 | `--bf16` | Half precision throughout |
| MoE balancing | `--aux_loss_coef 0.01` | Load balancing loss for MoE experts |

If you hit OOM, try:
1. Add more training nodes to reduce per-node `adam_offload` burden
2. Reduce `--rollout_batch_size`
3. Reduce `--generate_max_len` (e.g., 256 → 128)

## Custom Reward Function

The reward function (`language_reward.py`) uses the OpenRLHF API:

```python
def reward_func(queries, prompts, labels, **kwargs):
    # queries: decoded prompt+response strings (with special tokens)
    # prompts: original prompt strings (after chat template)
    # labels:  ground truth language codes from --label_key
    return {
        "rewards": torch.tensor([...]),   # per-sample rewards (1-d tensor!)
        "scores": torch.tensor([...]),    # normalized [0,1] for filtering
        "extra_logs": {                   # logged to console/wandb
            "accuracy": torch.tensor([...]),  # MUST be 1-d tensors (not 0-d!)
        },
    }
```

> **Important**: Values in `extra_logs` must be 1-d tensors (`torch.tensor([v])`)
> not 0-d scalars (`torch.tensor(v)`). OpenRLHF concatenates them with
> `torch.cat()`, which fails on 0-d tensors.

Scoring (identical logic to the veRL reward function):
- Answer in correct language: **+5.0** / **-5.0**
- Reasoning in correct language: **+1.5** / **-1.5**
- Brief final answer (≤2 sentences): **+0.5** / **-1.0**

## Key Discoveries

1. **Non-Hybrid Engine is required for 20B models on 24GB GPUs with adam_offload.**
   The Hybrid Engine's vLLM sleep backup adds ~40GB CPU RAM per node, exceeding
   the 160Gi pod limit when combined with adam_offload.

2. **Ray head must have `num-gpus: "0"`** in `rayStartParams`. Otherwise, Ray may
   schedule NCCL training processes on the head node, which has no EFA device,
   causing `NET/OFI Received an invalid tag` errors.

3. **`RAY_memory_monitor_refresh_ms=0`** must be set in the raycluster.yaml
   (not via runtime-env). DeepSpeed ZeRO-3 init temporarily uses most of the
   host RAM; Ray's OOM killer terminates processes before init completes.

4. **NumPy/cv2 fix required**: vLLM 0.11.0 pulls `opencv-python-headless` which
   crashes with NumPy 2.4+ from NGC. Fix: downgrade numpy to <2.3 and remove cv2.
   This is handled in the Dockerfile.

5. **Checkpoint sizing**: Each DeepSpeed checkpoint is ~234GB for 20B params across
   16 GPUs. Each HF checkpoint is ~39GB. Plan FSx capacity accordingly
   (`max_ckpt_num × 234GB + max_ckpt_num × 39GB` per save cycle).

## Software Versions

| Component | Version |
|-----------|---------|
| OpenRLHF | 0.9.0 |
| vLLM | 0.11.0 |
| DeepSpeed | (bundled with OpenRLHF) |
| Ray | (bundled with OpenRLHF) |
| CUDA | 12.8 (NGC 25.02-py3) |
| PyTorch | 2.8.0 |
| EFA installer | 1.43.3 |

## Troubleshooting

**vLLM "No available memory for KV cache"**
- This means `gpu_memory_utilization` is too high for the available GPU memory
- With Non-Hybrid, vLLM has dedicated GPUs — 0.8 should work
- If using Hybrid Engine, reduce to 0.4–0.5

**OOM during training (GPU)**
- Ensure `--adam_offload` is set (moves optimizer to CPU)
- Ensure `--gradient_checkpointing` is set
- Reduce `--micro_train_batch_size` to 1

**OOM during training (CPU / pod OOMKilled)**
- Add more training nodes to reduce per-node adam_offload burden
- With 16 training GPUs: ~80GB/node → safe at 160Gi
- With 12 training GPUs: ~107GB/node → tight, checkpoint saves may OOM
- Check: `kubectl describe pod <pod> | grep -A5 "Last State"`

**OOM during checkpoint save**
- The HF save gathers all weights to rank 0 (~39GB peak)
- Ensure per-node adam_offload leaves ≥50GB headroom in the pod limit
- If needed, increase `--max_ckpt_num` to allow cleanup of old checkpoints

**NCCL/EFA errors on head node**
- Verify `num-gpus: "0"` is set in head's `rayStartParams`
- The head node may not have EFA configured for GPU traffic
- All NCCL training must run on worker nodes (which have EFA)

**Ray OOM killer terminates processes during init**
- Set `RAY_memory_monitor_refresh_ms: "0"` in raycluster.yaml env vars
- This must be in the pod spec, not passed via runtime-env

**FSx disk full during checkpoint save**
- DeepSpeed checkpoints are large (~234GB for 20B model)
- Clean old DS checkpoints: `rm -rf .../ckpt/_actor/global_step*/`
- HF checkpoints (39GB) are much smaller and sufficient for evaluation
- Set `--max_ckpt_num 2` to reduce disk usage

## References

- [OpenRLHF Documentation](https://openrlhf.readthedocs.io/en/latest/)
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)
- [GRPO Paper (DeepSeek-R1)](https://arxiv.org/abs/2501.12948)
- [HuggingFaceH4/Multilingual-Thinking Dataset](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)
- [openai/gpt-oss-20b Model](https://huggingface.co/openai/gpt-oss-20b)
