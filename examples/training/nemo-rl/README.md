<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# NeMo RL GRPO Training with Fault Tolerance (NVRx) on Amazon EKS

## Overview

This example demonstrates fault-tolerant LLM fine-tuning on Amazon EKS using [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) with the [NVIDIA Resiliency Extension (NVRx)](https://github.com/NVIDIA/nvidia-resiliency-ext). It trains [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) using GRPO (Group Relative Policy Optimization) on a math reasoning dataset, with automatic checkpoint recovery on GPU failure.

**Duration:** ~15 minutes for 20 training steps on 2x g5.8xlarge
**GPU Memory:** Fits on A10G (24GB) using LoRA + CPU offload + vLLM sleep mode

### What is GRPO?

GRPO is a reinforcement learning algorithm for LLM alignment. Unlike SFT or DPO, GRPO generates its own training data on-the-fly:

1. **Generate** multiple responses per prompt using vLLM
2. **Score** responses with a reward function (math correctness)
3. **Compute advantages** relative to the group (better vs worse responses)
4. **Train** the model via LoRA to produce better responses

This is why the setup includes both **vLLM inference workers** and **DTensor training workers**.

### What is NVRx?

NVIDIA Resiliency Extension provides process-level fault tolerance:

- **ft_launcher**: Wraps the training process, monitors heartbeats, auto-restarts on failure (up to 3 times)
- **RankMonitorClient**: Sends heartbeats every 30s from training to ft_launcher
- **Straggler Detector**: Monitors GPU kernel timing across ranks to detect slow GPUs
- **Checkpoint integration**: Saves model state to FSx for resume-on-restart

## Architecture

```
kubectl apply → Amazon EKS → GPU Nodes (g5.8xlarge)
                                  ↓
                            KubeRay Operator
                                  ↓
                     ┌────────────┴────────────┐
                     │                         │
               Ray Head Pod              Ray Worker Pod
               (Node 1: A10G)            (Node 2: A10G)
                     │                         │
               ┌─────┴─────┐           ┌──────┴──────┐
               │ vLLM      │           │ vLLM        │
               │ (inference)│           │ (inference) │
               │ DTensor   │           │ DTensor     │
               │ (training) │           │ (training)  │
               └───────────┘           └─────────────┘
                     │                         │
                     └────── FSx Lustre ───────┘
                          (shared storage)
```

### Resiliency Stack

```
┌─────────────────────────────────────────────────┐
│ Layer 3: Application (NeMo RL + NVRx)           │
│  ft_launcher: fault-tolerant process launcher   │
│  Heartbeat: 30s interval health signal          │
│  Checkpointing: configurable interval to FSx    │
│  Straggler Detection: GPU perf monitoring       │
├─────────────────────────────────────────────────┤
│ Layer 2: Orchestration (KubeRay + Ray)          │
│  RayJob CRD: declarative cluster lifecycle      │
│  backoffLimit: auto-retry with fresh cluster    │
│  shutdownAfterJobFinishes: auto-cleanup         │
├─────────────────────────────────────────────────┤
│ Layer 1: Infrastructure (Amazon EKS + GPU)      │
│  g5.8xlarge: 1x NVIDIA A10G (24GB)             │
│  EFA networking: low-latency inter-node         │
│  FSx Lustre: shared checkpoint storage          │
│  Device plugins: GPU + EFA auto-detected        │
└─────────────────────────────────────────────────┘
```

## Supported GPU Instances

| Instance | GPU | VRAM | CUDA SM | Notes |
|----------|-----|------|---------|-------|
| g5.8xlarge | A10G | 24GB | 8.6 | Workshop default (LoRA + CPU offload) |
| g6e.48xlarge | L40S | 48GB | 8.9 | Larger models (Ministral-8B) |
| p5.48xlarge | H100 | 80GB | 9.0 | Production scale (8 GPUs/node, EFA) |

## Prerequisites

1. **Amazon EKS cluster** with GPU nodes (2x g5.8xlarge minimum)
2. **KubeRay operator** installed
3. **Amazon FSx for Lustre** with a PersistentVolumeClaim (`fsx-grpo-pvc`)
4. **NVIDIA device plugin** and **EFA device plugin** (for EFA-enabled instances)
5. **kubectl** configured for your cluster

## 0. Build the Container Image

The Dockerfile builds on CPU (no GPU required) and supports multiple GPU architectures.

```bash
# Multi-arch build (~40 min on c5.4xlarge)
docker build -f Dockerfile -t nemo-rl-workshop:latest .

# Single-arch build for faster iteration (~25 min)
docker build -f Dockerfile --build-arg TORCH_CUDA_ARCH_LIST="8.6" -t nemo-rl-workshop:g5 .

# Verify
docker run --rm nemo-rl-workshop:latest \
  python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
# Expected: PyTorch 2.9.0, CUDA 12.9
```

### Build Args

| Build arg | Default | Purpose |
|-----------|---------|---------|
| `TORCH_CUDA_ARCH_LIST` | `"8.6 8.9 9.0"` | Target GPU architectures |
| `NRL_GIT_REF` | pinned commit | NeMo RL git commit/branch |
| `MAX_JOBS` | auto | Limit parallel compilation (prevent OOM) |
| `OFI_NCCL_VERSION` | `v1.18.0` | aws-ofi-nccl version for EFA |

### Push to Amazon ECR

```bash
aws ecr-public create-repository --repository-name nemo-rl-workshop --region us-east-1

aws ecr-public get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin public.ecr.aws/<your-alias>

IMAGE_TAG=workshop-$(date +%Y%m%d)
docker tag nemo-rl-workshop:latest public.ecr.aws/<your-alias>/nemo-rl-workshop:$IMAGE_TAG
docker push public.ecr.aws/<your-alias>/nemo-rl-workshop:$IMAGE_TAG
```

## 1. Install KubeRay and Prepare FSx

```bash
# Install KubeRay operator
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator \
  --namespace kuberay-system --create-namespace

# Verify
kubectl get pods -n kuberay-system
# kuberay-operator-xxxx   1/1   Running

# Copy scripts to FSx (run from a pod with FSx mounted)
kubectl run fsx-setup --image=<your-image> --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"fsx","persistentVolumeClaim":{"claimName":"fsx-grpo-pvc"}}],
    "containers":[{"name":"c","image":"<your-image>",
    "command":["sleep","120"],
    "volumeMounts":[{"name":"fsx","mountPath":"/shared"}]}]}}'

kubectl wait --for=condition=Ready pod/fsx-setup --timeout=60s

kubectl exec fsx-setup -- mkdir -p \
  /shared/nvrx-demo/patches \
  /shared/nvrx-demo/scripts \
  /shared/nvrx-demo/checkpoints

kubectl cp patches/patch_nvrx_features.py default/fsx-setup:/shared/nvrx-demo/patches/
kubectl cp scripts/run_grpo_nvrx.py default/fsx-setup:/shared/nvrx-demo/scripts/
kubectl cp scripts/rayjob_entrypoint.sh default/fsx-setup:/shared/nvrx-demo/scripts/
kubectl cp scripts/evaluate_before_after.py default/fsx-setup:/shared/nvrx-demo/scripts/
kubectl exec fsx-setup -- chmod +x /shared/nvrx-demo/scripts/rayjob_entrypoint.sh

kubectl delete pod fsx-setup --now
```

## 2. Pre-cache Dataset and Model

Download the math dataset (~1.5GB) and model weights (~3GB) to FSx before training:

```bash
kubectl apply -f kubernetes/dataset-download-job.yaml

# Monitor progress
kubectl logs -f job/dataset-download

# Verify completion
kubectl get job dataset-download
# NAME               COMPLETIONS   DURATION   AGE
# dataset-download   1/1           3m42s      4m
```

## 3. Deploy the RayJob

```bash
kubectl apply -f kubernetes/rayjob.yaml
```

This creates:
- A **RayCluster** with 1 head + 1 worker (2 A10G GPUs total)
- A **Ray job** that runs the training entrypoint
- Automatic cleanup after the job finishes

## 4. Monitor Training

```bash
# Watch RayJob status
watch kubectl get rayjob grpo-g5-qwen-nvrx

# Stream head pod logs
HEAD=$(kubectl get pods -l ray.io/node-type=head \
  -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f $HEAD -c ray-head
```

**What to look for in the logs:**

```
=== NeMo RL GRPO Training via RayJob ===        # Entrypoint started
[NVRx patch] OK: added is_async to ...          # NVRx patches applied
[NVRx] GPU 0: NVIDIA A10G, 23.9GB, SM 8.6: OK  # GPU health check
[NVRx] Heartbeat thread started (interval=30s)   # Heartbeat active
[NVRx] Straggler detector initialized on rank 0  # Straggler detection
========================= Step 1/20 ===========  # Training started
========================= Step 5/20 ===========  # Checkpoint saved
========================= Step 20/20 ==========  # Done!
ft_launcher Exit=0                                # Clean exit
```

## 5. Fault Injection Demo

After training reaches Step 6+ (past first checkpoint):

```bash
# Kill the training process on the worker
WORKER=$(kubectl get pods -l ray.io/node-type=worker \
  -o jsonpath='{.items[0].metadata.name}')
kubectl exec $WORKER -c ray-worker -- \
  bash -c 'kill -9 $(pgrep -f "run_grpo" | head -1)'
```

**Recovery sequence:**

1. ft_launcher detects heartbeat timeout (~30 seconds)
2. Worker group is terminated
3. ft_launcher restarts worker group (attempt 1 of 3)
4. NeMo RL loads checkpoint from FSx
5. Training resumes from last checkpoint
6. Training completes 20/20 steps

## 6. Evaluate Before vs After

```bash
kubectl exec -it $HEAD -c ray-head -- python3 \
  /shared/nvrx-demo/scripts/evaluate_before_after.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint-dir /shared/nvrx-demo/checkpoints
```

Expected output:

```
Category                   Before    After   Change
-------------------------------------------------------
Math (arithmetic)            PASS     PASS     SAME
Math (word problem)          FAIL     PASS IMPROVED
Math (fractions)             FAIL     PASS IMPROVED
Reasoning                    PASS     PASS     SAME
Instruction following        PASS     PASS     SAME
Math (algebra)               FAIL     PASS IMPROVED
-------------------------------------------------------
TOTAL                         3/6      6/6
```

## 7. Clean Up

```bash
kubectl delete rayjob grpo-g5-qwen-nvrx
```

## Configuration

The RayJob manifest has configurable environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CLEAR_CHECKPOINTS` | `1` | `1` = wipe checkpoints for fresh demo, `0` = test resume |
| `GRPO_MAX_STEPS` | `20` | Number of training steps (reduce to `10` for faster demos) |
| `CHECKPOINT_PERIOD` | `10` | Save checkpoint every N steps |

## Files

| File | Purpose |
|------|---------|
| `README.md` | This file |
| `Dockerfile` | Multi-arch container image (g5 A10G / g6e L40S / p5 H100) |
| `kubernetes/rayjob.yaml` | RayJob manifest for EKS deployment |
| `kubernetes/dataset-download-job.yaml` | Pre-cache dataset + model to FSx |
| `scripts/rayjob_entrypoint.sh` | Training entrypoint (runs on FSx) |
| `scripts/run_grpo_nvrx.py` | NVRx wrapper (heartbeat + GPU health check) |
| `scripts/evaluate_before_after.py` | Before/after training evaluation |
| `patches/patch_nvrx_features.py` | Runtime patches for NVRx features |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Pods stuck Pending | Check GPU/EFA device plugins: `kubectl get pods -n kube-system -l app=nvidia-device-plugin` |
| RayJob FAILED immediately | Check entrypoint script on FSx: `kubectl exec $HEAD -- cat /shared/nvrx-demo/scripts/rayjob_entrypoint.sh` |
| vLLM OOM | Model too large for GPU. Use Qwen2.5-1.5B on A10G, not larger models |
| Checkpoint dir empty | Check `CLEAR_CHECKPOINTS` env var and FSx mount: `kubectl exec $HEAD -- df -h /shared` |
| Slow first run | Normal — downloading model/dataset to FSx. Run the dataset download job first. |

## References

- [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL)
- [NVIDIA Resiliency Extension (NVRx)](https://github.com/NVIDIA/nvidia-resiliency-ext)
- [KubeRay Operator](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html)
- [Amazon FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html)
- [EFA on EKS](https://docs.aws.amazon.com/eks/latest/userguide/node-efa.html)
