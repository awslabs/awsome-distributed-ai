<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Multi-node GRPO Training with NVIDIA NeMo RL on Amazon EKS

## Overview

Multi-node [GRPO](https://arxiv.org/abs/2402.03300) (Group Relative Policy Optimization) training using [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) on Amazon EKS. NeMo RL is built on [Megatron](https://github.com/NVIDIA/Megatron-LM) for distributed training and tested across multiple GPU instance types including P5en (H200), P5 (H100), and G6E (L40S).

NeMo RL orchestrates training with [Ray](https://docs.ray.io/) for cluster management, [vLLM](https://github.com/vllm-project/vllm) for fast generation rollouts, and PyTorch [DTensor](https://pytorch.org/docs/stable/distributed.tensor.html) for distributed policy optimization.

### Verified Instance Types

| Instance | GPU | GPUs/Node | EFA | Transport | Tested | Result |
|----------|-----|-----------|-----|-----------|--------|--------|
| **p5en.48xlarge** | NVIDIA H200 141GB | 8 | 16x 200Gbps | NCCL RDMA | 2x nodes | 50/50 steps, 34%->40% (+13 problems) |
| **p5.48xlarge** | NVIDIA H100 80GB | 8 | 32x 100Gbps | NCCL RDMA | 2x nodes | 5/5 steps, 47.87s/step |
| **g6e.8xlarge** | NVIDIA L40S 48GB | 1 | None | Socket | 2x nodes | 20/20 steps + fault recovery |

### Training Results (Nemotron-Mini-4B-Instruct)

| Metric | Before | After GRPO | Change |
|--------|--------|------------|--------|
| Math accuracy (200 problems) | 34% | 40% | **+6pp (+13 problems)** |
| Training | -- | 50 steps, LoRA | ~30 min on 2x P5en |

## Architecture

```
+------------------------------------------------------------------+
|  Amazon EKS Cluster (2x GPU nodes)                               |
|                                                                  |
|  +-----------------------------+  +----------------------------+ |
|  |  Head Node                  |  | Worker Node                | |
|  |                             |  |                            | |
|  |  Ray Head + GCS             |  | Ray Worker                 | |
|  |  vLLM Generation Workers    |  | vLLM Generation Workers   | |
|  |  DTensor Policy Workers     |  | DTensor Policy Workers    | |
|  |  GRPO Orchestrator          |  |                            | |
|  +-------------|---------------+  +-------------|---------------+ |
|                |     EFA RDMA (if available) / Socket       |    |
|                +--------------------------------------------+    |
+------------------------------------------------------------------+
```

The architecture automatically adapts to the instance type:
- **P5/P5en**: EFA RDMA with NCCL for GPU-direct network transfers
- **G5/G6E**: TCP socket transport (no EFA required)

## Prerequisites

- Amazon EKS cluster with GPU nodes
- [KubeRay operator](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html) installed
- Shared storage (FSx Lustre or EFS) for checkpoints and model weights
- For P5/P5en: EFA device plugin and VPC CNI with prefix delegation

## Data Preparation

This test case uses the Goldilocks math problems dataset for GRPO training. The dataset is a synthetic set of math word problems with deterministic answers, generated to be "just right" difficulty for a Nemotron-Mini-4B model — hard enough that the untrained model fails, but solvable with proper reasoning.

### Dataset format

Each example is a JSONL line:

```json
{
  "prompt": "A farmer has 17 chickens...",
  "answer": "42",
  "difficulty": "medium",
  "category": "arithmetic"
}
```

### Generating the dataset

The dataset is expected at `/fsx/goldilocks/train.jsonl` and `/fsx/goldilocks/test.jsonl` (mounted via FSx). To generate your own:

1. See the sibling generator: `3.test_cases/megatron/nemo-rl/data-prep/generate_goldilocks_data_designer.py`
2. Or use any math problem dataset with `prompt` and `answer` fields in JSONL format.

The `kubernetes/rayjob-grpo.yaml` manifest mounts `/fsx/goldilocks/` — update the `goldilocksPath` env var in the manifest if your path differs.

## Quick Start

### 1. Build the container image

```bash
docker buildx build --platform linux/amd64 \
  -f Dockerfile \
  -t <your-registry>/nemo-rl-workshop:latest \
  --push .
```

### 2. Deploy training

```bash
# Edit kubernetes/rayjob-grpo.yaml to set your image, namespace, and storage
kubectl apply -f kubernetes/rayjob-grpo.yaml
```

### 3. Monitor training

```bash
# Watch pods
kubectl get pods -n <namespace> -w

# Check training progress
SUB=$(kubectl get pods -n <namespace> --no-headers | grep grpo-nvrx-rayjob- | grep -v head | grep -v worker | awk '{print $1}' | head -1)
kubectl logs $SUB -n <namespace> -f | grep "Avg Filtered Reward"
```

### 4. Evaluate

```bash
python3 eval_nemotron_goldilocks.py \
  --model nvidia/Nemotron-Mini-4B-Instruct \
  --dataset /path/to/goldilocks/train.jsonl \
  --checkpoint-dir /path/to/phase2-checkpoints \
  --output results.json
```

## Instance-Specific Configuration

### P5en (H200) -- Recommended for production

```yaml
# Environment variables for P5en with EFA
env:
  - name: NCCL_NET_PLUGIN
    value: "ofi"
  - name: NCCL_TUNER_PLUGIN
    value: "ofi"
  - name: NCCL_NVLS_ENABLE
    value: "0"        # Required: NVLS bug on H200
  - name: NCCL_CUMEM_ENABLE
    value: "0"        # Required: cuMem penalty on H200
  - name: FI_EFA_USE_DEVICE_RDMA
    value: "1"
resources:
  limits:
    nvidia.com/gpu: 8
    vpc.amazonaws.com/efa: 16
```

### P5 (H100)

```yaml
env:
  - name: NCCL_NET_PLUGIN
    value: "ofi"
  - name: NCCL_TUNER_PLUGIN
    value: "ofi"
  - name: NCCL_SOCKET_IFNAME
    value: "^lo,docker,veth,eni"
resources:
  limits:
    nvidia.com/gpu: 8
    vpc.amazonaws.com/efa: 32
```

### G6E (L40S) -- No EFA required

```yaml
env:
  - name: NCCL_NET_PLUGIN
    value: "none"      # Socket transport
resources:
  limits:
    nvidia.com/gpu: 1  # 1 GPU per node on g6e.8xlarge
```

## NVRx Fault Tolerance

The training setup includes NVIDIA NVRx resiliency features:
- **GPU health check** at startup (CudaHealthCheck)
- **Async checkpointing** to FSx Lustre (save_period=25)
- **Checkpoint resume** -- training restarts from the latest checkpoint
- **RayJob retry** -- `backoffLimit: 2` automatically retries on failure

Tested: kill -9 at step 27 -> RayJob retry -> resume from step 25 checkpoint -> complete all 50 steps.

## NVIDIA Dynamo Inference

After training, serve the model with NVIDIA Dynamo disaggregated inference:
- Cross-node prefill/decode with NIXL LIBFABRIC over EFA RDMA
- 8 data-parallel workers per node (16 GPUs total)

See [awsome-inference/dynamo-inference](https://github.com/aws-samples/awsome-inference/tree/main/2.projects/dynamo-inference) for deployment details.

## Known Issues

| Issue | Instance | Workaround | Upstream |
|-------|----------|------------|----------|
| NCCL Ring deadlock >8M elements with Ray | P5en (H200) | `NCCL_ALGO=Tree` or broadcast-via-allreduce | [NVIDIA/nccl#2024](https://github.com/NVIDIA/nccl/issues/2024) |
| cuMem import penalty (3660ms) | P5en (H200) | `NCCL_CUMEM_ENABLE=0` | [NVIDIA/nccl#1749](https://github.com/NVIDIA/nccl/issues/1749) |
| NVLS rank ordering bug | P5en (H200) | `NCCL_NVLS_ENABLE=0` | [NVIDIA/nccl#1906](https://github.com/NVIDIA/nccl/issues/1906) |
| Ray removes CUDA_VISIBLE_DEVICES override | All | Patched in NeMo RL | [ray-project/ray#61073](https://github.com/ray-project/ray/issues/61073) |
| NVRx straggler detection deadlocks with DTensor | All | Disabled (process_group fix pending) | [NVIDIA/nvidia-resiliency-ext#277](https://github.com/NVIDIA/nvidia-resiliency-ext/issues/277) |

## SBOM

The Software Bill of Materials is generated at image build time inside the container at `/SBOM.txt` and `/THIRD-PARTY-LICENSES`. To extract it from a built image:

```bash
docker run --rm <your-image> cat /SBOM.txt
docker run --rm <your-image> pip list --format=freeze
docker run --rm <your-image> dpkg-query -W -f '${Package},${Version}\n'
```

Key versions:
- PyTorch 2.9.0+cu129, CUDA 12.9
- NCCL: 2.27.5 (pip/PyTorch wheel) / 2.26.5 (system dpkg, from base image)
- NeMo RL 0.5.0rc0, Ray 2.49.2, vLLM (in worker venvs), NVRx 0.4.1
- EFA installer 1.47.0, Libfabric 2.4.0, aws-ofi-nccl 1.18.0, GDRCopy 2.5.1

> **Note on NCCL versions**: The container has two NCCL installations. The PyTorch
> pip wheel bundles NCCL 2.27.5 (`nvidia-nccl-cu12`), which is what PyTorch actually
> uses at runtime. The system package `libnccl2` (2.26.5) comes from the CUDA base
> image and is used only for header compatibility during aws-ofi-nccl compilation.

## Container Image

| Image | Size | Architecture |
|-------|------|-------------|
| Build your own with the included `Dockerfile` | ~24 GB (compressed) | linux/amd64 |

## References

- [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL)
- [NVIDIA NVRx](https://github.com/NVIDIA/nvidia-resiliency-ext) -- [PR #278](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/278) (process_group fix)
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) -- [PR #7390](https://github.com/ai-dynamo/dynamo/pull/7390) (local path fix)
- [AWS EFA](https://aws.amazon.com/hpc/efa/)
- [AWS EFA Installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html)
- [GDRCopy](https://github.com/NVIDIA/gdrcopy)
