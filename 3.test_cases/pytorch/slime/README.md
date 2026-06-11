# Scaling Reinforcement Learning with SLIME on Amazon SageMaker HyperPod EKS

This post demonstrates how to run large-scale reinforcement learning (RL) post-training
for large language models (LLMs) using [SLIME](https://github.com/THUDM/slime) on
[Amazon SageMaker HyperPod](https://aws.amazon.com/sagemaker/hyperpod/) with Amazon EKS
orchestration. We walk through the end-to-end workflow: building a container image,
preparing data, deploying a multi-node Ray cluster, converting model checkpoints, and
launching GRPO training on NVIDIA H100 GPUs interconnected with Elastic Fabric Adapter
(EFA).

## Introduction

Reinforcement learning from human feedback (RLHF) and its more recent variants --
GRPO, DAPO, Reinforce++ -- have become the dominant paradigm for aligning large
language models after supervised fine-tuning. As model sizes scale beyond 10B parameters,
the computational demands of RL post-training grow substantially: each training step
requires online rollout generation (inference), advantage estimation, and policy
gradient updates, all coordinated across dozens of GPUs.

[**SLIME**](https://github.com/THUDM/slime) (an LLM post-training framework for RL
Scaling) addresses this challenge by natively integrating two best-of-breed backends:

- **[SGLang](https://github.com/sgl-project/sglang)** for high-throughput rollout
  generation (inference), providing RadixAttention, continuous batching, and tensor
  parallelism.
- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** for scalable distributed
  training, with support for TP, PP, CP, EP, and ZeRO-style sharding.

SLIME uses **Ray** for resource orchestration, supporting both colocated (same GPUs
for training and inference) and disaggregated (separate GPU pools) deployment topologies.
It powers the post-training of production models including GLM-5.1, GLM-4.7, and
GLM-4.5.

**Amazon SageMaker HyperPod** provides purpose-built infrastructure for distributed
model training with deep health checks, automatic node replacement, and managed
Kubernetes (EKS) integration. Combined with FSx for Lustre shared storage and EFA
networking, HyperPod delivers the resilient, high-performance fabric that large-scale
RL workloads demand.

### Why SLIME on HyperPod?

| Challenge | How SLIME + HyperPod Addresses It |
|-----------|-----------------------------------|
| RL requires both fast inference and fast training | SLIME natively integrates SGLang (inference) + Megatron-LM (training) on separate or shared GPU pools |
| Multi-node coordination is fragile | Ray manages GPU allocation; HyperPod health-checks nodes and auto-replaces failures |
| Large models need high-bandwidth interconnect | p5.48xlarge nodes with 32 EFA devices each provide 3,200 Gbps aggregate network bandwidth |
| Shared storage for checkpoints and data | FSx for Lustre provides a POSIX-compliant parallel filesystem accessible by all pods |
| MoE models suffer from train-inference mismatch | SLIME's Rollout Routing Replay (R3) and Unified FP8 pipeline eliminate quantization-induced divergence |

## Architecture

The deployment uses a disaggregated architecture where training and inference GPUs
are separate, connected by EFA networking and FSx for Lustre shared storage.

```
+-----------------------------------------------------------------------+
|                  Amazon SageMaker HyperPod EKS Cluster                |
|                  (2x ml.p5.48xlarge, EKS orchestration)               |
|                                                                       |
|  +-----------------------------+  +-----------------------------+     |
|  |  Node 1: ml.p5.48xlarge     |  |  Node 2: ml.p5.48xlarge     |     |
|  |  8x H100 80GB  |  32x EFA   |  |  8x H100 80GB  |  32x EFA   |     |
|  |  ~2TB RAM      |  96 vCPU   |  |  ~2TB RAM      |  96 vCPU   |     |
|  +-----------------------------+  +-----------------------------+     |
|           |           |                    |           |               |
|           +------EFA (3200 Gbps/node)------+           |               |
|                        |                               |               |
|  +----------------------------------------------------------+         |
|  |          FSx for Lustre (1.2 TB, RWX)                    |         |
|  |  /fsx/models  /fsx/data  /fsx/checkpoints  /fsx/slime    |         |
|  +----------------------------------------------------------+         |
|                                                                       |
|  Kubernetes Resources:                                                |
|  - KubeRay operator (kuberay-operator namespace)                     |
|  - EFA device plugin (kube-system)                                    |
|  - FSx CSI driver (kube-system)                                       |
|  - NVIDIA device plugin (kube-system)                                 |
|  - DCGM exporter (hyperpod-observability)                             |
+-----------------------------------------------------------------------+
```

### SLIME Internal Architecture

SLIME operates as a three-component loop orchestrated by Ray:

```
                    +------------------+
                    |   Data Buffer    |
                    | (prompt queue +  |
                    |  rollout cache)  |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
   +-------------------+        +-------------------+
   |     Rollout        |        |     Training       |
   |  (SGLang servers   |        |  (Megatron-LM      |
   |   via sgl-router)  | <----> |   TP/PP/CP/EP)     |
   |                    | weight  |                    |
   |  - RadixAttention  |  sync  |  - GRPO / DAPO     |
   |  - Cont. batching  |        |  - Dynamic batch   |
   |  - TP across GPUs  |        |  - Gradient ckpt   |
   +-------------------+        +-------------------+
```

1. **Data Buffer** manages prompts, dispatches them for rollout, and stores generated
   samples with rewards.
2. **Rollout** runs SGLang inference servers behind `sgl-router`, generating responses
   and computing rewards via a custom reward function.
3. **Training** reads batches from the Data Buffer, computes GRPO advantages, and
   updates the policy via Megatron-LM. Updated weights are synced back to SGLang.

## Hardware Requirements

This recipe is validated on the following HyperPod cluster configuration:

| Component | Specification |
|-----------|---------------|
| **Instance type** | ml.p5.48xlarge |
| **Nodes** | 2 |
| **GPUs per node** | 8x NVIDIA H100 80GB SXM5 |
| **Total GPUs** | 16 |
| **GPU memory** | 1,280 GB aggregate |
| **Host RAM per node** | ~2 TB |
| **EFA per node** | 32 devices (3,200 Gbps aggregate) |
| **Capacity** | On-Demand |
| **Storage** | FSx for Lustre, 1.2 TB, mounted as PVC `fsx-claim` |
| **Kubernetes** | EKS v1.34, KubeRay operator |

For the disaggregated reward workload, the cluster also has a **Spot CPU
instance group** for the reward pool:

| Component | Specification |
|-----------|---------------|
| **Instance type** | ml.c5.4xlarge |
| **Nodes** | 4 |
| **Capacity** | **EC2 Spot** (`CapacityRequirements: Spot`) |
| **EFA** | None (reward RPC is HTTP) |
| **Placement** | Same AZ/subnet as the GPU pool |
| **Provisioning** | `Continuous` (required for Spot) |

### Supported Model Sizes (on 2x p5.48xlarge)

| Model | Parameters | SLIME Topology | TP | PP | Rollout GPUs | Training GPUs |
|-------|-----------|----------------|----|----|-------------|---------------|
| Qwen3-4B | 4B Dense | Colocated | 1 | 1 | 8 (shared) | 8 (shared) |
| GLM-Z1-9B | 9B Dense | Colocated | 2 | 1 | 16 (shared) | 16 (shared) |
| Qwen3-30B-A3B | 30B MoE | Disaggregated | 2 | 1 | 4 | 12 |
| Qwen2.5-72B | 72B Dense | Disaggregated | 4 | 2 | 8 | 8 |

For larger models (DeepSeek-R1, GLM-4.7-355B-A32B), scale to 8+ p5.48xlarge nodes.

## Prerequisites

1. An Amazon SageMaker HyperPod cluster with EKS orchestration and p5.48xlarge
   instance groups
2. `kubectl` and `helm` configured to access the cluster
3. The KubeRay operator installed on the cluster (see step 0 below)
4. FSx for Lustre persistent volume claim (`fsx-claim`) available
5. Docker and Amazon ECR access for building/pushing images
6. A Hugging Face account and access token for model downloads
7. **(Disaggregated path only)** A **Spot-capacity CPU instance group** in the
   same AZ as the GPU pool, with no EFA, for the reward service. Add one to an
   existing HyperPod cluster with a one-time `update-cluster` call — Spot is
   selected via `CapacityRequirements: { Spot: {} }` and requires `Continuous`
   node provisioning:

   ```bash
   aws sagemaker update-cluster \
     --cluster-name <your-hyperpod-cluster> \
     --instance-groups '[{
       "InstanceGroupName": "reward-spot-c5",
       "InstanceType": "ml.c5.4xlarge",
       "InstanceCount": 4,
       "LifeCycleConfig": {"SourceS3Uri": "s3://<your-lifecycle-bucket>/on-create.sh", "OnCreate": "on_create.sh"},
       "ExecutionRole": "<your-hyperpod-execution-role-arn>",
       "ThreadsPerCore": 1,
       "CapacityRequirements": {"Spot": {}}
     }]'
   ```

   Reuse the same subnet/security-group/lifecycle config as an existing CPU
   ("general") group so the node joins EKS identically. The nodes appear labeled
   `sagemaker.amazonaws.com/instance-group-name=reward-spot-c5`. See
   [Adding instance groups](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-operate-console-ui-edit.html)
   and [Using Spot in HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-spot.html).

## Quick Start

### 0. Install the KubeRay Operator (one-time per cluster)

The Ray cluster is managed by the KubeRay operator. If it is not already present
(`kubectl get crd rayclusters.ray.io`), install it with Helm:

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator \
    --version 1.4.2 --namespace kuberay-operator --create-namespace
```

### 1. Configure Environment Variables

```bash
cd 3.test_cases/pytorch/slime
cp env_vars.colocated.example env_vars
# Edit env_vars with your cluster-specific values
source env_vars
```

Key variables to configure:

```bash
# AWS / ECR (region and account are auto-derived from your AWS credentials)
export AWS_REGION="${AWS_REGION:-$(aws configure get region)}"
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/"
export IMAGE="slime-hyperpod"
export TAG="slime0.2.4-sgl0.5.12"            # pinned image tag (avoid :latest)

# Model
export MODEL_NAME="Qwen/Qwen3-4B"           # HuggingFace model ID
export MODEL_LOCAL="/fsx/models/Qwen3-4B"    # Local path on FSx
export HF_TOKEN="hf_..."                     # Your HuggingFace token

# Cluster
export NAMESPACE="default"
export FSX_CLAIM="fsx-claim"
export NUM_NODES=2
export GPUS_PER_NODE=8
```

Create the Hugging Face token Secret that the manifests reference (the Ray
cluster and reward-service pods read `HF_TOKEN` from it via `secretKeyRef`):

```bash
kubectl create secret generic hf-token \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

### 2. Build and Push Docker Image

The Docker image packages SLIME, SGLang, Megatron-LM, and all dependencies on top
of the NVIDIA NGC PyTorch base image with EFA support.

```bash
# Authenticate to ECR
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${REGISTRY}

# Create repository (first time only)
aws ecr create-repository --repository-name ${IMAGE} --region ${AWS_REGION} || true

# Build image (build context is this test-case directory, 3.test_cases/pytorch/slime)
docker build -t ${REGISTRY}${IMAGE}:${TAG} -f slime.Dockerfile .

# Push to ECR
docker push ${REGISTRY}${IMAGE}:${TAG}
```

### 3. Download and Prepare Model

SSH into a pod with FSx access (or use a setup job) and download the model:

```bash
# Create a data-prep pod
kubectl apply -f kubernetes/data-prep-pod.yaml

# Wait for it to be ready, then exec in
kubectl exec -it data-prep -- bash

# Inside the pod:
pip install huggingface_hub
huggingface-cli login --token ${HF_TOKEN}

# Download model
huggingface-cli download Qwen/Qwen3-4B --local-dir /fsx/models/Qwen3-4B

# Download training dataset
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
    --local-dir /fsx/data/dapo-math-17k

# Download evaluation dataset
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
    --local-dir /fsx/data/aime-2024
```

### 4. Convert Model Weights to Megatron Format

SLIME's Megatron training backend requires weights in `torch_dist` format:

```bash
# Inside a pod with GPUs (or use the Ray head pod later)
cd /opt/slime

# Load model config
source scripts/models/qwen3-4B.sh

# Convert HuggingFace -> Megatron torch_dist
PYTHONPATH=/opt/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /fsx/models/Qwen3-4B \
    --save /fsx/models/Qwen3-4B_torch_dist
```

For larger models (30B+), use `torchrun` with multiple GPUs to accelerate conversion:

```bash
torchrun --nproc_per_node=8 tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /fsx/models/Qwen3-30B-A3B \
    --save /fsx/models/Qwen3-30B-A3B_torch_dist
```

### 5. Deploy Ray Cluster

The Ray cluster provides the orchestration layer for SLIME's training and rollout
workers.

```bash
# Substitute environment variables into manifest
envsubst < kubernetes/raycluster.yaml | kubectl apply -f -

# Watch pods come up (1 head + workers)
kubectl get pods -w -l ray.io/is-ray-node=yes

# Port-forward Ray dashboard for monitoring
kubectl port-forward svc/slime-ray-head-svc 8265:8265 &
```

### 6. Configure the Reward

This sample scores rollouts in one of three ways -- pick one in `env_vars`:

- **Built-in rule-based (default):** `RM_TYPE="deepscaler"` (also `dapo`,
  `math`, `f1`, `gpqa`). Runs in-process on the rollout actors; no extra setup.
- **Custom in-process function:** set `CUSTOM_RM_PATH=<module>:<func>` to point
  SLIME at your own `async def reward_func(args, sample)`.
- **Remote reward service (CPU pool):** `RM_TYPE="remote_rm"` + `RM_URL=...` to
  offload scoring (e.g. a reward model) to the CPU instance group. See
  [Disaggregated Reward Service](#advanced-disaggregated-reward-service-on-a-cpu-instance-group).

No file copy is needed for the default path -- `deepscaler` is built into SLIME.

### 7. Launch GRPO Training

```bash
# Submit the training job via recipe script
bash recipe/run_grpo_qwen3_4b.sh
```

Monitor training:
```bash
# Ray dashboard (after port-forward)
open http://localhost:8265

# Follow Ray job logs
RAY_JOB_ID=$(ray job list --address http://localhost:8265 | head -3 | tail -1 | awk '{print $1}')
ray job logs ${RAY_JOB_ID} --address http://localhost:8265 --follow

# Monitor GPU utilization via DCGM
kubectl exec -it ${HEAD_POD} -- nvidia-smi
```

### 8. Convert Checkpoints Back to HuggingFace Format

After training completes, convert Megatron checkpoints to HuggingFace format for
evaluation and deployment:

```bash
kubectl exec -it ${HEAD_POD} -- bash -c "
cd /opt/slime && \
PYTHONPATH=/opt/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir /fsx/checkpoints/qwen3-4b-grpo/iter_0060/ \
    --output-dir /fsx/models/Qwen3-4B-GRPO-step60 \
    --origin-hf-dir /fsx/models/Qwen3-4B
"
```

That completes the core (colocated) workflow. For heavier reward functions, the
next section shows how to disaggregate reward scoring onto a separate CPU pool.

## Advanced: Disaggregated Reward Service on a CPU Instance Group

By default the reward function runs **in-process on the GPU rollout actors**.
For the lighweight rule-based math verification this is fine, but for a heavier reward -- a
reward **model** (a small sequence classifier), code-execution sandboxes, unit
tests, or RAG lookups -- that CPU/IO work steals cycles from the scarce GPUs.
SLIME's `remote_rm` hook lets you move it to a dedicated, independently-scalable
**CPU instance group in the same Availability Zone**. Because the reward service
is **stateless and fault-tolerant** (SLIME's `remote_rm` client retries with
backoff), that pool is a great fit for **EC2 Spot** capacity -- up to ~90%
cheaper than On-Demand. This is especially valuable when the reward tier fans
out to many replicas (e.g. fleets of code-execution or tool sandboxes scored per
rollout): the expensive GPUs stay On-Demand while the bursty, interruption-
tolerant scoring fleet rides Spot. This sample assumes the cluster already has a
Spot-backed `reward-spot-c5` instance group (see
[Prerequisites](#prerequisites)).

### Three-pool topology

```
   +---------------------------+        weight sync (NCCL)        +---------------------------+
   |   GPU training pool        | <===== over EFA / RDMA =======> |   GPU rollout pool         |
   |   (Megatron-LM, p5)        |   update_weights_from_tensor     |   (SGLang engines, p5)     |
   |   On-Demand                |   (async, overlaps generation)   |   On-Demand                |
   +---------------------------+                                   +-------------+-------------+
                                                                                 |
                EFA matters HERE (high-bandwidth GPU<->GPU collectives)          | reward RPC
                                                                                 | (HTTP, low-bandwidth)
                                                                                 v
                                                                  +---------------------------+
                                                                  |   CPU reward pool          |
                                                                  |   (reward model / verifier)|
                                                                  |   c5 -- SPOT, NO EFA        |
                                                                  +---------------------------+
```

There are two distinct offload boundaries, and only one is CPU-appropriate:

| Boundary | Transport | Belongs on | EFA? |
|----------|-----------|------------|------|
| Train <-> Rollout (weights + samples) | NCCL collectives, GPU<->GPU | GPU pools (On-Demand) | **Yes** -- this is what EFA/RDMA accelerates |
| Rollout -> Reward (scoring) | HTTP, prompt+response+float | CPU pool (Spot) | **No** -- low-bandwidth, latency-tolerant |

> **Why no EFA on the CPU pool?** EFA accelerates RDMA collectives. The reward
> RPC carries text + a scalar over HTTP and never touches RDMA, so EFA-enabled
> CPU instances would add cost with no datapath that uses them. Keep them on
> standard ENA networking; just keep them in the **same AZ** to minimize RPC
> latency.

> **Why Spot for the reward pool?** Reward scoring is stateless and the
> `remote_rm` client retries on transient errors, so a reclaimed node only costs
> a few in-flight requests, not training progress. The GPU pools stay On-Demand
> (training/rollout hold long-lived NCCL groups and are not interruption-safe).
> HyperPod handles Spot interruptions by tainting the node, gracefully evicting
> pods within `terminationGracePeriodSeconds`, and replacing capacity. See
> [Spot instances in HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-spot.html).


### Step 1: Ensure the Spot CPU reward pool exists

This path requires a Spot-capacity CPU instance group (default name
`reward-spot-c5`) in the **same AZ** as the GPU pool, with **no EFA**. If your
cluster doesn't already have one, create it with the one-time
`aws sagemaker update-cluster` command documented in
[Prerequisites](#prerequisites), then set `REWARD_NODE_GROUP` to its name in the
disaggregated profile.

> **Why Spot is safe here:** `CapacityRequirements` is chosen at group-creation
> time (immutable afterward) and requires `Continuous` provisioning. The reward
> Deployment tolerates HyperPod's Spot-interruption taint, adds a `startupProbe`
> for cold model loads, and rolls with `maxUnavailable: 1` so reward capacity
> never drops to zero during a reclaim.

### Step 2: Build and deploy the reward service

```bash
# Load the disaggregated profile (sets RM_TYPE=remote_rm, REWARD_NODE_GROUP,
# REWARD_BACKEND, REWARD_REPLICAS, REWARD_TAG/REWARD_IMAGE, RM_URL, etc.)
cp env_vars.disaggregated.example env_vars.disaggregated   # edit REWARD_NODE_GROUP if needed
source env_vars                # base config
source env_vars.disaggregated  # reward-service overlay

# Build + push the lightweight CPU reward image (no CUDA), versioned tag
docker build -t ${REWARD_IMAGE} -f reward_service.Dockerfile .
docker push ${REWARD_IMAGE}

# Deploy onto the CPU instance group (pinned via REWARD_NODE_GROUP, off GPU
# nodes, no EFA)
envsubst < kubernetes/reward-service.yaml | kubectl apply -f -
kubectl rollout status deployment/slime-reward -n ${NAMESPACE}

# (sanity) score a sample through the in-cluster Service. The reward_model
# backend returns a scalar score; higher = better response.
kubectl run rc --rm -i --restart=Never --image=curlimages/curl -- \
  curl -s -X POST http://slime-reward.${NAMESPACE}.svc.cluster.local:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain gravity.","response":"Gravity is the curvature of spacetime caused by mass.","label":null}'
```

The reward service exposes `POST /score` (returns a bare float reward, matching
SLIME's `remote_rm` contract) and `GET /health`.

### Step 3: Run the disaggregated (heavier) workload

`env_vars.disaggregated.example` increases load on **both** tiers at once: a
larger rollout batch / more samples-per-prompt / longer responses on the GPU
side, and a real reward **model** (deberta-v3-large) scaled one replica per CPU
node on the reward side. With the reward service already running (Step 2), launch
training -- the recipe automatically passes `--rm-url ${RM_URL}` because
`RM_TYPE=remote_rm`:

```bash
bash recipe/run_grpo_qwen3_4b.sh
```


**Measured on this sample** (2x ml.p5.48xlarge GPU pool + 4x ml.c5.4xlarge
reward pool, deberta-v3-large reward model, 64 prompts x 16 samples = 1024
generations per rollout step):

| Tier | Work | Result |
|------|------|--------|
| GPU (p5, SGLang) | generate 1024 responses | ~18 s (~57 gen/s) |
| CPU (c5, reward model) | score 1024 generations | ~223 s (~4.6 score/s), pod loadavg ~24 |

The reward scoring took **~12x longer than generation**. Had it run in-process
on the GPU rollout actors, the H100s would have sat idle ~92% of the loop
waiting on CPU model inference (observed GPU util dropped to 0% during scoring).
Offloading it to the CPU pool keeps the GPUs free and lets the reward tier scale
independently -- the entire rationale for the separate CPU instance group.

## File Structure

```
slime/                                    # 3.test_cases/pytorch/slime
├── README.md                            # This documentation
├── .gitignore
├── env_vars.colocated.example           # Base config: colocated train+rollout, built-in reward
├── env_vars.disaggregated.example       # Overlay: reward model on a CPU pool + heavier GRPO
├── slime.Dockerfile                     # SLIME + SGLang + Megatron + EFA image
├── requirements.txt                     # Pinned Python RL dependencies
├── reward_service.Dockerfile            # CPU-only image for the remote reward service
├── reward_service/
│   ├── app.py                           # FastAPI reward server (reward_model / math_verify)
│   └── requirements.txt                 # Pinned CPU deps (no CUDA)
├── kubernetes/
│   ├── raycluster.yaml                  # KubeRay cluster manifest (p5.48xlarge)
│   ├── reward-service.yaml              # CPU reward service Deployment + Service
│   └── data-prep-pod.yaml               # Utility pod for data preparation
├── recipe/
│   ├── run_grpo_qwen3_4b.sh             # GRPO training launcher (Qwen3-4B, colocated)
│   └── run_grpo_qwen3_30b_a3b.sh        # GRPO training launcher (Qwen3-30B-A3B MoE, disaggregated)
└── scripts/
    ├── convert_checkpoint.sh            # HF <-> Megatron conversion helper
    └── evaluate.sh                      # Evaluation launcher
```

## Training Configuration Deep Dive

### GRPO (Group Relative Policy Optimization)

GRPO is a critic-free RL algorithm that estimates advantages by comparing rewards
within a group of responses generated for the same prompt. This eliminates the need
for a separate value model, significantly reducing memory requirements.

**SLIME's GRPO implementation includes:**

| Parameter | Description | Our Setting |
|-----------|-------------|-------------|
| `--advantage-estimator grpo` | Use GRPO advantage estimation | GRPO |
| `--rollout-batch-size` | Number of prompts per rollout | 16 |
| `--n-samples-per-prompt` | Responses generated per prompt | 8 |
| `--global-batch-size` | Samples per optimizer step | 128 |
| `--num-steps-per-rollout` | Optimizer steps per rollout cycle | 1 |
| `--eps-clip` | PPO-style clipping lower bound | 0.2 |
| `--eps-clip-high` | Clipping upper bound (DAPO-style) | 0.28 |
| `--kl-loss-coef` | KL divergence penalty coefficient | 0.0 |
| `--entropy-coef` | Entropy bonus coefficient | 0.0 |

The constraint `rollout_batch_size * n_samples_per_prompt = global_batch_size * num_steps_per_rollout` (16 * 8 = 128 * 1) must always hold.

### Parallelism Strategy

SLIME inherits Megatron-LM's full parallelism stack. For our 2-node (16 GPU) setup:

**Qwen3-4B (Colocated mode):**
```
Tensor Parallel (TP) = 1
Pipeline Parallel (PP) = 1
Context Parallel (CP) = 1
Data Parallel (DP) = 16 (implicitly: total_gpus / TP / PP)
```

**Qwen3-30B-A3B MoE (Disaggregated mode):**
```
# Training: 12 GPUs
Tensor Parallel (TP) = 2
Pipeline Parallel (PP) = 1
Expert Parallel (EP) = 2
Context Parallel (CP) = 2
Data Parallel (DP) = 12 / (TP * PP) = 6

# Rollout (SGLang): 4 GPUs
rollout-num-gpus = 4
rollout-num-gpus-per-engine = 2  (TP=2 per SGLang server, 2 servers via sgl-router)
```

### Dynamic Batching

SLIME's `--use-dynamic-batch-size` combined with `--max-tokens-per-gpu` intelligently
packs variable-length samples to maximize GPU utilization:

```bash
--use-dynamic-batch-size
--max-tokens-per-gpu 8192   # Per-GPU token budget per micro-batch
```

This is particularly effective for RL workloads where response lengths vary wildly
(math proofs range from 100 to 8,000+ tokens).

### Dynamic Sampling with Over-Sampling

For higher-quality training data, SLIME supports DAPO-style dynamic sampling that
filters out uninformative prompt groups:

```bash
--rollout-batch-size 32
--n-samples-per-prompt 8
--over-sampling-batch-size 64
--dynamic-sampling-filter-path \
    slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

This generates 64 prompt groups, filters to keep only those where reward standard
deviation > 0, and continues sampling until 32 valid groups are collected.

## Reward Function

The sample supports three reward strategies, selected via `RM_TYPE` /
`CUSTOM_RM_PATH` / `RM_URL` in `env_vars`:

1. **Built-in rule-based rewards (default).** SLIME ships several rule-based
   reward types -- `deepscaler`, `dapo`, `math`, `f1`, `gpqa` -- selected with
   `--rm-type`. The math types extract the `\boxed{...}` answer from the
   response and grade it against the label using `math_verify` (LaTeX/sympy
   equivalence). For the bundled DAPO-math recipe, `RM_TYPE="deepscaler"` is the
   default and needs no extra setup.

2. **Custom in-process function.** To use your own logic, implement
   `async def reward_func(args, sample, **kwargs) -> float` and point SLIME at it
   with `CUSTOM_RM_PATH=<module>:<func>`. It runs inside the rollout actors.

3. **Remote reward service (CPU instance group).** For a heavy reward -- a
   reward **model**, code execution, or RAG -- set `RM_TYPE="remote_rm"` and
   `RM_URL` to offload scoring to the CPU pool. See
   [Disaggregated Reward Service](#advanced-disaggregated-reward-service-on-a-cpu-instance-group).
   The service (`reward_service/app.py`) exposes both a `reward_model` backend
   and a `math_verify` backend behind `POST /score`.

## Performance Considerations

### EFA Networking on HyperPod

Each p5.48xlarge node has 32 EFA devices providing 3,200 Gbps aggregate bandwidth.
This is critical for:

1. **NCCL all-reduce** during Megatron training (gradient synchronization)
2. **Weight sync** from Megatron training workers to SGLang inference servers
3. **Ray object store** transfers between actors

Environment variables for optimal EFA performance are set in the Ray cluster manifest:

```yaml
env:
  - name: FI_PROVIDER
    value: "efa"
  - name: FI_EFA_USE_DEVICE_RDMA
    value: "1"
  - name: NCCL_PROTO
    value: "Simple"
  - name: FI_EFA_FORK_SAFE
    value: "1"
  - name: NCCL_DEBUG
    value: "WARN"
```

### FSx for Lustre Storage Layout

Organize FSx storage for optimal parallel I/O:

```
/fsx/
├── models/                    # Pre-trained model weights (read-heavy)
│   ├── Qwen3-4B/
│   ├── Qwen3-4B_torch_dist/
│   └── Qwen3-30B-A3B/
├── data/                      # Training datasets (read-heavy)
│   ├── dapo-math-17k/
│   └── aime-2024/
├── checkpoints/               # Training checkpoints (write-heavy)
│   └── qwen3-4b-grpo/
│       ├── iter_0020/
│       └── iter_0040/
└── ...
```

### Memory Budget (p5.48xlarge, Colocated Mode)

| Component | Per-Node Memory | Notes |
|-----------|----------------|-------|
| Node allocatable | ~1,950 Gi | After system pods |
| Pod memory limit | 1,800 Gi | Leaves room for system overhead |
| Megatron training | ~200 GB | Model params + gradients + optimizer states |
| SGLang inference (colocated) | ~40 GB GPU | `--sglang-mem-fraction-static 0.8` |
| Ray overhead | ~16 GB | Object store + GCS |
| Checkpoints (write buffer) | ~50 GB | During save operations |

### Scaling to More Nodes

SLIME scales linearly with additional p5.48xlarge nodes:

| Nodes | GPUs | Recommended Models | Approx. Throughput |
|-------|------|--------------------|--------------------|
| 2 | 16 | Qwen3-4B, GLM-Z1-9B | ~500 samples/hour |
| 4 | 32 | Qwen3-30B-A3B MoE | ~800 samples/hour |
| 8 | 64 | GLM-4.7-355B-A32B MoE | ~1,200 samples/hour |
| 16 | 128 | DeepSeek-R1 671B MoE | ~2,000 samples/hour |

## Comparison with Other RL Frameworks

| Feature | SLIME | OpenRLHF | veRL |
|---------|-------|----------|------|
| Training backend | Megatron-LM | DeepSpeed ZeRO | FSDP2 |
| Inference backend | SGLang (native) | vLLM | vLLM (inline) |
| Orchestration | Ray | Ray | Ray |
| Topology | Colocated or Disaggregated | Disaggregated only | Colocated only |
| MoE support | Full (EP, FP8, R3) | Partial | Partial |
| Custom rollout | Full programmatic control | Fixed pipeline | Fixed pipeline |
| Weight sync | CUDA IPC zero-copy | HTTP/RPC | In-process |
| Dynamic sampling | Native (DAPO-style) | Manual | Manual |
| Multi-turn agents | Native interface | Requires modification | Requires modification |
| Checkpoint format | Megatron torch_dist | HuggingFace direct | FSDP shards |
| Production lineage | GLM-5.1, GLM-4.7 | Research | Research |

## Troubleshooting

### Common Issues

**Pod stuck in `Pending` state**
```bash
kubectl describe pod <pod-name>
# Check for resource constraints -- GPU/EFA/memory requests may exceed node capacity
# Ensure no other workloads are consuming GPUs
kubectl get pods --all-namespaces -o json | \
    python3 -c "import json,sys; [print(p['metadata']['name'], p['spec'].get('containers',[{}])[0].get('resources',{}).get('requests',{}).get('nvidia.com/gpu','0')) for p in json.load(sys.stdin)['items'] if p['status']['phase']=='Running']"
```

**Ray workers fail to connect to head node**
```bash
# Verify Ray head service is accessible
kubectl get svc slime-ray-head-svc
# Check that worker pods can resolve the service name
kubectl exec <worker-pod> -- nslookup slime-ray-head-svc
# Ensure RAY_memory_monitor_refresh_ms=0 is set (prevents OOM kills during init)
```

**NCCL/EFA initialization errors**
```bash
# Verify EFA devices are available
kubectl exec <pod> -- fi_info -p efa
# Check NCCL environment
kubectl exec <pod> -- env | grep NCCL
# Ensure FI_PROVIDER=efa and FI_EFA_USE_DEVICE_RDMA=1 are set
```

**OOM during Megatron training**
- Reduce `--max-tokens-per-gpu` (e.g., from 8192 to 4096)
- Enable gradient checkpointing: `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`
- For colocated mode, reduce `--sglang-mem-fraction-static` (e.g., from 0.8 to 0.6)

**SGLang fails to start (CUDA OOM)**
- In colocated mode, SGLang launches after Megatron occupies GPU memory.
  Reduce `--sglang-mem-fraction-static` to leave room.
- Ensure the model fits within the available GPU memory per TP shard.

**Weight conversion fails**
- Ensure `PYTHONPATH` includes the Megatron-LM directory
- Verify model config parameters match (check `--rotary-base`, `--vocab-size`, etc.)
- For MoE models, ensure `--expert-model-parallel-size` is set correctly

**FSx disk full during checkpointing**
- SLIME Megatron checkpoints can be large (model_params * 12 bytes for Adam)
- Set `--save-interval` to a higher value
- Clean old checkpoints: `rm -rf /fsx/checkpoints/*/iter_00*/`
- Monitor FSx usage: `df -h /fsx`

### Monitoring

**Ray Dashboard** (port-forward to 8265):
- View active actors, resource usage, and job logs
- Monitor GPU utilization per Ray worker

**DCGM Metrics** (pre-installed on HyperPod):
```bash
# GPU utilization and memory per node
kubectl exec -it <dcgm-exporter-pod> -n hyperpod-observability -- \
    curl localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL
```

**Training Metrics** (logged by SLIME):
- `reward_mean`: Average reward across rollout batch
- `policy_loss`: GRPO policy gradient loss
- `kl_divergence`: KL between policy and reference model
- `gen_length_mean`: Average response length
- `mfu`: Model FLOPs utilization (training efficiency)

## Software Versions

| Component | Version |
|-----------|---------|
| SLIME | v0.2.4 |
| SGLang | 0.5.12.post1 |
| Megatron-LM | commit `3714d81d` (SLIME-compatible) |
| Ray | 2.55.1 |
| transformers | 5.6.0 (pulled by SGLang) |
| CUDA | 13.0 |
| PyTorch | 2.11.0+cu130 |
| EFA installer | 1.48.0 |
| GDRCopy | v2.5.2 |
| aws-ofi-nccl | v1.19.0 |
| Base image | nvcr.io/nvidia/pytorch:26.02-py3 |
| Reward service base | python:3.12-slim (CPU-only) |

## Cost Optimization

| Strategy | Description | Savings |
|----------|-------------|---------|
| Colocated mode | Share GPUs for training + inference | ~50% fewer GPUs needed |
| Spot reward pool | Run the disaggregated reward service on EC2 Spot CPU nodes | up to ~90% vs On-Demand for that tier |
| Dynamic batching | Pack variable-length samples efficiently | 20-30% faster training |
| FP8 inference | Use bf16 training with fp8 rollout | 40-60% faster inference |
| Partial rollout | Reuse aborted partial generations | 10-15% less wasted compute |
| Checkpoint pruning | Save only at key intervals | Reduced FSx costs |

## References

- [SLIME GitHub Repository](https://github.com/THUDM/slime)
- [SLIME Blog: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/)
- [Miles (Enterprise fork of SLIME)](https://github.com/radixark/miles)
- [SGLang Project](https://github.com/sgl-project/sglang)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [GRPO Paper (DeepSeek-R1)](https://arxiv.org/abs/2402.03300)
- [Amazon SageMaker HyperPod Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html)
- [Spot instances in SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-spot.html)
- [awsome-distributed-ai](https://github.com/awslabs/awsome-distributed-ai)
- [KubeRay Documentation](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)

## Security

See [CONTRIBUTING](https://github.com/awslabs/awsome-distributed-ai/blob/main/CONTRIBUTING.md) for more information.

## License

This sample code is made available under the MIT-0 license. See the LICENSE file.
