# OpenEnv Wordle GRPO Training on SageMaker HyperPod (EKS)

Multi-GPU GRPO (Group Relative Policy Optimization) training with [OpenEnv](https://github.com/meta-pytorch/OpenEnv) Wordle environment and [TRL](https://huggingface.co/docs/trl) on [Amazon SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html) with EKS orchestration and NVIDIA L40S GPUs.

This test case trains a language model to play [Wordle](https://en.wikipedia.org/wiki/Wordle) using reinforcement learning. The OpenEnv Wordle environment runs as a Kubernetes service, and the GRPO trainer connects to it over the cluster network, demonstrating the client-server separation that OpenEnv provides.

## What is OpenEnv?

[OpenEnv](https://github.com/meta-pytorch/OpenEnv) is an open-source framework by Meta for building, deploying, and interacting with isolated execution environments for **agentic reinforcement learning**. It is a collaborative effort between Meta, Hugging Face, Unsloth, GPU Mode, Reflection, and others.

OpenEnv provides a standardized [Gymnasium](https://gymnasium.farama.org/)-compatible interface through three simple APIs:

| API | Purpose |
|-----|---------|
| `reset()` | Start a new episode and receive the initial observation |
| `step(action)` | Submit an action and receive the next observation, reward, and done flag |
| `state()` | Query the current environment state (episode ID, step count) |

**Why does OpenEnv matter for RL training?**

- **Client-server architecture** — Environments run inside Docker containers and communicate over HTTP/WebSocket. Training code never imports environment code, making the two fully decoupled.
- **Any RL framework** — TRL, torchforge, Unsloth, SkyRL, veRL, or custom loops can all connect via typed Python clients.
- **Scales on Kubernetes** — Each environment is a container behind a Service. Add replicas to increase rollout throughput without changing training code.
- **30+ built-in environments** — Chess, Blackjack, Atari, CARLA, coding sandboxes, BrowserGym, financial trading, and more ship with the project.
- **MCP support** — Environments expose [Model Context Protocol](https://modelcontextprotocol.io/) tool endpoints so agents can call environment-specific tools directly.
- **Rewards stay in the environment** — Domain-specific reward logic is encapsulated inside the environment container, not scattered across training scripts.

```
Training Loop (GPU)                    OpenEnv Environment (CPU container)
     │                                         │
     │ ── WebSocket / HTTP ──────────────────> │
     │      env.reset()                        │   Docker container
     │ <── observation ──────────────────────  │   running FastAPI
     │                                         │
     │ ── env.step(action) ──────────────────> │   Endpoints:
     │ <── observation, reward, done ────────  │     /ws      (Gym API)
     │                                         │     /mcp     (MCP tools)
     │ ── env.state() ──────────────────────> │     /health  (liveness)
     │ <── episode_id, step_count ───────────  │
```

In this test case, the Wordle environment runs as a lightweight CPU pod on a HyperPod EKS cluster, and the GRPO training pod connects to it over the Kubernetes internal network. Because the environment is a separate service, you can:

1. **Scale environments independently** — Add more Wordle replicas if rollout collection is the bottleneck.
2. **Swap environments without changing training code** — Replace Wordle with Chess, coding, or any other OpenEnv environment by changing a single URL.
3. **Run the environment anywhere** — On the same cluster, on a HuggingFace Space, or on a remote server. The training pod just needs a URL.

For more details, see the [OpenEnv documentation](https://meta-pytorch.org/OpenEnv/) and the [Getting Started tutorials](https://meta-pytorch.org/OpenEnv/auto_getting_started/index.html).


## Overview

| Component | Details |
|-----------|---------|
| Framework | [TRL](https://huggingface.co/docs/trl) (GRPOTrainer) + [OpenEnv](https://meta-pytorch.org/OpenEnv/) |
| Algorithm | GRPO (Group Relative Policy Optimization) |
| Model | Qwen/Qwen3-1.7B (configurable) |
| Environment | [TextArena Wordle](https://huggingface.co/spaces/burtenshaw/wordle) via OpenEnv |
| GPU | 4x NVIDIA L40S (ml.g6e.12xlarge) — 48 GB VRAM each |
| Inference Engine | vLLM (colocate or server mode) |
| Reward Signals | 4 (correctness, green letters, yellow letters, repetition penalty) |
| Platform | SageMaker HyperPod with EKS orchestration |

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              SageMaker HyperPod (EKS Orchestrator)           │
│                                                              │
│  ┌────────────────────┐     ┌──────────────────────────────┐ │
│  │ OpenEnv Wordle     │     │ GRPO Training Pod            │ │
│  │ Environment (CPU)  │     │ (ml.g6e.12xlarge, 4x L40S)   │ │
│  │                    │     │                              │ │
│  │  Deployment (x2)   │◄────│  Option A: colocate          │ │
│  │  svc/openenv-wordle│ WS  │    1 GPU: vLLM+train         │ │
│  │  :7860             │     │                              │ │
│  │                    │     │  Option B: server mode        │ │
│  │  /reset            │     │    GPU 0: vLLM server        │ │
│  │  /step             │     │    GPU 1-3: FSDP train       │ │
│  │  /health           │     │                              │ │
│  └────────────────────┘     └──────────────────────────────┘ │
│                                                              │
│  HyperPod features:                                          │
│    - Node health monitoring + deep health checks             │
│    - Job auto-resume on hardware failure                     │
│    - Automatic node replacement                              │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ FSx for Lustre (/fsx)                                   ││
│  │  /fsx/hf_cache          — HuggingFace model cache       ││
│  │  /fsx/checkpoints/      — training checkpoints          ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **SageMaker HyperPod EKS cluster** with GPU worker groups (e.g. `ml.g6.12xlarge` or `ml.g6e.12xlarge`). See [`architectures/sagemaker-hyperpod-eks/`](../../../../architectures/sagemaker-hyperpod-eks/) for cluster setup instructions.

2. **HyperPod Helm chart** (`hyperpod-dependencies`) installed in the `kube-system` namespace. This bundles the NVIDIA device plugin, health monitoring agents, and other HyperPod components:
   ```bash
   helm list -n kube-system | grep hyperpod
   ```

3. **FSx for Lustre** shared filesystem accessible from all nodes, mounted as a PVC (set the name in `env_vars` as `FSX_PVC`).

4. **NVIDIA device plugin** running on GPU nodes (installed by the HyperPod Helm chart):
   ```bash
   kubectl get daemonset -n kube-system dependencies-nvidia-device-plugin
   ```

5. **Docker** with BuildKit support for building container images.

6. **HuggingFace token** with read access for gated model downloads.

### Connect to your HyperPod EKS cluster

```bash
# Get the underlying EKS cluster name from your HyperPod cluster
aws eks update-kubeconfig --name <EKS_CLUSTER_NAME>
kubectl config current-context
```

Expected output:

```
Added new context arn:aws:eks:us-east-1:123456789012:cluster/my-eks-cluster to /home/ubuntu/.kube/config
arn:aws:eks:us-east-1:123456789012:cluster/my-eks-cluster
```

Verify HyperPod nodes are healthy:

```bash
kubectl get nodes -L sagemaker.amazonaws.com/node-health-status,node.kubernetes.io/instance-type
```

Expected output:

```
NAME                           STATUS   ROLES    AGE   VERSION               NODE-HEALTH-STATUS   INSTANCE-TYPE
hyperpod-i-0a1b2c3d4e5f67890   Ready    <none>   2d    v1.33.5-eks-ecaa3a6   Schedulable          ml.g6.12xlarge
hyperpod-i-0f9e8d7c6b5a43210   Ready    <none>   2d    v1.33.5-eks-ecaa3a6   Schedulable          ml.g6.12xlarge
```

### Clone the repository

```bash
git clone https://github.com/awslabs/awsome-distributed-training/
cd awsome-distributed-training/examples/pytorch/trl/openenv-wordle-grpo
```

## 1. Build Container Image

### 1.1. Configure environment variables

```bash
cp env_vars.example env_vars
# Edit env_vars: set HF_TOKEN, HYPERPOD_CLUSTER_NAME, EKS_CLUSTER_NAME, and NAMESPACE
source env_vars
```

### 1.2. Create the HuggingFace token secret

Store your HuggingFace token in a Kubernetes Secret (used by all training and inference manifests):

```bash
kubectl create secret generic hf-token \
  --from-literal=token=$HF_TOKEN \
  --namespace=$NAMESPACE
```

### 1.3. Build and push the container image

Build the image (takes ~15-20 minutes on first build):

```bash
docker build -t ${REGISTRY}${IMAGE}:${TAG} .
```

Expected output (abbreviated):

```
[+] Building 850.2s (12/12) FINISHED
 => [stage-0  1/11] FROM public.ecr.aws/hpc-cloud/nccl-tests:latest
 => [stage-0  2/11] RUN apt update && apt install -y nvtop
 => [stage-0  5/11] RUN mkdir -p /opt/miniconda3 ...
 => [stage-0  6/11] RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
 => [stage-0  7/11] COPY src/ /wordle-grpo/
 => [stage-0  8/11] RUN pip install -r /wordle-grpo/requirements.txt
 => [stage-0  9/11] RUN pip install flash-attn>=2.5.0 --no-build-isolation
 => [stage-0 10/11] RUN pip install flashinfer-python ...
 => [stage-0 11/11] WORKDIR /wordle-grpo
```

Push to ECR:

```bash
aws ecr create-repository --repository-name ${IMAGE} 2>/dev/null || true
aws ecr get-login-password | docker login --username AWS --password-stdin ${REGISTRY}
docker push ${REGISTRY}${IMAGE}:${TAG}
```

Expected output:

```
Login Succeeded
The push refers to repository [123456789012.dkr.ecr.us-east-1.amazonaws.com/openenv-wordle-grpo]
...
latest: digest: sha256:2d2cac478f72... size: 856
```

## 2. Deploy OpenEnv Wordle Environment

The Wordle environment runs as a CPU-only Kubernetes Deployment with 2 replicas behind a ClusterIP service:

```bash
envsubst '$NAMESPACE $REGISTRY $IMAGE $TAG' < kubernetes/openenv-wordle-env.yaml | kubectl apply -f -
```

Verify the environment is running:

```bash
kubectl get pods -l app=openenv-wordle
kubectl get svc openenv-wordle
```

Expected output:

```
NAME                              READY   STATUS    RESTARTS   AGE
openenv-wordle-6d4f8b7c9-abc12   1/1     Running   0          45s
openenv-wordle-6d4f8b7c9-def34   1/1     Running   0          45s

NAME             TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)    AGE
openenv-wordle   ClusterIP   172.20.45.67   <none>        7860/TCP   45s
```

Test connectivity:

```bash
kubectl run test-env --rm -it --restart=Never \
  --image=curlimages/curl -- \
  curl -s http://openenv-wordle:7860/health
# Expected: {"status": "healthy"}
```

> **Note:** You can also skip the local deployment and use the public HuggingFace Space instead by setting `--env-url https://burtenshaw-wordle.hf.space` in the training command. The local deployment avoids rate limits and provides lower latency.

## 3. Launch GRPO Training

### 3.1. Single-GPU / Colocate Mode

The simplest option: vLLM and GRPO training share the same GPU. Good for models up to ~3B parameters on a single L40S (48 GB).

```bash
source env_vars
envsubst '$NAMESPACE $REGISTRY $IMAGE $TAG $INSTANCE_TYPE $MODEL_NAME $VLLM_MODE $NUM_GPU_PER_NODE $NUM_GENERATIONS $GRADIENT_ACCUMULATION_STEPS $LEARNING_RATE $FSX_PVC' < kubernetes/train-grpo-wordle.yaml | kubectl apply -f -
```

### 3.2. Monitor training

```bash
# Watch pod status
kubectl get pod grpo-wordle -w

# Follow training logs
kubectl logs grpo-wordle -f

# Check checkpoints
kubectl exec grpo-wordle -- ls -la /fsx/checkpoints/wordle-grpo/
```

Example training output:

```
INFO:     Starting GRPO training with Wordle environment
INFO:     Model: Qwen/Qwen3-1.7B | vLLM mode: colocate
INFO:     Reward functions: reward_correct, reward_greens, reward_yellows, reward_repetition
INFO:     Using 2 rollouts per dataset prompt

 Step  Loss     LR        Reward/correct  Reward/greens  Reward/yellows
 1     0.682    5.00e-06  0.000           0.120          0.080
 2     0.641    5.00e-06  0.000           0.140          0.100
 5     0.573    5.00e-06  0.050           0.180          0.120
 10    0.489    5.00e-06  0.100           0.220          0.160
 25    0.412    5.00e-06  0.200           0.280          0.180
 50    0.358    5.00e-06  0.300           0.340          0.200
 100   0.301    5.00e-06  0.400           0.400          0.220

Saving checkpoint to /fsx/checkpoints/wordle-grpo/checkpoint-25 ...
Saving checkpoint to /fsx/checkpoints/wordle-grpo/checkpoint-50 ...
Saving checkpoint to /fsx/checkpoints/wordle-grpo/checkpoint-75 ...
Saving checkpoint to /fsx/checkpoints/wordle-grpo/checkpoint-100 ...
Training complete.
```

### 3.3. Stop training

```bash
kubectl delete pod grpo-wordle
```

## 4. Multi-GPU Training (vLLM Server Mode)

For higher throughput and larger models, split inference and training across GPUs:

| GPU | Role |
|-----|------|
| GPU 0 | Dedicated vLLM inference server |
| GPU 1-3 | FSDP-sharded GRPO training (via Accelerate) |

```bash
source env_vars
envsubst '$NAMESPACE $REGISTRY $IMAGE $TAG $INSTANCE_TYPE $MODEL_NAME $NUM_GENERATIONS $GRADIENT_ACCUMULATION_STEPS $LEARNING_RATE $FSX_PVC' < kubernetes/train-grpo-wordle-multigpu.yaml | kubectl apply -f -
```

The pod runs two containers:
- **vllm-server**: Starts a vLLM server on GPU 0 (1 GPU), serving the model for fast rollout generation
- **trainer**: Waits for the vLLM server, then launches Accelerate with 3 processes for FSDP training on GPUs 1-3 (3 GPUs)

### 4.1. Monitor multi-GPU training

```bash
# Check both containers
kubectl get pod grpo-wordle-multigpu -w

# Follow vLLM server logs
kubectl logs grpo-wordle-multigpu -c vllm-server --tail=20

# Follow training logs
kubectl logs grpo-wordle-multigpu -c trainer -f
```

### 4.2. Stop

```bash
kubectl delete pod grpo-wordle-multigpu
```

## 5. Inference

After training completes, deploy an inference pod to test the trained model:

```bash
source env_vars
envsubst '$NAMESPACE $REGISTRY $IMAGE $TAG $INSTANCE_TYPE $MODEL_NAME $FSX_PVC' < kubernetes/inference-wordle.yaml | kubectl apply -f -
```

Interactive inference:

```bash
kubectl exec -it inference-wordle -- python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-1.7B',
    torch_dtype=torch.bfloat16,
    device_map='auto',
)

prompt = 'Play Wordle. My first guess is: [crane]'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
output = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))
"
```

## 6. Cleanup

```bash
# Stop all pods
kubectl delete pod grpo-wordle grpo-wordle-multigpu inference-wordle 2>/dev/null
kubectl delete -f kubernetes/openenv-wordle-env.yaml
kubectl delete secret hf-token 2>/dev/null
```


## Configuration Reference

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-id` | Qwen/Qwen3-1.7B | HuggingFace model identifier |
| `--env-url` | http://openenv-wordle:7860 | OpenEnv Wordle server URL |
| `--vllm-mode` | colocate | `colocate` (1 GPU) or `server` (multi-GPU) |
| `--num-generations` | 2 | Rollouts per prompt (higher = better estimates, slower) |
| `--gradient-accumulation-steps` | 64 | Effective batch = batch_size x grad_accum |
| `--learning-rate` | 5e-6 | AdamW learning rate |
| `--max-turns` | 6 | Max Wordle guesses per episode |
| `--temperature` | 0.8 | Sampling temperature for rollout generation |
| `--dataset-size` | 3000 | Number of training prompts |
| `--save-interval` | 25 | Checkpoint every N steps |

### Reward Functions

| Reward | Signal | Description |
|--------|--------|-------------|
| `reward_correct` | Binary | 1.0 if the word was solved, 0.0 otherwise |
| `reward_greens` | 0.0 - 1.0 | Fraction of letters in the correct position |
| `reward_yellows` | 0.0 - 1.0 | Fraction of letters present but wrong position |
| `reward_repetition` | 0.0 - 1.0 | Penalizes repeating the same guess |
