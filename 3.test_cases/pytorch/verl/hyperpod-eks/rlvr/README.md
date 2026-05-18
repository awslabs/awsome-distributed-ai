# RLVR Recipe

This repository provides a complete setup for running reinforcement learning from verifiable rewards (RLVR) on EKS clusters using Ray and verl. RLVR trains language models using verifiable rewards from math and coding tasks, where correctness can be automatically verified. The project uses verl, an efficient RL training framework from ByteDance, to run algorithms like GRPO (Group Relative Policy Optimization) and DAPO (Direct Advantage Policy Optimization) on distributed GPU clusters.

## What is verl?

[verl (Volcano Engine Reinforcement Learning)](https://github.com/volcengine/verl) is a flexible, production-ready RL training library for large language models. It provides seamless integration with popular frameworks like FSDP, Megatron-LM, vLLM, and Ray, enabling efficient distributed training with state-of-the-art throughput. This repo includes the full verl codebase with custom run scripts optimized for HyperPod.

## What is RLVR?

[Reinforcement Learning from Verifiable Rewards (RLVR)](https://arxiv.org/abs/2506.14245) is a training approach where models learn from tasks with objectively verifiable outcomes, such as math problems or code execution. Unlike human preference-based RL, RLVR uses ground-truth correctness as the reward signal, making it particularly effective for reasoning tasks.

## Getting started

### Prerequisites

**Cluster**:
From here on out, we will assume you have an EKS cluster with GPU nodes (e.g., p5en.48xlarge). This example can be run on an EKS or HyperPod EKS cluster. 

This example was tested on 4 p5en.48xlarge nodes (8xH200 GPUs each). If you are using different node types, modify the cluster environment variables in `env_vars`. Feel free to change the model type/size, and training parameters to accomodate smaller or larger node types. 

**Storage**:
- This examples uses a FSx for Lustre file system that mounts to the pods via a pvc called `fsx-claim`. We store the dataset, as well as model checkpoints here. Feel free to substitute this claim with your own. 

**Versions**:
The example was tested on versions:
- EKS: 1.32 / 1.33
- KubeRay: 1.4.2
- VERL: v0.6.1

### Clone this repo
```bash
git clone https://github.com/awslabs/awsome-distributed-training.git 
cd awsome-distributed-training/3.test_cases/pytorch/verl/hyperpod-eks/rlvr
```

### Install verl repository
This repository contains the verl framework and scripts needed for RLVR training. We install it to get access to the distributed RL training algorithms (GRPO, DAPO, and more) and the integration code that connects verl with EKS/Ray clusters for scalable language model fine-tuning on math and coding tasks.

```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout v0.6.1
cd ..
```

### Create RayCluster

Install KubeRay operator to manage Ray clusters on Kubernetes:
```bash
./setup/install-kuberay.sh
```

Configure your cluster settings (AWS region, cluster name, GPU counts, model paths):
```bash
# Copy the example file and customize it with your values
cp setup/env_vars.example setup/env_vars
vim setup/env_vars
```

> **Important**: The `env_vars` file contains sensitive information like your HuggingFace token, AWS account details, and cluster IDs. This file is gitignored to prevent accidentally committing credentials. Always use `env_vars.example` as your template.

Load the environment variables into your shell session:
```bash
source setup/env_vars
```

Build a Docker image with verl, EFA networking support, and push to ECR:
```bash
./setup/build-push.sh
```

Deploy the Ray cluster with head and worker pods configured for distributed training:
```bash
envsubst < setup/raycluster.yaml | kubectl apply -f -
```

> **Note**: Considerations before applying raycluster.yaml
> - Ensure you have a file system before applying the RayCluster. This raycluster.yaml is assuming you have a pvc in place called `fsx-claim`. Feel free to modify the configuration depending on your file system setup
> - This Raycluster is assuming you have 4 p5en.48xlarge instance types. Modify your setup/env_vars and NodeSelector in the yaml to adjust for your cluster. 


Download the GSM8K math dataset and prepare it for GRPO training:
```bash
./setup/load_data_grpo.sh
```

Forward the Ray dashboard to localhost for monitoring training progress:
```bash
./ray-expose.sh
```

Submit a GRPO training job to the Ray cluster. This trains a language model on math reasoning using group relative policy optimization:
```bash
./recipe/run_grpo_configurable.sh
```

The `verl/` directory contains the official verl framework, and `recipe/` includes custom run scripts (`run_grpo_configurable.sh`, `run_dapo_configurable.sh`) that integrate with your environment variables for easy configuration.

### Running on g5 Instances (A10G 24GB GPUs)

The default configuration targets p5en.48xlarge (8 × H200 80GB). For g5.12xlarge (4 × A10G 24GB), additional memory optimizations are required. A complete recipe is provided at `recipe/run_gptoss_grpo.sh` for training `openai/gpt-oss-20b` (a 20B MoE model) on g5 instances.

Key differences for small-GPU deployments:

| Parameter | p5en (80GB) | g5 (24GB) | Why |
|-----------|-------------|-----------|-----|
| `actor.strategy` | `fsdp` | `fsdp2` | FSDP1 disables CPUOffload for actor role |
| `fsdp_config.offload_policy` | not set | `True` | FSDP2-specific flag for proper CPU offloading |
| `fsdp_config.model_dtype` | default (fp32) | `bf16` | veRL defaults actor to fp32; explicit bf16 halves memory |
| `rollout.gpu_memory_utilization` | `0.6` | `0.6` | Fraction of TOTAL GPU for vLLM (model+cache) |
| `rollout.enforce_eager` | `False` | `True` | CUDA graphs need extra workspace; OOM on 24GB |
| `use_kl_loss` | `True` | `False` | Eliminates ref model log-probs during actor update |
| `rollout.tensor_model_parallel_size` | 2 | 4 | Shard model across all 4 GPUs per node |
| `trainer.nnodes` | matches node count | workers only | Head pod without GPUs causes NCCL hang |

Setup steps:

```bash
# 1. Update env_vars for g5 (see env_vars.example for g5 section)
source setup/env_vars

# 2. Prepare data (Multilingual-Thinking dataset)
./setup/load_data_gptoss.sh

# 3. Copy reward function to shared storage
cp recipe/language_reward.py /fsx/verl/reward/language_reward.py

# 4. Submit training
./recipe/run_gptoss_grpo.sh
```

### Checkpoint Management

Each veRL checkpoint saves the full FSDP model state for all GPUs. For a 20B parameter model across 12 GPUs, each checkpoint is approximately **117GB**.

Important settings:
- `trainer.save_freq=20` — Save every 20 steps (not every step)
- `trainer.max_actor_ckpt_to_keep=3` — Automatically delete old checkpoints
- `trainer.resume_mode=auto` — Resume from latest checkpoint after crash

For a 1.2TB FSx volume, `save_freq=1` will fill the disk in ~9 steps. Use `save_freq=20` with `max_actor_ckpt_to_keep=3` to keep disk usage under 351GB.

### Observability

For EKS:
Please see this documentation to set up Prometheus and Grafana dashboards for Ray clusters: [Using Prometheus & Grafana](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/prometheus-grafana.html)

For HyperPod EKS:
Check out the `observability/` directory to integrate Ray's native metrics dashboards with HyperPod's Amazon Managed Prometheus and Grafana

## Troubleshooting

### OOM during vLLM initialization (KV cache)

**Symptom**: `ValueError: No available memory for the cache blocks` during vLLM init.

**Cause**: `gpu_memory_utilization` is the fraction of **total** GPU memory, not just KV cache. If the model shard is 3.3GB on a 23GB GPU and you set `gpu_memory_utilization=0.3`, vLLM only gets 6.9GB total — less than the model itself.

**Fix**: Set `gpu_memory_utilization=0.6` or higher. Ensure `(gpu_memory_utilization × total_gpu_memory) > model_shard_size`.

### OOM during backward pass

**Symptom**: `torch.OutOfMemoryError` during actor update, even though vLLM init succeeded.

**Cause**: With FSDP1, the actor model stays on GPU during training because FSDP1 explicitly disables `CPUOffload` for the actor role (to avoid incorrect results with gradient accumulation). When vLLM releases GPU memory via sleep, FSDP1 still holds ~20GB of actor params.

**Fix**: Switch to `actor.strategy=fsdp2` with `fsdp_config.offload_policy=True`. FSDP2 properly offloads actor params to CPU during training.

### NCCL hang or fi_av_insert failure

**Symptom**: Training hangs at NCCL init, or error `fi_av_insert failed`.

**Cause**: Mixing EFA and non-EFA pods in the same NCCL communicator. The Ray head pod may not request `vpc.amazonaws.com/efa`, causing NCCL to use Socket transport on the head while workers use Libfabric.

**Fix**: Set `trainer.nnodes` to the number of **worker** nodes only (exclude the head). For example, with 1 head + 3 workers, set `trainer.nnodes=3`.

### Checkpoint save fills disk

**Symptom**: `RuntimeError: PytorchStreamWriter failed writing file data/...: file write failed` during `torch.save`.

**Cause**: Each checkpoint is ~117GB for a 20B model across 12 GPUs. With `save_freq=1`, a 1.2TB FSx fills in 9 steps.

**Fix**: Set `trainer.save_freq=20` and `trainer.max_actor_ckpt_to_keep=3`. Verify free space with `lfs df -h /fsx` (standard `df` may lag on Lustre).

### Zombie Ray jobs

**Symptom**: `ray job status` shows RUNNING, but workers have crashed.

**Cause**: If workers crash (SIGTERM/OOM), the Ray job driver may not detect it immediately. The job appears RUNNING for hours.

**Fix**: Always verify worker health: `kubectl exec <head-pod> -- bash -c "ps aux | grep verl"` on worker pods. Check worker logs in `/tmp/ray/session_*/logs/` for the job hex ID.

### Actor defaults to fp32

**Symptom**: Actor model uses ~40GB instead of ~20GB (for a 20B model).

**Cause**: veRL defaults actor to `torch.float32` in FSDP1 (`torch_dtype = torch.float32 if self._is_actor`). Only ref and critic default to bf16.

**Fix**: Set `actor_rollout_ref.actor.fsdp_config.model_dtype=bf16` explicitly.

### expandable_segments incompatible with vLLM

**Symptom**: `AssertionError: Expandable segments are not compatible with memory pool` at init.

**Cause**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` conflicts with vLLM's CuMemAllocator.

**Fix**: Do not set `expandable_segments:True` when using vLLM rollout.

### EFA on g5 instances

For g5 instances (no GPUDirect RDMA), ensure these environment variables are set:
```bash
export FI_EFA_USE_DEVICE_RDMA=0   # No GPUDirect RDMA on g5
export NCCL_PROTO=simple           # Required without GPUDirect RDMA
export NCCL_NET=ofi                # Use libfabric for EFA
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

Without `NCCL_NET=ofi` and the correct `LD_LIBRARY_PATH`, NCCL silently falls back to TCP, giving much worse inter-node bandwidth.
