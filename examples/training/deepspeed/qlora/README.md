# Fine-tuning Qwen3-8B with QLoRA on SageMaker HyperPod

A comprehensive tutorial for fine-tuning the Qwen3-8B large language model using QLoRA (Quantized Low-Rank Adaptation) with multi-GPU support via DDP + DeepSpeed ZeRO-2 or ZeRO-3.

## Overview

This project demonstrates how to:

- Fine-tune Qwen3-8B using QLoRA with DDP + DeepSpeed ZeRO-2 **or ZeRO-3**
- Deploy on SageMaker HyperPod with **EKS** or **Slurm** orchestrator
- Train on a medical reasoning dataset for clinical Q&A
- Use the LoRA adapter for inference (4-bit or merged full-precision)
- (Optional) Use NVIDIA MIG to run parallel experiments on partitioned GPUs


## Key Features

- **Memory Efficient**: QLoRA 4-bit training + DeepSpeed ZeRO-2/3 optimizer sharding across 4 GPUs
- **Crash Resilient**: Automatic checkpoint resume + HyperPod auto-resume on node failure
- **Production Ready**: Containerized training on Kubernetes or Slurm with `torchrun` multi-GPU launcher
- **Health Monitoring**: HyperPod deep health checks (GPU/NCCL) with automatic faulty node replacement
- **ZeRO-2 & ZeRO-3**: Switch between strategies with a single environment variable

## Quick Start

### Prerequisites

- AWS Account with appropriate IAM permissions
- A SageMaker HyperPod cluster (EKS or Slurm orchestrator)
- (Optional) For MIG: A100 or H100 GPUs with MIG-capable drivers

### Option 1: Deploy on HyperPod (EKS)

Requires a HyperPod EKS cluster with the HyperPod Helm chart installed (bundles Training Operator, NVIDIA device plugin, and health monitoring agents) and an FSx for Lustre filesystem.

```bash
# 1. Build and push Docker image to ECR
./0.build-image.sh

# 2. Set required environment variables
export IMAGE=<account>.dkr.ecr.<region>.amazonaws.com/qwen3-qlora-training:latest
export NUM_GPUS=4
export INSTANCE_TYPE=ml.g5.12xlarge
export FSX_FILESYSTEM_ID=fs-0123456789abcdef0
export FSX_DNS_NAME=fs-0123456789abcdef0.fsx.us-east-1.amazonaws.com
export FSX_MOUNT_NAME=abcdef01

# 3. Deploy training job (single-node, ZeRO-2 by default)
./1.deploy-training.sh

# 4. Monitor training
kubectl logs -f qwen3-qlora-training-zero2-master-0 -n ml-training

# --- Multi-node: 2 nodes x 4 GPUs = 8 GPUs total ---

NUM_NODES=2 ./1.deploy-training.sh

# --- OR deploy with ZeRO-3 for lower per-GPU memory ---

DEEPSPEED_STRATEGY=zero3 ./1.deploy-training.sh
kubectl logs -f qwen3-qlora-training-zero3-master-0 -n ml-training
```

### Option 2: Deploy on HyperPod (Slurm)

```bash
# 1. Clone the repository to the shared filesystem
cd /fsx
git clone https://github.com/awslabs/awsome-distributed-training.git
cd awsome-distributed-training/examples/pytorch/deepspeed/qlora

# 2. Create virtual environment and install dependencies
python3 -m venv /fsx/venvs/qwen3-qlora
source /fsx/venvs/qwen3-qlora/bin/activate
pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 3. Submit training job (ZeRO-2)
cd slurm
sbatch qwen3_8b-qlora-zero2.sbatch

# --- OR submit ZeRO-3 ---
sbatch qwen3_8b-qlora-zero3.sbatch

# 4. Monitor training
tail -f logs/qwen3_8b-qlora-zero2_<job-id>.out
```

HyperPod auto-resume, MIG auto-detection, and container support are built in. See [slurm/README.md](slurm/README.md) for full instructions including MIG setup.

### Option 3: Local Training (GPU Required)

```bash
# Install dependencies
pip install -r requirements.txt

# Single-GPU
python src/train.py --config_path configs/training_config_zero2.yaml --output_dir ./outputs

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 src/train.py \
    --config_path configs/training_config_zero2.yaml \
    --output_dir ./outputs
```

## Project Structure

```
qlora/
├── README.md                          # This file
├── Dockerfile                         # Training container image
├── entrypoint.sh                      # Container entrypoint (handles torchrun)
├── requirements.txt                   # Python dependencies
├── 0.build-image.sh                   # Build & push Docker image to ECR
├── 1.deploy-training.sh               # Deploy PyTorchJob to HyperPod EKS
├── 2.cleanup.sh                       # Delete training resources
│
├── configs/                           # Configuration files
│   ├── training_config_zero2.yaml      # Training hyperparameters (ZeRO-2)
│   ├── training_config_zero3.yaml     # Training hyperparameters (ZeRO-3)
│   ├── deepspeed_zero2.json           # DeepSpeed ZeRO-2 config
│   └── deepspeed_zero3.json           # DeepSpeed ZeRO-3 config
│
├── kubernetes/                        # HyperPod EKS manifests
│   ├── README.md                      # HyperPod EKS deployment guide
│   ├── storage.yaml                   # FSx for Lustre PV + PVC
│   ├── qwen3_8b-qlora-zero2.yaml      # PyTorchJob manifest (ZeRO-2)
│   └── qwen3_8b-qlora-zero3.yaml     # PyTorchJob manifest (ZeRO-3)
│
├── slurm/                             # HyperPod Slurm scripts
│   ├── README.md                      # HyperPod Slurm + MIG setup guide
│   ├── qwen3_8b-qlora-zero2.sbatch    # Slurm batch script (ZeRO-2, MIG-aware)
│   └── qwen3_8b-qlora-zero3.sbatch   # Slurm batch script (ZeRO-3, MIG-aware)
│
├── src/                               # Python training code
│   ├── __init__.py
│   ├── config.py                      # Configuration dataclasses
│   ├── model_setup.py                 # QLoRA model loading + LoRA setup
│   ├── data_preparation.py            # Dataset loading + tokenization
│   ├── train.py                       # Main training script
│   └── inference_demo.py              # Inference with LoRA adapter
│
└── docs/                              # Documentation
    ├── QLORA_EXPLAINED.md             # QLoRA concepts & code walkthrough
    └── TROUBLESHOOTING.md             # Common issues & solutions
```

## Model & Dataset

### Qwen3-8B

- **Parameters**: 8.2B (6.95B non-embedding)
- **Context Length**: 32,768 tokens
- **Architecture**: 36 layers, GQA attention
- **License**: Apache 2.0

### Medical Reasoning Dataset

- **Source**: [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- **Size**: 19.7k samples (English)
- **Format**: Question + Chain-of-Thought + Response

## QLoRA Configuration

```yaml
# 4-bit Quantization
load_in_4bit: true
bnb_4bit_quant_type: "nf4"
bnb_4bit_use_double_quant: true

# LoRA Adapters
lora_r: 16
lora_alpha: 32
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 1 | Per-device batch size |
| Gradient Accumulation | 4 | Steps before weight update |
| Effective Batch Size | 16 | 1 x 4 GPUs x 4 accumulation |
| Learning Rate | 2e-4 | Peak learning rate |
| Max Sequence Length | 1536 | Tokens per sample |
| Epochs | 3 | Training epochs |
| Optimizer | paged_adamw_8bit | Memory-efficient |
| Precision | bfloat16 | Mixed precision |

## Infrastructure

| Component | Spec | Notes |
|-----------|------|-------|
| GPU Node | 1x ml.g5.12xlarge | Single node with 4x NVIDIA A10G GPUs (24GB VRAM each) |
| Storage | FSx for Lustre (ReadWriteMany) | Shared across nodes; survives node replacements |
| Orchestrator | HyperPod Helm chart | Bundles Training Operator, NVIDIA device plugin, health agents |
| Health Monitoring | HyperPod deep health checks | GPU/NCCL checks; faulty nodes auto-replaced |
| GPU Launcher | torchrun | `--nproc_per_node=4`, spawns 1 process per GPU |

> **Why g5.12xlarge?** Training Qwen3-8B with QLoRA on a single A10G (24GB)
> hits ~99% VRAM utilization and crashes at checkpoint-save points due to memory
> spikes. The g5.12xlarge provides 4x A10G GPUs **within a single machine**
> (intra-node communication, no cross-node networking overhead), allowing DDP +
> DeepSpeed to shard memory across GPUs.

## DeepSpeed ZeRO-2 vs ZeRO-3

This project supports two DeepSpeed strategies. Both use DDP underneath for data parallelism.

### ZeRO-2 (Default)

Shards **optimizer states and gradients** across GPUs. Each GPU still holds the full model parameters.

```
Strategy:  deepspeed_zero2
Config:    configs/training_config_zero2.yaml
Per-GPU:   ~15-17 GB
```

### ZeRO-3

Shards **optimizer states, gradients, AND model parameters** across GPUs. Maximum memory savings.

```
Strategy:  deepspeed_zero3
Config:    configs/training_config_zero3.yaml
Per-GPU:   ~12-14 GB
```

### Per-GPU Memory Comparison

| Component | Single GPU | 4x GPU + ZeRO-2 | 4x GPU + ZeRO-3 |
|-----------|-----------|------------------|------------------|
| Base model (4-bit) | ~5 GB | ~5 GB | **~1.25 GB** |
| LoRA adapters | ~0.1 GB | ~0.1 GB | ~0.1 GB |
| Activations | ~8 GB | ~5 GB | ~5 GB |
| Optimizer states | ~3 GB | **~0.75 GB** | **~0.75 GB** |
| Gradients | ~2 GB | **~0.5 GB** | **~0.5 GB** |
| Misc / NCCL buffers | ~2 GB | ~3 GB | ~4 GB |
| **Total** | **~20 GB** (of 24) | **~15 GB** (of 24) | **~12 GB** (of 24) |

### When to Use ZeRO-3

- Your model doesn't fit with ZeRO-2 (OOM errors)
- You want headroom for longer sequence lengths or larger batch sizes
- You're willing to accept slightly more communication overhead

### Switching Strategies

```bash
# HyperPod EKS: set env var before deploying
DEEPSPEED_STRATEGY=zero3 ./1.deploy-training.sh

# HyperPod EKS: multi-node (2 nodes) with ZeRO-3
NUM_NODES=2 DEEPSPEED_STRATEGY=zero3 ./1.deploy-training.sh

# HyperPod Slurm: submit the zero3 sbatch
sbatch slurm/qwen3_8b-qlora-zero3.sbatch

# Local: point to the zero3 config
torchrun --nproc_per_node=4 src/train.py \
    --config_path configs/training_config_zero3.yaml \
    --output_dir ./outputs-zero3
```

## Inference

Use `src/inference_demo.py` to load the fine-tuned LoRA adapter:

```bash
# 4-bit quantized inference (~8GB VRAM)
python src/inference_demo.py \
    --adapter_path ./outputs/final_model \
    --question "What are the symptoms of pneumonia?"

# Full-precision merged inference (~16GB VRAM)
python src/inference_demo.py \
    --adapter_path ./outputs/final_model \
    --merge
```

## Cleanup

Delete training jobs when done:

```bash
./2.cleanup.sh
```

To also delete the ECR repository:
```bash
aws ecr delete-repository --repository-name qwen3-qlora-training --region $AWS_REGION --force
```

## Documentation

- [HyperPod EKS Deployment Guide](kubernetes/README.md) - Kubernetes deployment instructions
- [HyperPod Slurm + MIG Guide](slurm/README.md) - Slurm deployment with MIG instructions
- [QLoRA Explained](docs/QLORA_EXPLAINED.md) - Understanding QLoRA
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Requirements

- Python 3.10+
- PyTorch >= 2.6.0
- Transformers >= 4.51.0 (required for Qwen3)
- PEFT >= 0.10.0
- BitsAndBytes >= 0.42.0
- DeepSpeed >= 0.13.0

## License

Apache 2.0

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the Qwen3 model
- [FreedomIntelligence](https://github.com/FreedomIntelligence) for the medical dataset
- [Hugging Face](https://huggingface.co/) for transformers and PEFT
- [Tim Dettmers](https://github.com/TimDettmers) for bitsandbytes
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) for distributed training optimization
