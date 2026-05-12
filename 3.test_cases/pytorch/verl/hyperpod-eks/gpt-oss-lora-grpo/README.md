# GPT-OSS 20B Training & Inference Guide

This guide explains how to train the GPT-OSS 20B model with LoRA, then improve it using Group Relative Policy Optimization (GRPO) for better language compliance in reasoning and final answers. It is designed to be as simple as possible, requires no data preparation, and uses a container image. For further background information look at https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transformers

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GPT-OSS 20B Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Base Model  │───▶│  SFT LoRA    │───▶│  GRPO LoRA   │              │
│  │  (20B)       │    │  Training    │    │  Training    │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│   inference-base     inference-trained    inference-grpo               │
└─────────────────────────────────────────────────────────────────────────┘
```

## 0. Prerequisites

### 0.1. EKS Cluster

Before running this training, you'll need to create an Amazon EKS or a SageMaker HyperPod EKS cluster. Instructions can be found in [1.architectures](../../../../1.architectures), the [aws-do-eks](https://github.com/aws-samples/aws-do-eks) project, or the [eks-blueprints](https://github.com/aws-ia/terraform-aws-eks-blueprints) project.

### 0.2. Connect to your EKS Cluster

Run the [aws eks update-kubeconfig](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/eks/update-kubeconfig.html) command to update your local kube config file with the credentials needed to connect to your EKS cluster.

```bash
aws eks update-kubeconfig --name <EKS_CLUSTER_NAME>
```

Verify connection:

```bash
kubectl config current-context
```

```
arn:aws:eks:us-east-2:xxxxxxxxxxxx:cluster/xxx-eks-cluster
```

### 0.3. Clone the repository

```bash
git clone https://github.com/awslabs/awsome-distributed-training/
```

## 1. Build container image

Copy `env_vars.example` to `env_vars` and update with your values:
```bash
cp env_vars.example env_vars
# Edit env_vars with your AWS account, region, and HuggingFace token
```

Alternative to Docker Desktop (macOS):
```bash
brew install colima docker
colima start
docker context use colima
```

Use `/artifacts/build_push.sh` to build and push the image to Amazon ECR.
```bash
source env_vars
# Build the shared TRL base image first
docker build -t trl-base:latest ../../../
# Build and push the GPT-OSS image
cd artifacts
./build_push.sh
cd ..
```

### 1.1. Deploy FSx Storage Manager

```bash
envsubst '$REGISTRY $IMAGE $TAG $HF_TOKEN' < kubernetes/fsx-storage-manager.yaml | kubectl apply -f -
```


## 2. Data

For this example, we use the [HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) dataset. This is a reasoning dataset where the chain-of-thought has been translated into several languages (French, Spanish, German, Italian).

**For this dataset, you need a Hugging Face access token**. First, create a [Hugging Face account](https://huggingface.co/welcome). Then [generate your access token with read permissions](https://huggingface.co/docs/hub/en/security-tokens).

## 3. Launch SFT LoRA training job

Apply the training manifest:

```bash
envsubst '$REGISTRY $IMAGE $TAG $HF_TOKEN' < hyperpod-eks/train-lora-hyperpod-elastic-g6e.yaml | kubectl apply -f -
```

### 3.1. Monitor training job

```bash
kubectl get pods -l app=gpt-oss-lora-elastic
kubectl logs -f -l app=gpt-oss-lora-elastic
kubectl logs -f lora-hyperpod-elastic-worker-<pod-id>
kubectl exec -it fsx-storage-manager -- ls -la /fsx/checkpoints/
```

Example output:

```log
{'loss': 1.035, 'grad_norm': 0.163, 'learning_rate': 0.0002, 'mean_token_accuracy': 0.716, 'epoch': 3.25}
{'loss': 1.003, 'grad_norm': 0.171, 'learning_rate': 0.0002, 'mean_token_accuracy': 0.722, 'epoch': 3.32}
{'loss': 0.976, 'grad_norm': 0.159, 'learning_rate': 0.0002, 'mean_token_accuracy': 0.726, 'epoch': 3.57}
```

### 3.2. Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `openai/gpt-oss-20b` |
| Dataset | `HuggingFaceH4/Multilingual-Thinking` |
| LoRA Rank | 8 |
| Max Steps | 1000 |
| Distributed | FSDP (16 GPUs) |

### 3.3. Convert checkpoint

After training completes, convert FSDP checkpoint to PEFT format:

```bash
kubectl exec -it fsx-storage-manager -- python /app/convert_fsdp_checkpoint.py \
    --checkpoint /fsx/checkpoints/checkpoint-1000 \
    --output /fsx/checkpoints/converted-peft/lora-checkpoint-1000-peft
```

### 3.4. Stop training job

```bash
kubectl delete hyperpodpytorchjob lora-hyperpod-elastic
```

## 4. Launch GRPO training job

GRPO improves model behavior by generating K=8 completions per prompt, scoring each with a reward function, and using relative rewards to update the policy.

### 4.1. Start training

> **Note:** Before running, update `--peft_checkpoint` in `kubernetes/train-grpo-singlenode.yaml` to point to your converted SFT checkpoint from step 3.3.

```bash
envsubst '$REGISTRY $IMAGE $TAG $HF_TOKEN' < kubernetes/train-grpo-singlenode.yaml | kubectl apply -f -
```

### 4.2. Monitor training job

```bash
kubectl logs grpo-single --tail=50
```

After training completes, the checkpoint is saved to `/fsx/checkpoints/grpo-singlenode/`.

### 4.3. Configuration

| Parameter | Value |
|-----------|-------|
| SFT Checkpoint | `converted-peft/lora-checkpoint-1000-peft` |
| K (generations) | 8 |
| Epochs | 10 |
| Learning Rate | 1e-6 |

### 4.4. Reward function

| Component | Correct | Wrong |
|-----------|---------|-------|
| Answer language | +5.0 | -5.0 |
| Reasoning language | +1.5 | -1.5 |
| Answer brevity | +0.5 | -1.0 |

### 4.5. Stop training job

```bash
kubectl delete pod grpo-single
```

## 5. Run inference

### 5.1. Models available

| Model | Description | Reasoning Language |
|-------|-------------|-------------------|
| Base | Original GPT-OSS 20B | ❌ Ignores |
| SFT | Base + LoRA fine-tuned | ✅ Follows |
| GRPO | SFT + GRPO LoRA | ✅✅ Better |

### 5.2. Deploy inference pods

```bash
envsubst '$REGISTRY $IMAGE $TAG $HF_TOKEN' < kubernetes/inference-g6e-base.yaml | kubectl apply -f -
envsubst '$REGISTRY $IMAGE $TAG $HF_TOKEN' < kubernetes/inference-g6e-trained.yaml | kubectl apply -f -
envsubst '$REGISTRY $IMAGE $TAG $HF_TOKEN' < kubernetes/inference-g6e-grpo.yaml | kubectl apply -f -
```


### 5.3. Run inference commands

> **Note:** Replace checkpoint paths below with your actual checkpoint paths from training.

**Base Model:**

```bash
kubectl exec -it inference-base -- python /app/inference_g6e.py
```

**SFT Model:**

```bash
kubectl exec -it inference-trained -- python /app/inference_g6e.py \
    --checkpoint /fsx/checkpoints/converted-peft/lora-checkpoint-1000-peft \
    --reasoning_language Spanish
```

**GRPO Model:**

```bash
kubectl exec -it inference-grpo -- python /app/inference_grpo_new.py \
    --use_grpo \
    --grpo_checkpoint /fsx/checkpoints/converted-peft/checkpoint-1470-peft \
    --reasoning_language Spanish
```

### 5.4. Command options

| Option | Description |
|--------|-------------|
| `--checkpoint` | Path to PEFT checkpoint |
| `--reasoning_language` | Language for reasoning |
| `--use_grpo` | Enable GRPO model (inference_grpo_new.py) |
| `--grpo_checkpoint` | Path to GRPO checkpoint |
| `--prompt` | Single prompt (non-interactive) |

## 6. Evaluation

Evaluates whether the fine-tuned models reason in the specified language. Tests 10 prompts across 5 languages (English, French, German, Spanish, Italian) - 50 test cases total. Measures language compliance in both reasoning chain and final answer.

### 6.1. Run evaluation

```bash
envsubst '$REGISTRY $IMAGE $TAG $HF_TOKEN' < kubernetes/eval-grpo.yaml | kubectl apply -f -
```

### 6.2. Check results

```bash
kubectl logs eval-grpo --tail=50
kubectl exec fsx-storage-manager -- cat /fsx/experiments/grpo_eval_checkpoint{checkpoint_number}_{timestamp}.txt
```

## References

### Checkpoints

```
/fsx/checkpoints/converted-peft/
├── lora-checkpoint-1000-peft/   # SFT checkpoint
└── checkpoint-1470-peft/        # GRPO checkpoint
```

### Files Reference

| File | Purpose |
|------|---------|
| `artifacts/Dockerfile` | Container image definition |
| `artifacts/build_push.sh` | Build and push image to ECR |
| `artifacts/requirements.txt` | Python dependencies |
| `artifacts/src/sft.py` | SFT training script |
| `artifacts/src/grpo_singlenode.py` | GRPO training script |
| `artifacts/src/convert_fsdp_checkpoint.py` | FSDP to PEFT converter |
| `artifacts/src/convert_grpo_checkpoint.py` | GRPO checkpoint converter |
| `artifacts/src/evaluate_grpo.py` | GRPO evaluation script |
| `artifacts/src/inference_g6e.py` | Base/SFT inference script |
| `artifacts/src/inference_grpo_new.py` | GRPO inference script |
| `artifacts/src/sft_teacher_data.py` | Teacher data generation script |
| `artifacts/src/configs/sft_lora.yaml` | SFT LoRA training config |
| `hyperpod-eks/train-lora-hyperpod-elastic-g6e.yaml` | SFT training job manifest (HyperPod) |
| `kubernetes/train-grpo-singlenode.yaml` | GRPO training pod manifest |
| `kubernetes/fsx-storage-manager.yaml` | FSx storage manager pod |
| `kubernetes/inference-g6e-base.yaml` | Base model inference pod |
| `kubernetes/inference-g6e-trained.yaml` | SFT model inference pod |
| `kubernetes/inference-g6e-grpo.yaml` | GRPO model inference pod |
| `kubernetes/eval-grpo.yaml` | Evaluation pod spec |
| `env_vars.example` | Environment variables template |
