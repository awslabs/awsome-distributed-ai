<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Pi0 (LeRobot) Fine-tuning on SageMaker HyperPod EKS

Fine-tune [Physical Intelligence's π0](https://huggingface.co/lerobot/pi0_base), a 3B-parameter flow matching Vision-Language-Action (VLA) model, using [HuggingFace LeRobot](https://github.com/huggingface/lerobot)'s PyTorch implementation on Amazon SageMaker HyperPod with EKS orchestration.

π0 generates robot actions via flow matching — learning a velocity field that transports noise to actions via ODE integration. Unlike diffusion (10-20 denoising steps), π0 produces high-quality 50-step action chunks in as few as 1 ODE step, enabling sub-200ms inference on H100 GPUs.

## Layout

```
pi0-lerobot/
├── Dockerfile                        # Container: DLC PyTorch 2.9 + LeRobot + π0
├── buildspec.yml                     # AWS CodeBuild spec for image builds
├── README.md                         # This file
├── src/
│   ├── run_finetuning.sh             # Training entrypoint (accelerate + FSDP)
│   ├── evaluate_pi0.py               # Evaluation script (MSE/MAE + ODE sweep)
│   ├── evaluate_pi0.sh               # Eval entrypoint wrapper
│   └── lerobot_local_patch.py        # Hub-skip patch for FSx-local datasets
└── kubernetes/
    ├── pvc-fsx-lustre.yaml           # FSx PVC (dynamic provisioning, if needed)
    ├── droid/
    │   ├── README.md                 # DROID walkthrough
    │   ├── droid-download.yaml       # Dataset staging Job
    │   ├── droid-finetune.yaml       # Training PyTorchJob (8× H100, FSDP)
    │   └── droid-eval.yaml           # Evaluation Job (1× GPU, ODE sweep)
    └── libero/
        ├── README.md                 # LIBERO walkthrough
        ├── libero-download.yaml      # Dataset staging Job
        ├── libero-finetune.yaml      # Training PyTorchJob
        └── libero-eval.yaml          # Evaluation Job
```

## Results (p5.48xlarge — 8× H100 80GB)

| Dataset | Base MSE | Fine-Tuned MSE | Improvement | Latency (1 ODE step) |
|---------|----------|----------------|-------------|---------------------|
| **DROID** | 3.897e-01 | 1.353e-02 | **96.5%** | 199 ms |
| **LIBERO** | 7.214e-01 | 5.003e-02 | **93.1%** | 197 ms |

Training time: ~6.5 hours per dataset (20K steps, FSDP FULL_SHARD, 8× H100).

## Prerequisites

- A SageMaker HyperPod EKS cluster with GPU nodes (p5.48xlarge or p4de.24xlarge)
- The [Kubeflow Training Operator](https://github.com/kubeflow/training-operator) installed
- FSx for Lustre PVC (`fsx-claim`) mounted at `/fsx`
- A HuggingFace token (for base model download)

## Quick Start

```bash
# 1. Build and push container image
#    (see "Container Image" section below)

# 2. Install Training Operator + create secrets
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.9.1"
kubectl create secret generic pi0-lerobot-secrets --from-literal=HF_TOKEN="<your-token>"

# 3. Train DROID
kubectl apply -f kubernetes/droid/droid-finetune.yaml
kubectl logs -f pi0-lerobot-droid-finetune-worker-0

# 4. Evaluate DROID (after training completes)
kubectl delete pytorchjob pi0-lerobot-droid-finetune
kubectl apply -f kubernetes/droid/droid-eval.yaml

# 5. Train + Evaluate LIBERO
kubectl apply -f kubernetes/libero/libero-finetune.yaml
# ... wait for completion ...
kubectl delete pytorchjob pi0-lerobot-libero-finetune
kubectl apply -f kubernetes/libero/libero-eval.yaml
```

## Container Image

The Dockerfile builds against the AWS Deep Learning Container (DLC) PyTorch training image so that PyTorch, NCCL, EFA, and libfabric are version-pinned together by AWS.

The build uses `docker buildx build --platform linux/amd64 --load` because every AWS GPU instance this recipe targets is x86_64. A plain `docker build` on Apple Silicon produces an arm64 manifest and the kubelet rejects the pull.

```bash
cd pi0-lerobot

# One-time: create buildx builder + QEMU
docker buildx create --use --name pi0-builder
docker run --privileged --rm tonistiigi/binfmt --install all

export AWS_REGION=${AWS_REGION:-$(aws configure get region)}
export REGISTRY=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.${AWS_REGION}.amazonaws.com
export IMAGE_TAG=v1.0.0

# Login to DLC registry (base image pull)
aws ecr get-login-password --region ${AWS_REGION} \
  | docker login --username AWS --password-stdin \
      763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build
docker buildx build --platform linux/amd64 \
  --build-arg AWS_REGION=${AWS_REGION} \
  --load -f Dockerfile -t ${REGISTRY}/pi0-lerobot:${IMAGE_TAG} .

# Push
aws ecr create-repository --repository-name pi0-lerobot --region ${AWS_REGION} 2>/dev/null || true
aws ecr get-login-password --region ${AWS_REGION} \
  | docker login --username AWS --password-stdin ${REGISTRY}
docker image push ${REGISTRY}/pi0-lerobot:${IMAGE_TAG}
```

### Alternative: AWS CodeBuild

Upload source to S3 and use CodeBuild for native x86_64 builds (no QEMU):

```bash
zip -r pi0-lerobot-source.zip Dockerfile src/ buildspec.yml
aws s3 cp pi0-lerobot-source.zip s3://<bucket>/codebuild/pi0-lerobot-source.zip
aws codebuild start-build --project-name pi0-lerobot-build --region ${AWS_REGION}
```

### Alternative: DLC Direct (no custom image)

For quick iteration, you can use the DLC image directly and install LeRobot at pod startup (~5 min overhead). See `droid-finetune.yaml` for the pattern — change the `image:` field to:

```yaml
image: 763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker-v1.9
```

Then add `pip install "lerobot[pi,dataset]@git+..."` to the args block before training.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base checkpoint | `lerobot/pi0_base` |
| Training steps | 20,000 |
| Batch size (per GPU) | 4 |
| Number of GPUs | 8 (H100 80GB) |
| Effective batch size | 32 |
| Learning rate | 2.5e-5 |
| Action horizon | 50 |
| Precision | bf16 (mixed via FSDP) |
| Sharding | FSDP FULL_SHARD |
| Gradient checkpointing | Enabled |
| Checkpoint interval | Every 2,000 steps |

## Key Differences: HyperPod EKS vs SageMaker Training Jobs

| Aspect | SageMaker Training Job | HyperPod EKS (this recipe) |
|--------|----------------------|---------------------------|
| Data | Downloaded from S3 per job | Persists on FSx across jobs |
| Checkpoints | Archived to model.tar.gz on S3 | Stay on FSx (no tar/untar) |
| Orchestration | ModelTrainer API (Python SDK) | kubectl + PyTorchJob YAML |
| Multi-node | instance_count parameter | Set `replicas` in PyTorchJob |
| Debugging | CloudWatch logs only | kubectl exec into running pod |
| Iteration speed | ~5 min cold start per job | Instant (data on FSx, image cached) |

## Troubleshooting

### Webhook connection refused

If `kubectl apply` fails with webhook errors, delete the stale webhook:
```bash
kubectl delete validatingwebhookconfiguration validator.training-operator.kubeflow.org
```

### flash_attn / libcudart.so.12

The DLC image ships CUDA 13; flash_attn needs CUDA 12. The training YAML uninstalls flash_attn at startup (π0 doesn't use it — only the XVLA policy does).

### MPIJob CRD missing

If the training operator crashes with "no matches for MPIJob", install the CRD:
```bash
kubectl apply -f https://raw.githubusercontent.com/kubeflow/training-operator/v1.9.1/manifests/base/crds/kubeflow.org_mpijobs.yaml
```

## References

- [π0 paper](https://arxiv.org/abs/2410.24164) — Physical Intelligence, 2024
- [LeRobot pi0_base](https://huggingface.co/lerobot/pi0_base)
- [DROID dataset](https://droid-dataset.github.io/)
- [LIBERO benchmark](https://libero-project.github.io/)
- [LeRobot training docs](https://huggingface-lerobot.mintlify.app/api/scripts/train)
