<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Pi0 (LeRobot) Fine-tuning on SageMaker HyperPod EKS

Fine-tune [Physical Intelligence's π0](https://huggingface.co/lerobot/pi0_base), a 3B-parameter flow matching Vision-Language-Action (VLA) model, using [HuggingFace LeRobot](https://github.com/huggingface/lerobot)'s PyTorch implementation on Amazon SageMaker HyperPod with EKS orchestration.

π0 generates robot actions via flow matching — learning a velocity field that transports noise to actions via ODE integration. Unlike diffusion (10-20 denoising steps), π0 produces high-quality 50-step action chunks in as few as 1 ODE step, enabling sub-200ms inference on H100 GPUs.

## Layout

```
pi0-lerobot/
├── Dockerfile                        # Optional: pre-built image (faster startup)
├── buildspec.yml                     # AWS CodeBuild spec (if using pre-built image)
├── README.md                         # This file
├── src/
│   └── evaluate_pi0.py               # Evaluation script (MSE/MAE + ODE sweep)
└── kubernetes/
    ├── pvc-fsx-lustre.yaml           # FSx PVC (dynamic provisioning, reclaimPolicy: Retain)
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
  - **Note:** HyperPod labels instances with an `ml.` prefix (e.g., `ml.p5.48xlarge`). On plain EKS, the label is `p5.48xlarge`. The manifests use the HyperPod convention by default — edit the `nodeSelector` if running on plain EKS.
- The [Kubeflow Training Operator](https://github.com/kubeflow/training-operator) installed
- FSx for Lustre PVC (`fsx-claim`) mounted at `/fsx`
- A HuggingFace token that has accepted **both**:
  - [lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base) (Apache 2.0, but gated)
  - [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) (Gemma license — π0's processor downloads this at train time)

## Quick Start

The manifests use the [AWS DLC PyTorch image](https://github.com/aws/deep-learning-containers) directly and install LeRobot v0.5.0 at pod startup (~3-5 min). No custom image build required.

```bash
# 1. Install Training Operator + create secrets
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.9.1"
kubectl create secret generic pi0-lerobot-secrets --from-literal=HF_TOKEN="<your-token>"

# 2. Train DROID
kubectl apply -f kubernetes/droid/droid-finetune.yaml
kubectl logs -f pi0-lerobot-droid-finetune-worker-0

# 3. Evaluate DROID (after training completes)
kubectl delete pytorchjob pi0-lerobot-droid-finetune
kubectl apply -f kubernetes/droid/droid-eval.yaml

# 4. Train + Evaluate LIBERO
kubectl apply -f kubernetes/libero/libero-finetune.yaml
# ... wait for completion ...
kubectl delete pytorchjob pi0-lerobot-libero-finetune
kubectl apply -f kubernetes/libero/libero-eval.yaml
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base checkpoint | `lerobot/pi0_base` |
| LeRobot version | v0.5.0 (pinned) |
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

## Optional: Pre-built Container Image

For faster pod startup (30s vs 5min), you can build a custom image with LeRobot pre-installed. The `Dockerfile` and `buildspec.yml` are provided for this purpose.

```bash
cd pi0-lerobot

# One-time: create buildx builder + QEMU (for arm64 hosts)
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

Then update the `image:` field in the manifests to point to your ECR image.

## Troubleshooting

### Webhook connection refused

If `kubectl apply` fails with webhook errors, delete the stale webhook:
```bash
kubectl delete validatingwebhookconfiguration validator.training-operator.kubeflow.org
```

### flash_attn / libcudart.so.12

The DLC image ships CUDA 13; `flash_attn` is compiled for CUDA 12. The manifests uninstall it at startup — π0 doesn't use flash attention (only the XVLA policy does).

### MPIJob CRD missing

If the training operator crashes with "no matches for MPIJob", install the CRD:
```bash
kubectl delete crd mpijobs.kubeflow.org 2>/dev/null
kubectl apply -f https://raw.githubusercontent.com/kubeflow/training-operator/v1.9.1/manifests/base/crds/kubeflow.org_mpijobs.yaml
```

### FileExistsError on restart

LeRobot v0.5.0 raises `FileExistsError` if `--output_dir` exists. Set `FORCE_RESTART=1` env var on the finetune pod to delete and restart, or manually remove:
```bash
kubectl exec <pod> -- rm -rf /fsx/runs/pi0-droid/training
```

### HF_TOKEN / Gemma license

Training fails ~15 min in with a 403 when downloading `google/paligemma-3b-pt-224` if:
- `HF_TOKEN` is missing from the `pi0-lerobot-secrets` Secret
- The token hasn't accepted the [Gemma license](https://huggingface.co/google/gemma-2b)

## References

- [π0 paper](https://arxiv.org/abs/2410.24164) — Physical Intelligence, 2024
- [LeRobot pi0_base](https://huggingface.co/lerobot/pi0_base)
- [DROID dataset](https://droid-dataset.github.io/)
- [LIBERO benchmark](https://libero-project.github.io/)
- [LeRobot training docs](https://huggingface-lerobot.mintlify.app/api/scripts/train)
