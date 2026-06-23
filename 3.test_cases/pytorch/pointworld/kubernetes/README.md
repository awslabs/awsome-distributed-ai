<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# PointWorld on Amazon EKS / SageMaker HyperPod EKS

This directory contains the Kubernetes manifests for pre-training and evaluating
[PointWorld](https://github.com/NVlabs/PointWorld) on an EKS-orchestrated GPU
cluster using the Kubeflow PyTorchJob operator.

| Manifest | Kind | Purpose |
|---|---|---|
| [`pointworld-pretrain.yaml`](./pointworld-pretrain.yaml) | `PyTorchJob` | Multi-node DDP pre-training (DROID + BEHAVIOR, large PTv3) |
| [`pointworld-eval.yaml`](./pointworld-eval.yaml) | `Job` | Single-GPU evaluation of a trained or released checkpoint |

See the [top-level README](../README.md) for the model overview, container build,
and the full data pipeline. This page covers only the Kubernetes specifics.

## Prerequisites

- An EKS or SageMaker HyperPod EKS cluster with **8 x p5en.48xlarge** (64 x H200)
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/)
  (provides the `PyTorchJob` CRD)
- [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) (`nvidia.com/gpu`)
- [EFA device plugin](https://github.com/aws-samples/aws-efa-eks) (`vpc.amazonaws.com/efa`)
- An [FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html)
  PVC named `fsx-claim` mounted at `/fsx`
- The container image built from [`../pointworld.Dockerfile`](../pointworld.Dockerfile)
  and pushed to Amazon ECR

## Expected FSx layout

Both manifests assume this layout on the shared filesystem:

```text
/fsx/pointworld/
├── dataset/
│   ├── droid/wds/{train,test}/...          # from scripts/1.convert_wds.py
│   │   └── test/expert_confidence-seed=42.h5   # for DROID filtered metrics
│   └── behavior/wds/{train,test}/...
├── dinov3/checkpoints/
│   └── dinov3_vitl16_pretrain.pth          # gated; from scripts/2.download_dinov3.sh
├── train_logs/                             # written during pre-training
└── checkpoints/                            # written during pre-training
```

The WDS root is mounted into the pod at `/dataset` via `subPath: pointworld/dataset`,
so PointWorld's `LOCAL_DATASET_DIR=/dataset` resolves `/dataset/droid/wds` and
`/dataset/behavior/wds`.

## Push the image to ECR

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1   # adjust to your cluster region
REPO=pointworld
TAG=05484826        # matches POINTWORLD_COMMIT in the Dockerfile

aws ecr create-repository --repository-name ${REPO} --region ${REGION} || true
aws ecr get-login-password --region ${REGION} \
  | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

docker build -t ${REPO}:${TAG} -f ../pointworld.Dockerfile ..
docker tag ${REPO}:${TAG} ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}
```

Then replace `<ACCOUNT_ID>` and `<REGION>` in both manifests with your values.

## Run pre-training

```bash
kubectl apply -f pointworld-pretrain.yaml
kubectl get pytorchjob -n kubeflow
kubectl logs -f pointworld-pretrain-worker-0 -n kubeflow
```

The job launches 8 worker pods (one per node), each running `torchrun` with 8
processes (one per H200). The PyTorchJob operator injects `MASTER_ADDR`,
`MASTER_PORT`, `WORLD_SIZE`, and `RANK`; PointWorld's `Trainer` initializes the
process group with `init_method="env://"` and reads `LOCAL_RANK` directly, so no
extra launcher is needed.

### Launch pattern detail

The container `command` is a small `bash -c` wrapper that:

1. Symlinks the gated DINOv3 weights from `/fsx/pointworld/dinov3/checkpoints`
   into `/pointworld/third_party/dinov3/checkpoints` (keeping gated weights out
   of the image), then
2. `exec`s `torchrun ... train.py` with the flagship DROID + BEHAVIOR flags.

Doing the symlink in the main container (rather than an initContainer) ensures
it lives in the same filesystem namespace as the training process.

## Run evaluation

Evaluation is single-process, so it uses a plain `batch/v1` Job with one GPU:

```bash
# Edit MODEL_PATH in pointworld-eval.yaml to point at your checkpoint, or a
# released checkpoint downloaded from nvidia/PointWorld_models.
kubectl apply -f pointworld-eval.yaml
kubectl logs -f job/pointworld-eval -n kubeflow
```

The DROID metric of interest is `full_eval/test/filtered_l2_moved/mean`, which
uses the released expert-confidence artifact to focus on reliable moving-point
regions. For a quick smoke test, set `EVAL_NUM_BATCHES` to e.g. `100`.

## Tuning notes

- **Batch size**: `--batch_size=22` is the upstream default. H200 has 141 GB
  HBM; raise it if memory allows, and update `--global_batch_size` accordingly
  when parsing throughput with `../scripts/parse_benchmark.py`.
- **EFA**: `vpc.amazonaws.com/efa: 16` matches p5en.48xlarge (16 EFA network
  cards per node, as advertised by the EFA device plugin). Adjust for other
  instance types (for example p5.48xlarge advertises 32).
- **Shared memory**: the `shmem` `emptyDir` backs the PyTorch DataLoader workers;
  reduce `sizeLimit` for smaller instances.
- **B200**: this image targets H200. B200 EFA needs NCCL >= 2.29 — see
  "Known Limitations" in the [top-level README](../README.md).
