<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# PointWorld — Kubernetes manifests

Manifests for pre-training and evaluating
[PointWorld](https://github.com/NVlabs/PointWorld) on Amazon EKS / SageMaker
HyperPod EKS.

| Manifest | Kind | Purpose |
|---|---|---|
| [`build-image.yaml`](./build-image.yaml) | `Pod` (Kaniko) | Build the container on an x86_64 node and push to ECR (run via [`build-image.sh`](./build-image.sh)) |
| [`pointworld-data-prep.yaml`](./pointworld-data-prep.yaml) | `Job` | Stage DINOv3 weights + WebDataset shards onto FSx (runs in-cluster) |
| [`trainer-v2/`](./trainer-v2/) | `TrainJob` + `ClusterTrainingRuntime` | **Multi-node DDP pre-training on Kubeflow Trainer v2 — the validated default path** (BEHAVIOR, large PTv3) |
| [`pointworld-pretrain.yaml`](./pointworld-pretrain.yaml) | `PyTorchJob` | Multi-node DDP pre-training on the classic Training Operator (PyTorchJob v1) — **alternative path, not validated multi-node** (see note below) |
| [`pointworld-eval.yaml`](./pointworld-eval.yaml) | `Job` | Single-GPU evaluation of a trained or released checkpoint |

Build the image on the cluster (linux/amd64), then render+apply manifests with
`${VAR}` tokens from the test case's `env_vars` file:

```bash
source ../env_vars                            # after: cp ../env_vars.template ../env_vars
./build-image.sh                              # Kaniko build -> ECR (run once)
./deploy.sh trainer-v2/pointworld-runtime.yaml   # validated default: Trainer v2 runtime + TrainJob
./deploy.sh trainer-v2/pointworld-trainjob.yaml
./deploy.sh --dry-run <manifest>.yaml         # preview rendered YAML
./deploy.sh --delete  <manifest>.yaml         # tear down
```

> [!important] Single source of truth
> All instructions — prerequisites, building and pushing the image, staging data
> onto FSx, running pre-training and evaluation, the FSx layout, and tuning notes
> — live in the **[top-level README](../README.md)**. This page is just the
> manifest index.

> [!warning] Pre-training path: Trainer v2 is the validated default
> The end-to-end validation (1-node smoke + 2-node / 16-GPU run on p5en/H200) was
> performed on **Kubeflow Trainer v2** ([`trainer-v2/`](./trainer-v2/),
> `trainer.kubeflow.org`). Use it if your cluster runs Trainer v2.
>
> The classic **PyTorchJob v1** manifest ([`pointworld-pretrain.yaml`](./pointworld-pretrain.yaml),
> `kubeflow.org/v1`) is provided for clusters that still run the Training Operator.
> It now carries the `elasticPolicy` (`rdzvBackend: c10d`) required for correct
> multi-node `torchrun` rendezvous, but it has **not been validated multi-node on a
> live cluster** — verify (e.g. `env | grep -E 'PET_|WORLD_SIZE'` shows
> `WORLD_SIZE` = nodes × GPUs, and NCCL init reports all ranks) before relying on it
> at scale.
