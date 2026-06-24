<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# PointWorld — Kubernetes manifests

Manifests for pre-training and evaluating
[PointWorld](https://github.com/NVlabs/PointWorld) on Amazon EKS / SageMaker
HyperPod EKS.

| Manifest | Kind | Purpose |
|---|---|---|
| [`pointworld-data-prep.yaml`](./pointworld-data-prep.yaml) | `Job` | Stage DINOv3 weights + WebDataset shards onto FSx (runs in-cluster) |
| [`pointworld-pretrain.yaml`](./pointworld-pretrain.yaml) | `PyTorchJob` | Multi-node DDP pre-training (DROID + BEHAVIOR, large PTv3) |
| [`pointworld-eval.yaml`](./pointworld-eval.yaml) | `Job` | Single-GPU evaluation of a trained or released checkpoint |
| [`trainer-v2/`](./trainer-v2/) | `TrainJob` + `ClusterTrainingRuntime` | Kubeflow Trainer v2 variants of pre-training |

> [!important] Single source of truth
> All instructions — prerequisites, building and pushing the image, staging data
> onto FSx, running pre-training and evaluation, the FSx layout, and tuning notes
> — live in the **[top-level README](../README.md)**. This page is just the
> manifest index. Use [`trainer-v2/`](./trainer-v2/) if your cluster runs Kubeflow
> Trainer v2 (`trainer.kubeflow.org`) instead of the PyTorchJob v1 operator.
