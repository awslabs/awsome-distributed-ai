<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Fine-tune π0 on DROID on Amazon EKS

End-to-end walkthrough for fine-tuning π0 on the DROID-100 dataset using
SageMaker HyperPod-on-EKS.

## Prerequisites

- HyperPod EKS cluster with a p5.48xlarge (8× H100) node
- Kubeflow Training Operator installed
- FSx for Lustre PVC (`fsx-claim`) bound and mounted
- Container image pushed to ECR (see parent README)
- `pi0-lerobot-secrets` Kubernetes Secret with HF_TOKEN

## 1. Submit the Training Job

The training YAML downloads DROID-100, installs patches, and launches
FSDP training on 8 GPUs for 20K steps (~6.5 hours):

```bash
kubectl apply -f droid-finetune.yaml
kubectl logs -f pi0-lerobot-droid-finetune-worker-0
```

## 2. Submit the Evaluation Job

After training completes:

```bash
kubectl delete pytorchjob pi0-lerobot-droid-finetune
kubectl apply -f droid-eval.yaml
kubectl logs -f $(kubectl get pods -l app=pi0-lerobot-droid-eval --sort-by=.metadata.creationTimestamp -o name | tail -1)
```

## 3. Clean up

```bash
kubectl delete job pi0-lerobot-droid-eval
```

## Results (p5.48xlarge — 8× H100)

| Metric | Base π0 | Fine-Tuned | Improvement |
|--------|---------|------------|-------------|
| Avg MSE | 3.897e-01 | 1.353e-02 | **96.5% reduction** |
| Avg MAE | 4.617e-01 | 5.614e-02 | **87.9% reduction** |
| Time/chunk (10 steps) | 366 ms | 385 ms | — |
| Time/chunk (1 step) | — | 199 ms | — |

### ODE Step Sweep

| Steps | MSE | MAE | Time/chunk |
|-------|-----|-----|------------|
| 1 | 8.352e-03 | 4.941e-02 | 199 ms |
| 3 | 1.072e-02 | 5.152e-02 | 241 ms |
| 5 | 1.063e-02 | 5.060e-02 | 284 ms |
| 10 | 1.353e-02 | 5.614e-02 | 385 ms |
