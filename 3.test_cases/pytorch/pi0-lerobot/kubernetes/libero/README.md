<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Fine-tune π0 on LIBERO on Amazon EKS

End-to-end walkthrough for fine-tuning π0 on the LIBERO-10 dataset using
SageMaker HyperPod-on-EKS.

## Prerequisites

- HyperPod EKS cluster with a p5.48xlarge (8× H100) node
- Kubeflow Training Operator installed
- FSx for Lustre PVC (`fsx-claim`) bound and mounted
- Container image pushed to ECR (see parent README)
- `pi0-lerobot-secrets` Kubernetes Secret with HF_TOKEN

## 1. Submit the Training Job

```bash
kubectl apply -f libero-finetune.yaml
kubectl logs -f pi0-lerobot-libero-finetune-worker-0
```

Training downloads LIBERO-10, patches LeRobot, and runs FSDP on 8 GPUs
for 20K steps (~6.5 hours).

## 2. Submit the Evaluation Job

After training completes:

```bash
kubectl delete pytorchjob pi0-lerobot-libero-finetune
kubectl apply -f libero-eval.yaml
kubectl logs -f $(kubectl get pods -l app=pi0-lerobot-libero-eval --sort-by=.metadata.creationTimestamp -o name | tail -1)
```

## 3. Clean up

```bash
kubectl delete job pi0-lerobot-libero-eval
```

## Results (p5.48xlarge — 8× H100)

| Metric | Base π0 | Fine-Tuned | Improvement |
|--------|---------|------------|-------------|
| Avg MSE | 7.214e-01 | 5.003e-02 | **93.1% reduction** |
| Avg MAE | 6.302e-01 | 7.505e-02 | **88.1% reduction** |
| Time/chunk (10 steps) | 361 ms | 383 ms | — |
| Time/chunk (1 step) | — | 197 ms | — |

### ODE Step Sweep

| Steps | MSE | MAE | Time/chunk |
|-------|-----|-----|------------|
| 1 | 3.235e-02 | 7.735e-02 | 197 ms |
| 3 | 3.939e-02 | 6.875e-02 | 239 ms |
| 5 | 4.133e-02 | 6.902e-02 | 281 ms |
| 10 | 5.003e-02 | 7.505e-02 | 383 ms |
