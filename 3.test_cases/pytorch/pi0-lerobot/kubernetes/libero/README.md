<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Fine-tune π0 on LIBERO on Amazon EKS

End-to-end walkthrough for fine-tuning π0 on the LIBERO-10 dataset using
SageMaker HyperPod-on-EKS.

## Prerequisites

- HyperPod EKS cluster with a p5.48xlarge (8× H100) node
- Kubeflow Training Operator installed
- FSx for Lustre PVC (`fsx-claim`) bound and mounted
- `pi0-lerobot-secrets` Kubernetes Secret with HF_TOKEN
  (token must have accepted [Gemma license](https://huggingface.co/google/gemma-2b))
- `evaluate_pi0.py` staged on FSx at `/fsx/pi0-lerobot/evaluate_pi0.py`

## 1. Submit the Training Job

Training downloads LIBERO-10 (`lerobot/libero_10`), installs LeRobot (pinned commit
`ddc2aa7`), and runs FSDP on 8 GPUs for 20K steps (~6.5 hours). Only episodes 0-303
are used for training.

```bash
kubectl apply -f libero-finetune.yaml
kubectl logs -f pi0-lerobot-libero-finetune-worker-0
```

## 2. Submit the Evaluation Job

After training completes, evaluate on held-out episodes 304-378:

```bash
kubectl delete pytorchjob pi0-lerobot-libero-finetune
kubectl apply -f libero-eval.yaml
kubectl logs -f $(kubectl get pods -l app=pi0-lerobot-libero-eval --sort-by=.metadata.creationTimestamp -o name | tail -1)
```

## 3. Clean up

```bash
kubectl delete job pi0-lerobot-libero-eval
```

## Train/Test Split

| Set | Episodes | Purpose |
|-----|----------|---------|
| Train | 0-303 (304 episodes) | Passed via `--dataset.episodes` |
| Eval (held-out) | 304-378 (75 episodes) | Passed via `--eval-episodes` to evaluate_pi0.py |

Total episodes in `lerobot/libero_10`: 379.

## Results (p5.48xlarge — 8× H100 80GB)

| Metric | Base | Fine-Tuned | Improvement |
|--------|------|-----------|-------------|
| MSE | 7.677e-01 | 8.548e-02 | **88.9%** |
| MAE | 6.565e-01 | 1.175e-01 | **82.1%** |
| Latency (1 ODE step) | — | 197ms | — |

**ODE step-count sweep (fine-tuned):**

| ODE Steps | MSE |
|-----------|-----|
| 1 | 7.768e-02 |
| 3 | 8.789e-02 |
| 5 | 9.236e-02 |
| 10 | 8.548e-02 |
