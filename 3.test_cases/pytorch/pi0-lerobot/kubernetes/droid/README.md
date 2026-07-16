<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Fine-tune π0 on DROID on Amazon EKS

End-to-end walkthrough for fine-tuning π0 on the DROID-100 dataset using
SageMaker HyperPod-on-EKS.

## Prerequisites

- HyperPod EKS cluster with a p5.48xlarge (8× H100) node
- Kubeflow Training Operator installed
- FSx for Lustre PVC (`fsx-claim`) bound and mounted
- `pi0-lerobot-secrets` Kubernetes Secret with HF_TOKEN
  (token must have accepted [Gemma license](https://huggingface.co/google/gemma-2b))
- `evaluate_pi0.py` staged on FSx at `/fsx/pi0-lerobot/evaluate_pi0.py`

## 1. Submit the Training Job

Training downloads DROID-100, installs LeRobot v0.5.0, and runs FSDP on 8 GPUs
for 20K steps (~6.5 hours). Only episodes 0-79 are used for training.

```bash
kubectl apply -f droid-finetune.yaml
kubectl logs -f pi0-lerobot-droid-finetune-worker-0
```

## 2. Submit the Evaluation Job

After training completes, evaluate on held-out episodes 80-99:

```bash
kubectl delete pytorchjob pi0-lerobot-droid-finetune
kubectl apply -f droid-eval.yaml
kubectl logs -f $(kubectl get pods -l app=pi0-lerobot-droid-eval --sort-by=.metadata.creationTimestamp -o name | tail -1)
```

## 3. Clean up

```bash
kubectl delete job pi0-lerobot-droid-eval
```

## Train/Test Split

| Set | Episodes | Purpose |
|-----|----------|---------|
| Train | 0-79 | Passed via `--dataset.episodes` |
| Eval (held-out) | 80-99 | Passed via `--eval-episodes` to evaluate_pi0.py |

> **Note:** Results pending re-run with this proper split. See top-level README.
