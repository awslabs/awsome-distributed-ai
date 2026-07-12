<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# GLM-5.2 (NVFP4) — 4× tp=2 replicas + SGLang router on B300 (EKS / HyperPod)

The NVFP4 checkpoint fits in 2 GPUs, so instead of one big engine the node runs **four independent tp=2 engines** (4 × 2 GPUs = 8 GPUs), fronted by an **SGLang router** (`lmsysorg/sgl-model-gateway`) that load-balances across them with its cache-aware policy. Each engine runs `tp=2, dp=2,--enable-dp-attention` internally.

## Why 4× tp=2 replicas instead of one tp=2/dp=4 engine

- **Elastic scaling** — replicas are the scaling unit: `kubectl scale` (or an HPA) adds/removes engines two GPUs at a time, and spilling onto a second
  node needs no config change. One monolithic engine can only scale by redeploying.
- **Failure isolation & rolling updates** — one wedged engine restarts alone while the other 3 keep serving; a rolling update keeps 3/4 capacity online
  (`maxSurge: 0, maxUnavailable: 1`, since all 8 GPUs are taken).
- **Engine constraint** — with `--enable-dp-attention` SGLang requires `dp` to divide `tp`, so `tp=2, dp=4` is not a valid single-engine config anyway.

## Prerequisites

1. **Provision the B300 node** — on self-managed (eksctl) EKS, create the nodegroup first: see [`../dsv4pro-b300-single-node/NODEGROUP-EKS.md`](../dsv4pro-b300-single-node/NODEGROUP-EKS.md).
   On HyperPod-on-EKS, nodes are already provisioned.
2. **Pre-stage the weights** (required) — the downloader is a DaemonSet, so every matching B300 node gets the weights staged into its NVMe HF cache
   (the same dir the engine pods mount at `/root/.cache/huggingface`); replicas landing on that node then load from local disk instead of downloading:

   ```bash
   ../download-model.sh nvidia/GLM-5.2-NVFP4 ml.p6-b300.48xlarge
   ```

   Without pre-staging the engines still come up — replicas sharing a node converge on one download via HF's file locks — but each node pulls the full
   465 GB itself, so on a multi-node cluster expect one full download per node either way; pre-staging just moves it ahead of pod startup.

## Deploy

```bash
kubectl apply -f manifests/glm52-deploy.yaml
kubectl rollout status deploy/glm52-tp2      # expect 4/4 Running
kubectl rollout status deploy/glm52-router
```

The manifest runs on **both** HyperPod-on-EKS and self-managed EKS (same `nodeAffinity` + GPU-taint-toleration scheme as the other samples).

The **one** thing you must set per environment is the NVMe `hostPath` near the bottom of the manifest — the local disk is mounted at a different path on each AMI (`/opt/dlami/nvme/...` on HyperPod, `/mnt/k8s-disks/0/...` on self-managed EKS). The default is HyperPod's.

The router exposes an OpenAI-compatible endpoint on `glm52-router:30080` (`ClusterIP`) — port-forward to call it:

```bash
kubectl port-forward svc/glm52-router 30080:30080
curl http://localhost:30080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "nvidia/GLM-5.2-NVFP4", "prompt": "The capital of France is", "max_tokens": 32}'
```

Tear down with `kubectl delete -f manifests/glm52-deploy.yaml`.

## Benchmark

Benchmark through the **router** (not a single engine pod) so all 4 engines
are exercised:

```bash
kubectl port-forward svc/glm52-router 30080:30080 &
python3 -m sglang.bench_serving --backend sglang \
  --base-url http://localhost:30080 \
  --dataset-name random --num-prompts 1000 \
  --random-input 2048 --random-output 256 \
  --request-rate inf --max-concurrency 100
```
