<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Kimi2.6 — 1P1D on H200 (EKS / HyperPod)

Node-level prefill/decode disaggregation with SGLang on **two `ml.p5en.48xlarge`
(8×H200) nodes** — one prefill StatefulSet, one decode StatefulSet, KV cache
transferred over **NIXL (LIBFABRIC over EFA)**, fronted by the SGLang router.
Optional DCGM + Prometheus-agent → Amazon Managed Prometheus monitoring is
included.

## Indicative results

> **Not yet measured.** The numbers below are placeholders. Run a smoke
> completion plus a `sglang.bench_serving` sweep on the deployed topology and
> fill them in before quoting anything downstream.

| Config | Nodes | GPUs | Burst RPS | Burst tok/s | P50 TTFT | P50 TPOT | tok/s/GPU |
| ------ | :---: | :--: | --------: | ----------: | -------: | -------: | --------: |
| 1P+1D  |   2   |  16  |       TBD |         TBD |      TBD |      TBD |       TBD |

Workload: TBD (`sglang.bench_serving`).

## Prerequisites

### Cluster

- Amazon EKS or SageMaker HyperPod EKS, version **1.33+** (1.33 added native
  EFA traffic in the default EKS security group)
- 2 × `ml.p5en.48xlarge` GPU nodes (8×H200 141GB, 16×EFA), one for prefill and
  one for decode
- NVIDIA device plugin and EFA device plugin installed
- Local NVMe under `/opt/dlami/nvme` on each node for the model cache
  (~555 GB of Kimi-K2.5 weights are pre-staged there)

### Software

| Component | Version |
| --- | --- |
| Kubernetes | 1.33+ (EKS / HyperPod EKS tested) |
| kubectl | 1.33+ |
| Docker | 24.0+ |
| AWS CLI v2 | latest |
| SGLang | `v0.5.12.post1` (cu130) |
| NIXL | 1.1.0 |
| EFA installer | 1.47.0 |
| NCCL | 2.28.9 (cu13.0) |
| CUDA | 13.0 |

> **Do not bump to the SGLang nightly (`dev-cu13`).** That build ships
> **NIXL 1.2.0**, whose `LIBFABRIC` GPU-HMEM path breaks prefill→decode KV-cache
> transfer over EFA — transfers are pathologically slow (~256s for 4 tokens vs.
> ~11s for a 32-token completion on 1.1.0) and requests exceed the 300s timeout
> with `Decode transfer failed … timed out … in KVPoll.WaitingForInput`. The
> Dockerfile is pinned to `v0.5.12.post1-cu130` (**NIXL 1.1.0**) specifically to
> avoid this. Verify `python3 -c "import nixl; print(nixl.__version__)"` reports
> `1.1.0` in the container before deploying.

### Accounts and tokens

- Container registry (e.g. ECR) writable from your build host
- `kubectl` configured for your cluster
- Around ~555 GB of fast local storage per node for the weights cache

## Quick start

```bash
cd examples/inference/sglang/kimi2.6-h200-1p1d

# 1. build + push the image — prints the ECR image URI
./build-image.sh

# 2. pre-stage the weights to every matching node's NVMe (wait for
#    "Download complete!" in each downloader pod's logs)
../download-model.sh moonshotai/Kimi-K2.5 ml.p5en.48xlarge

# 3. set <YOUR_ECR_IMAGE> in kimi-pd-deploy.yaml to the URI from step 1,
#    then deploy prefill + decode StatefulSets and the router
kubectl apply -f kimi-pd-deploy.yaml

# 4. (optional) GPU metrics + remote-write to Amazon Managed Prometheus —
#    fill in the AMP/IAM placeholders in ../prometheus-agent-amp.yaml first
kubectl apply -f ../dcgm-exporter-daemonset.yaml
kubectl apply -f ../prometheus-agent-amp.yaml
```

Wait until `prefill-0`, `decode-0`, and the router are all `1/1 Ready` (engine
startup ≈ 6–8 min — weights load from local NVMe in ~30s, the rest is CUDA-graph
capture for the 1T MoE). The router exposes an OpenAI-compatible endpoint on
`sglang-router:30000` (`ClusterIP`) — port-forward it to test:

```bash
kubectl port-forward svc/sglang-router 30000:30000
curl http://localhost:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "/nvme/moonshotai-Kimi-K2.5", "prompt": "The capital of France is", "max_tokens": 32}'
```

## Files

| File | Purpose |
|---|---|
| [`Dockerfile`](./Dockerfile) | `lmsysorg/sglang:v0.5.12.post1-cu130` + AWS EFA installer |
| [`build-image.sh`](./build-image.sh) | build and push to ECR |
| [`kimi-pd-deploy.yaml`](./kimi-pd-deploy.yaml) | prefill + decode StatefulSets, router |

Model pre-staging, GPU metrics, and AMP remote-write use the shared helpers one
level up ([`../download-model.sh`](..), [`../dcgm-exporter-daemonset.yaml`](..),
[`../prometheus-agent-amp.yaml`](..)).

Per-engine SGLang flags live inline in the StatefulSet `command:` blocks:
`tp-size 8`, `ep-size 8`, `mem-fraction-static 0.92`, hierarchical cache
(prefill only), and the `kimi_k2` tool-call / reasoning parsers (decode only).
