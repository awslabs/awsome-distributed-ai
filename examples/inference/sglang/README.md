<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# SGLang test cases

[SGLang](https://github.com/sgl-project/sglang) deployments on AWS EKS / SageMaker HyperPod. Each sub-directory is a self-contained sample — apply its manifest with `kubectl`.

| Test case | Hardware | Topology | Engine image |
| --- | --- | --- | --- |
| [`qwen3.5-27b-b300-intra-pd`](./qwen3.5-27b-b300-intra-pd) | 1× B300 (8 GPU) | Intra-node PD — 6 prefill + 2 decode in one pod, NIXL, SGLang router sidecar | `lmsysorg/sglang:v0.5.12.post1-cu130`, no inter-node RDMA support |
| [`kimi2.6-h200-1p1d`](./kimi2.6-h200-1p1d) | 2× H200 nodes | Node-level 1P1D — prefill + decode StatefulSets, NIXL over EFA | Custom ECR build from [`Dockerfile.efa`](./Dockerfile.efa) (lmsysorg/sglang:v0.5.12.post1-cu130 + EFA layer), inter-node RDMA enabled |
| [`dsv4pro-b300-single-node`](./dsv4pro-b300-single-node) | 1× B300 (8 GPU) | Unified (non-PD) baseline | `lmsysorg/sglang:v0.5.12.post1-cu130`, no inter-node RDMA support |
| [`dsv4flash-b300-intra-3p1d`](./dsv4flash-b300-intra-3p1d) | 1× B300 (8 GPU) | Intra-node PD — 3 prefill + 1 decode (tp=2 each) in one pod, NIXL, SGLang router sidecar | `lmsysorg/sglang:v0.5.12.post1-cu130`, no inter-node RDMA support |
| [`glm5.2-b300-tp2-dp4`](./glm5.2-b300-tp2-dp4) | 1× B300 (8 GPU) | 4× independent tp=2 engines behind an SGLang router (cache-aware LB, cluster-level dp=4) | `lmsysorg/sglang@sha256:bafcd0…` (the `dev-glm52-nvfp4` tag pinned by digest — GLM-5.2 NVFP4 support not yet in a tagged release) |

> The intra-node PD samples deliberately run several engine processes in one pod — not the usual one-process-per-pod shape. Intra-node KV transfer rides NVLink via CUDA IPC, which requires all engines to share an IPC namespace and see each other's GPUs — impossible across separate pods or containers, so the engines share one container (each sample's README explains the full rationale).

> All samples except GLM-5.2 serve on the same upstream `lmsysorg/sglang:v0.5.12.post1-cu130` image (Kimi adds only an EFA layer on top
for inter-node RDMA); the GLM-5.2 sample uses the `dev-glm52-nvfp4` image (pinned by digest in its manifest, since `dev-*` tags are mutable) until NVFP4 support for that model lands in a tagged release.

## Shared helpers

Reusable across all the samples above:

### EFA-enabled image (multi-node only)

Multi-node samples (e.g. `kimi2.6-h200-1p1d`) move the KV cache between nodes
over EFA, which the stock `lmsysorg/sglang` image can't do on its own.
[`Dockerfile.efa`](./Dockerfile.efa) layers the AWS EFA installer onto the same
upstream image; [`build-image.sh`](./build-image.sh) builds it and pushes it to
ECR, printing the image URI on the last line:

```bash
./build-image.sh   # -> <account>.dkr.ecr.<region>.amazonaws.com/sgl-dev-cu13:<tag>
```

Set that URI as `<YOUR_ECR_IMAGE>` in the sample's manifest. Single-node samples run the upstream image directly and don't need this build.

### Pre-stage model weights

Download a Hugging Face repo to every matching node's local NVMe so the serving pods read weights from fast local disk instead of pulling them at
startup. [`download-model.sh`](./download-model.sh) renders [`download-model-daemonset.yaml`](./download-model-daemonset.yaml) and applies it. The
weights are staged in **HF cache layout** under `<nvme>/huggingface` — the dir every serving manifest mounts at `/root/.cache/huggingface`, with the
engines loading by repo id, so the staged snapshot is found as a cache hit:

```bash
./download-model.sh moonshotai/Kimi-K2.6        ml.p5en.48xlarge
./download-model.sh deepseek-ai/DeepSeek-V4-Pro ml.p6-b300.48xlarge
# watch: kubectl logs -f -l app=model-downloader   (each node prints "Download complete!")
# then:  kubectl delete daemonset model-downloader
```

Like the serving manifests, the daemonset's NVMe `hostPath` defaults to HyperPod's `/opt/dlami/nvme` — on self-managed EKS change it to `/mnt/k8s-disks/0` so it stages to the disk the serving pods actually mount.

### Metrics

Every serving pod already exposes SGLang metrics on `:30000/metrics` (the engines start with `--enable-metrics`) and carries the `sglang-metrics=true` label plus the `prometheus.io/{scrape,port,path}` scrape annotations. Point any Prometheus-compatible scraper at those pods — e.g. an in-cluster Prometheus **agent** remote-writing to **Amazon Managed Prometheus (AMP)** with **Amazon Managed Grafana** reading from AMP, or a self-managed Prometheus + Grafana stack. Wiring up the scrape backend is left to your cluster's existing observability setup; the manifests here only produce the metrics.
