<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# SGLang test cases

[SGLang](https://github.com/sgl-project/sglang) deployments on AWS EKS /
SageMaker HyperPod. Each sub-directory is a self-contained sample — apply its
manifest with `kubectl`.

| Test case | Hardware | Topology |
| --- | --- | --- |
| [`qwen3.5-27b-b300-intra-pd`](./qwen3.5-27b-b300-intra-pd) | 1× B300 (8 GPU) | Intra-node PD — 6 prefill + 2 decode in one pod, NIXL, SGLang router sidecar |
| [`kimi2.6-h200-1p1d`](./kimi2.6-h200-1p1d) | 2× H200 nodes | Node-level 1P1D — prefill + decode StatefulSets, NIXL over EFA |
| [`dsv4pro-b300-single-node`](./dsv4pro-b300-single-node) | 1× B300 (8 GPU) | Unified (non-PD) baseline |

## Shared helpers

Reusable across all the samples above:

### Pre-stage model weights

Download a Hugging Face repo to every matching node's local NVMe
(`/opt/dlami/nvme`) so the serving pods read weights from fast local disk
instead of pulling them at startup. [`download-model.sh`](./download-model.sh)
renders [`download-model-daemonset.yaml`](./download-model-daemonset.yaml) and
applies it — `LOCAL_DIR_NAME` defaults to the repo id with `/` → `-`:

```bash
./download-model.sh moonshotai/Kimi-K2.5       ml.p5en.48xlarge
./download-model.sh deepseek-ai/DeepSeek-V4-Pro ml.p6-b300.48xlarge
# watch: kubectl logs -f -l app=model-downloader   (each node prints "Download complete!")
# then:  kubectl delete daemonset model-downloader
```

### GPU metrics

[`dcgm-exporter-daemonset.yaml`](./dcgm-exporter-daemonset.yaml) runs a DCGM
exporter DaemonSet on `:9400` for Prometheus. Generic — apply as-is:

```bash
kubectl apply -f dcgm-exporter-daemonset.yaml
```
