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

### Monitoring (Prometheus + Grafana)

The serving pods already expose SGLang metrics on `:30000/metrics` (started with
`--enable-metrics`) and carry the `sglang-metrics=true` label plus the
`prometheus.io/*` scrape annotations. The monitoring path is fully AWS-managed:
an in-cluster Prometheus **agent** remote-writes to **Amazon Managed Prometheus
(AMP)**, and **Amazon Managed Grafana** reads from AMP — there is no in-cluster
Grafana.

**1. AMP + Prometheus agent (scripted)** —
[`setup-amp-monitoring.sh`](./setup-amp-monitoring.sh) is idempotent and does the
three one-time steps in order: create (or reuse) an AMP workspace, enable the
cluster OIDC provider and create the AMP ingest IAM role bound to the
`amp-iamproxy-ingest-service-account` ServiceAccount, then render
[`prometheus-agent-amp.yaml`](./prometheus-agent-amp.yaml) with the real
workspace id / role ARN / region and apply it.

```bash
./setup-amp-monitoring.sh <CLUSTER_NAME> [REGION] [AMP_ALIAS]
# e.g. ./setup-amp-monitoring.sh eks-hypd-0512-b2ad us-west-2 sglang-kimi
# then watch the agent leave CrashLoopBackOff:
#   kubectl rollout status deployment/prometheus-agent
```

The agent scrapes every pod labeled `sglang-metrics=true` or `dcgm-metrics=true`
and remote-writes via SigV4. (Requires `awscli`, `eksctl`, `kubectl`, `envsubst`
and AWS creds with AMP + IAM permissions.)

**2. GPU metrics** — [`dcgm-exporter-daemonset.yaml`](./dcgm-exporter-daemonset.yaml)
runs a DCGM exporter DaemonSet on `:9400` (labeled `dcgm-metrics=true`, so the
agent above picks it up automatically). Apply it:

```bash
kubectl apply -f dcgm-exporter-daemonset.yaml
```

The manifest schedules onto nodes labeled `nvidia.com/gpu.present=true`. This
label is **not** present by default on SageMaker HyperPod nodes — it is the
NVIDIA GPU Operator convention, and HyperPod doesn't run the Operator. So on a
plain HyperPod cluster the DaemonSet comes up with `DESIRED 0` and never starts
a pod. Two ways to fix it:

- **Quick:** label the GPU nodes by hand —
  `kubectl label nodes <node>... nvidia.com/gpu.present=true`. Simple, but the
  label does **not** survive node replacement: if HyperPod swaps a node, the new
  one won't carry it and no DCGM pod will schedule there until you re-label.
- **Durable:** install the NVIDIA GPU Operator / device-plugin, which labels GPU
  nodes automatically (and can manage DCGM itself).

Verify the pods landed (one per GPU node) before checking Grafana:

```bash
kubectl get ds dcgm-exporter            # DESIRED should match your GPU node count
kubectl get pods -l app=dcgm-exporter -o wide
```

**3. Amazon Managed Grafana** — create an Amazon Managed Grafana workspace
(console or `aws grafana create-workspace`) with the **Amazon Managed Service for
Prometheus** data-source / IAM permission enabled. In the workspace, add a
Prometheus data source pointing at the AMP query endpoint
(`https://aps-workspaces.<region>.amazonaws.com/workspaces/<workspace-id>/`) with
**SigV4 auth** turned on, then import an SGLang or DCGM dashboard. The script
prints the workspace id and remote-write URL when it finishes.
