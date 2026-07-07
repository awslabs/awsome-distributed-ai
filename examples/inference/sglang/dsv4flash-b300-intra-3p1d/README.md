<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# DeepSeek-V4-Flash — Intra-node PD on B300 (EKS / HyperPod)

Prefill/decode disaggregation **within a single 8-GPU B300 node** using SGLang.
All engines run in one pod on one node: three prefill instances and one decode
instance, each `tp=2` spanning **two GPUs**, split via `--base-gpu-id`:

| Role | GPUs | HTTP port | Bootstrap port |
|---|---|---|---|
| prefill 0 | 0, 1 | 30000 | 9000 |
| prefill 1 | 2, 3 | 30100 | 9100 |
| prefill 2 | 4, 5 | 30200 | 9200 |
| decode    | 6, 7 | 30300 | 9300 |

HTTP ports are spaced 100 apart on purpose: with `--enable-dp-attention`,
SGLang derives a **block** of internal TCP ports at `--port + 233` (dist-init,
detokenizer, rpc, metrics, scheduler-input, one per dp rank). Adjacent base
ports would overlap those blocks and fail with *"Address already in use"*.

KV cache moves prefill → decode over **NIXL**, staying intra-node. A router
sidecar shares the pod network namespace and reaches every engine on
`127.0.0.1`.

## Deploy

```bash
kubectl apply -f dsv4flash-pd-deploy.yaml
kubectl rollout status deploy/dsv4flash-intra-pd
```

Targets a `p6-b300.48xlarge` node (`nodeAffinity` in the manifest matches both
the bare EKS `p6-b300.48xlarge` and the HyperPod `ml.p6-b300.48xlarge`
instance-type label).

The **one** thing you must set per environment is the NVMe `hostPath` near the
bottom of the manifest — the local disk is mounted at a different path on each
AMI (`/opt/dlami/nvme/...` on HyperPod, `/mnt/k8s-disks/0/...` on self-managed
EKS). The default is HyperPod's.

The router exposes an OpenAI-compatible endpoint on `dsv4flash-router:30080`
(`ClusterIP`) — port-forward to call it:

```bash
kubectl port-forward svc/dsv4flash-router 30080:30080
curl http://localhost:30080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "deepseek-ai/DeepSeek-V4-Flash", "prompt": "The capital of France is", "max_tokens": 32}'
```

Tear down with `kubectl delete -f dsv4flash-pd-deploy.yaml`.