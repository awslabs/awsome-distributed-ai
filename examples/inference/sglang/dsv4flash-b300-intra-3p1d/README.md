<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# DeepSeek-V4-Flash — Intra-node PD on B300 (EKS / HyperPod)

Prefill/decode disaggregation **within a single 8-GPU B300 node** using SGLang. All engines run in one pod on one node: three prefill instances and one decode instance, each `tp=2` spanning **two GPUs**, split via `--base-gpu-id`:

| Role | GPUs | HTTP port | Bootstrap port |
|---|---|---|---|
| prefill 0 | 0, 1 | 30000 | 9000 |
| prefill 1 | 2, 3 | 30100 | 9100 |
| prefill 2 | 4, 5 | 30200 | 9200 |
| decode    | 6, 7 | 30300 | 9300 |

HTTP ports are spaced 100 apart on purpose: with `--enable-dp-attention`, SGLang derives a **block** of internal TCP ports at `--port + 233` (dist-init,
detokenizer, rpc, metrics, scheduler-input, one per dp rank). Adjacent base ports would overlap those blocks and fail with *"Address already in use"*.

KV cache moves prefill → decode over **NIXL**, staying intra-node. A router sidecar shares the pod network namespace and reaches every engine on
`127.0.0.1`.

## Why one pod with four engine processes?

Running each engine as its own pod (or its own container) would be the more Kubernetes-native shape, but it breaks the one thing this topology exists for: **NVLink KV transfer**. Intra-node NIXL moves KV pages via CUDA IPC, which requires the prefill and decode processes to share an IPC namespace *and* see each other's GPUs. The device plugin hands each container a **disjoint** GPU set and pods get isolated IPC namespaces, so CUDA IPC handles cannot be opened across them (this fails even between two containers in the *same* pod, because neither can see the peer's GPUs). NIXL then falls back to TCP over the pod network — NVIDIA's [disaggregated-inference guide](https://docs.dynamo.nvidia.com/dynamo/kubernetes-deployment/operate/disagg-communication) measures that fallback at 200–500× worse TTFT.

So the rule of thumb encoded by this repo's samples:

- **PD split across pods/nodes** — correct when the pods are connected by RDMA(EFA); see [`kimi2.6-h200-1p1d`](../kimi2.6-h200-1p1d), where prefill and decode are separate StatefulSets.
- **PD within one node, no RDMA between engines** — the engines must live in one container that owns all 8 GPUs, as here. The pod is still a single
  schedulable, restartable unit: probes watch the foreground engine and the whole group restarts together (`strategy: Recreate`), which is also the
  correct failure semantics — a half-alive PD group can't serve anyway.
- **No PD at all** — if you just want N independent engines on a node, one-engine-per-pod is the native shape; see [`glm5.2-b300-tp2-dp4`](../glm5.2-b300-tp2-dp4).

## Deploy

```bash
kubectl apply -f dsv4flash-pd-deploy.yaml
kubectl rollout status deploy/dsv4flash-intra-pd
```

Targets a `p6-b300.48xlarge` node (`nodeAffinity` in the manifest matches both the bare EKS `p6-b300.48xlarge` and the HyperPod `ml.p6-b300.48xlarge` instance-type label).

The **one** thing you must set per environment is the NVMe `hostPath` near the bottom of the manifest — the local disk is mounted at a different path on each AMI (`/opt/dlami/nvme/...` on HyperPod, `/mnt/k8s-disks/0/...` on self-managed EKS). The default is HyperPod's.

The router exposes an OpenAI-compatible endpoint on `dsv4flash-router:30080` (`ClusterIP`) — port-forward to call it:

```bash
kubectl port-forward svc/dsv4flash-router 30080:30080
curl http://localhost:30080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "deepseek-ai/DeepSeek-V4-Flash", "prompt": "The capital of France is", "max_tokens": 32}'
```

Tear down with `kubectl delete -f dsv4flash-pd-deploy.yaml`.