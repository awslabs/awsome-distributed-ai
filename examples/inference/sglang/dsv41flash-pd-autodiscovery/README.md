<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# DeepSeek-V4.1-Flash PD with router auto-discovery (B200 / B300, EKS / HyperPod)

Prefill/decode disaggregation for DeepSeek-V4.1-Flash on `p6-b200.48xlarge` or `p6-b300.48xlarge`: one engine per pod (`tp=2 / ep=2`, 2 GPUs), prefill and decode scaled independently, and an SGLang router that derives its worker list from the Kubernetes API as pods become Ready — nothing to reconfigure and no router restart. The two variants differ in how the KV cache crosses from prefill to decode. Every manifest accepts both node types; per-flag rationale lives in the manifest comments.

## Which variant

| | KV transport | Pod | What it is |
|---|---|---|---|
| [`manifests-pd-rdma/`](./manifests-pd-rdma) | NIXL `LIBFABRIC` — GPUDirect RDMA over EFA | unprivileged, `IPC_LOCK`, per-pod GPU isolation | RDMA reaches the peer without CUDA IPC or any visibility into its GPUs, so each engine is an ordinary pod that the device plugin hands exactly its own 2 GPUs + 2 EFA interfaces: plain `sglang serve` as the entrypoint, independent restarts, same-node or cross-node placement. |
| [`manifests-pd-nvlink/`](./manifests-pd-nvlink) | NIXL `UCX` — CUDA IPC over NVLink | unprivileged, but `hostIPC` + `hostPID` and **all 8 GPUs visible** | Cross-pod CUDA IPC needs each engine to open the peer's GPU memory, which costs per-pod GPU isolation (treat the node as dedicated) and forces the engines to claim GPU indices themselves under an on-node lock, because the device-plugin allocation is unreadable once all devices are exposed. |

Both set `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, keeping the ~188.83 GiB Engram tables in host memory instead of ~94.42 GiB HBM per rank; without it `tp=2` runs out of HBM while constructing the model.

Only one variant can sit behind the router at a time: SGLang creates one NIXL backend per engine, and UCX and LIBFABRIC workers cannot exchange KV.

## The KV path makes no measurable difference

Three configurations, 1P1D, identical in every respect but the KV path — `sglang.bench_serving`, `random` dataset, in 1024 / out 256, concurrency swept to saturation. The spread is ~1.5%, within run-to-run noise.

| KV path | tok/s @ c=512 | req/s @ c=512 | median TTFT @ c=1 |
|---|---|---|---|
| NVLink, same node (`cuda_ipc`) | 34525 | 53.72 | 178 ms |
| EFA, same node (device loopback) | 34328 | 53.41 | 179 ms |
| EFA, across two nodes (NIC → switch → NIC) | 34183 | 53.16 | 177 ms |

## Setup

- EKS or HyperPod EKS **1.33+** with the NVIDIA device plugin; `manifests-pd-rdma` also needs the EFA device plugin — `kubectl describe node` must show `vpc.amazonaws.com/efa: 8` (B200) or `16` (B300). On eksctl the nodegroup needs `efaEnabled: true`
- Local NVMe with room for 510 GB of weights: `/opt/dlami/nvme` on HyperPod. On self-managed EKS `/mnt/k8s-disks/0` is a plain directory on the root EBS volume until the instance store is assembled — run the AMI's `/usr/bin/setup-local-disks raid0 --no-bind-mounts` on the node (it RAID0s the local NVMe devices and mounts them there), and point `../download-model-daemonset.yaml`'s hostPath at the same path

```bash
cd examples/inference/sglang/dsv41flash-pd-autodiscovery

# build + push the EFA-enabled V4.1 image from the shared ../Dockerfile.efa
# (ECR URI on the last stdout line).
BASE_IMAGE=lmsysorg/sglang:v0.5.21 EFA_VERSION=1.50.0 \
  ALGORITHM_NAME=sgl-dsv41-efa ../build-image.sh

# pre-stage the weights onto every matching node's NVMe
../download-model.sh deepseek-ai/DeepSeek-V4.1-Flash p6-b200.48xlarge
```

`BASE_IMAGE` is `lmsysorg/sglang:v0.5.21`, the first SGLang release with DeepSeek-V4.1 support. The results in this README were measured before that release, on the `dev-dsv41` preview build (`lmsysorg/sglang@sha256:e56358a68b06427362283c8c8a9d7d706448082ae53098ea1131b5d51aa1fd62`, SGLang commit [`e087e662`](https://github.com/sgl-project/sglang/commit/e087e662ba1ac4ef7747537e2a9141085efd4561), NIXL 1.4.1); pass that digest as `BASE_IMAGE` to reproduce them exactly. They have not been re-run on v0.5.21. `build-image.sh` prints the base digest it pulled and the `nixl`, `nixl-cu13` and `sglang` versions baked in. `nixl-cu13` carries the CUDA 13 backend, and it is the version to check against the LIBFABRIC GPU-HMEM slowdown described in [`../kimi2.6-h200-1p1d/README.md`](../kimi2.6-h200-1p1d/README.md).

TODO: re-run the benchmark on v0.5.21. When the AWS Deep Learning Containers image ([`public.ecr.aws/deep-learning-containers/sglang`](https://gallery.ecr.aws/deep-learning-containers/sglang), which already adds EFA to upstream SGLang) moves to 0.5.21, it may remove the need for this build; its newest tags are on SGLang 0.5.20, which predates V4.1 support.

## Deploy

Both directories take the same two template values and nothing else — everything tunable is a literal in the manifests, with comments where a value is worth changing.

```bash
export SGLANG_IMAGE='<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>'
export NVME_HOST_PATH='/opt/dlami/nvme'   # HyperPod; /mnt/k8s-disks/0 on self-managed EKS
```

### `manifests-pd-rdma` — PD over EFA (default)

Prefill (`deploy/dsv41flash-prefill`, bootstrap 8998), decode (`deploy/dsv41flash-decode`), router (`deploy/dsv41flash-router`, `:30080`). All engines listen on 30000: each has its own pod IP and the router dials `<podIP>:30000` per worker, which is also why `hostNetwork` is not used.

```bash
cat manifests-pd-rdma/*.yaml | envsubst '${SGLANG_IMAGE} ${NVME_HOST_PATH}' | kubectl apply -f -

kubectl rollout status deploy/dsv41flash-prefill deploy/dsv41flash-decode deploy/dsv41flash-router
kubectl port-forward svc/dsv41flash-router 30080:30080
curl -s http://localhost:30080/workers | python3 -m json.tool   # expect 1 prefill + 1 decode
```

### `manifests-pd-nvlink` — PD over NVLink

```bash
# Same Deployment names as the RDMA variant — tear that down first or this rolls it onto UCX.
cat manifests-pd-rdma/*.yaml   | envsubst '${SGLANG_IMAGE} ${NVME_HOST_PATH}' | kubectl delete -f -
cat manifests-pd-nvlink/*.yaml | envsubst '${SGLANG_IMAGE} ${NVME_HOST_PATH}' | kubectl apply -f -
kubectl get pod -l app=dsv41flash -o wide      # all on ONE node
kubectl logs deploy/dsv41flash-prefill | grep claimed   # e.g. "claimed physical GPUs 0..1"
```

Each engine claims a free pair of physical GPUs under a lock in `${NVME_HOST_PATH}/gpu-leases`, logging the pair it took; a `preStop` hook releases the lease on graceful termination and a stale lease left by `SIGKILL` is reclaimed by the next claimant. Verify the claims are disjoint with `nvidia-smi` on the node before reading any numbers off the deployment.

## Scaling and auto-discovery

The router runs `--service-discovery` with `--prefill-selector` / `--decode-selector`, so scaling is the whole operation:

```bash
kubectl scale deploy/dsv41flash-prefill --replicas=2
kubectl scale deploy/dsv41flash-decode  --replicas=2
curl -s http://localhost:30080/workers | python3 -m json.tool
```

A pod joins `/workers` once its `/health` readiness probe passes and is dropped when it is deleted, in both directions for both roles, without restarting the router.

## Notes

- **Verified on** `p6-b200.48xlarge` nodes (8× B200, 8× EFA each) on EKS 1.33: both variants 1P1D Ready with correct completions, RDMA PD benchmarked both same-node and with prefill and decode on separate nodes, and independent scaling to 2P1D and 1P2D picked up by router auto-discovery without a restart.
