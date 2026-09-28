<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# DeepSeek-V4.1-Flash on B200 / B300 (EKS / HyperPod)

Three ways to serve DeepSeek-V4.1-Flash with SGLang on `p6-b200.48xlarge` or `p6-b300.48xlarge`, sharing one container image and one pre-staged weight cache. Every manifest accepts both node types. Each manifest's own comments carry the per-flag rationale.

## Which variant

| | KV transport | Pod | What it is |
|---|---|---|---|
| [`manifests-single/`](./manifests-single) | none — no PD | unprivileged | One engine, `tp=2 / ep=2` on 2 GPUs. Non-PD baseline, scaled down from the cookbook's verified B300 TP4/EP4 cell, so read its actual behaviour off this run's startup logs rather than assuming the cell's numbers carry over. |
| [`manifests-pd-rdma/`](./manifests-pd-rdma) | NIXL `LIBFABRIC` — GPUDirect RDMA over EFA | unprivileged, `IPC_LOCK` | **Start here.** 1P1D of 2-GPU pods, `kubectl scale` to grow. RDMA reaches the peer without CUDA IPC or visibility into its GPUs, so each engine is an ordinary pod: independent restarts, same-node or cross-node placement. 2P2D smoke-tested on B200. |
| [`manifests-pd-nvlink/`](./manifests-pd-nvlink) | NIXL `UCX` — CUDA IPC over NVLink | **`privileged: true`** — sees all 8 GPUs | **Experimental, unverified.** 3P + 1D of 2-GPU pods pinned to one node (`podAffinity`). Cross-pod CUDA IPC costs `privileged` (defeating the device plugin's GPU isolation), `hostIPC` and `hostPID`, and buys little: at ~2–4 KB of KV per token a 4K prompt moves ~10–20 MB, under a millisecond over EFA. |

All three set `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, keeping the ~188.83 GiB Engram tables in host memory instead of ~94.42 GiB HBM per rank; without it `tp=2` runs out of HBM while constructing the model.

## Setup

- EKS or HyperPod EKS **1.33+**, NVIDIA device plugin, and for `manifests-pd-rdma` the EFA device plugin — `kubectl describe node` must show `vpc.amazonaws.com/efa: 8` (B200) or `16` (B300). On eksctl the nodegroup needs `efaEnabled: true`
- Local NVMe for the weight cache: `/opt/dlami/nvme` on HyperPod, `/mnt/k8s-disks/0` on self-managed EKS

```bash
cd examples/inference/sglang/dsv41flash-blackwell

# build + push the EFA-enabled V4.1 image (ECR URI on the last stdout line)
./build-image.sh

# pre-stage the weights onto every matching node's NVMe, then drop the daemonset
../download-model.sh deepseek-ai/DeepSeek-V4.1-Flash ml.p6-b200.48xlarge
kubectl delete daemonset model-downloader
```

V4.1 support is in no tagged SGLang release (checked through v0.5.19), only in the mutable `dev-dsv41` tag — pin `BASE_IMAGE=lmsysorg/sglang@sha256:<digest>` to what `build-image.sh` prints for any build you keep. That image also lacks the EFA layer, hence this recipe's own [`Dockerfile`](./Dockerfile) rather than [`../Dockerfile.efa`](../Dockerfile.efa).

## Deploy

All three directories take the same two template values and nothing else — everything tunable is a literal in the manifests, with comments where a value is worth changing.

```bash
export SGLANG_IMAGE='<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>'
export NVME_HOST_PATH='/opt/dlami/nvme'   # HyperPod; /mnt/k8s-disks/0 on self-managed EKS
```

Each `envsubst` call below names those two variables explicitly rather than substituting everything, which is what keeps the Kubernetes runtime values `$(POD_NAME)` / `$(POD_NAMESPACE)` and the NVLink variant's startup script untouched.

All three also answer the same call — `:30000` on the single engine, `:30080` on the PD router. Thinking is off by default; `reasoning_effort` (`low`/`high`/`xhigh`/`max`) turns it on and `--reasoning-parser auto` splits it into `reasoning_content`.

```bash
curl http://localhost:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "deepseek-ai/DeepSeek-V4.1-Flash",
       "messages": [{"role": "user", "content": "What is 15% of 240?"}],
       "reasoning_effort": "high", "max_tokens": 256}'
```

### `manifests-single` — one engine, no PD

`tp=2 / ep=2` on two GPUs, the non-PD baseline. It is scaled down from the cookbook's verified B300 TP4/EP4 cell, so read its actual behaviour off this run's startup logs instead of assuming the cell's numbers carry over. Its label (`app=dsv41flash-single`) is invisible to the PD router, so it can coexist with a PD deployment; the manifest comments show the swap to the cookbook's low-latency DSPARK cell, which PD cannot use.

```bash
cat manifests-single/*.yaml | envsubst '${SGLANG_IMAGE} ${NVME_HOST_PATH}' | kubectl apply -f -
kubectl rollout status deploy/dsv41flash-single
kubectl port-forward svc/dsv41flash-single 30000:30000
```

### `manifests-pd-rdma` — PD over EFA (default)

Prefill (`deploy/dsv41flash-prefill`, bootstrap 8998), decode (`deploy/dsv41flash-decode`), router (`deploy/dsv41flash-router`, `:30080`). All engines listen on 30000: each has its own pod IP and the gateway dials `<podIP>:30000` per worker, which is also why `hostNetwork` is not used.

```bash
cat manifests-pd-rdma/*.yaml | envsubst '${SGLANG_IMAGE} ${NVME_HOST_PATH}' | kubectl apply -f -

kubectl rollout status deploy/dsv41flash-prefill deploy/dsv41flash-decode deploy/dsv41flash-router
kubectl port-forward svc/dsv41flash-router 30080:30080
curl -s http://localhost:30080/workers | python3 -m json.tool   # expect 1 prefill + 1 decode
```

Scale with `kubectl scale deploy/dsv41flash-prefill --replicas=N`; a pod joins `/workers` once its `/health` readiness probe passes, and extra replicas stay `Pending` until a node has GPU/EFA capacity. At 2 GPUs + 2 EFA per pod, four engine pods fill one node — 2P2D is the B200 offload profile. With a single decode replica any decode rollout is a short outage, so scale decode to 2 first if that matters. Tear down with the same `envsubst … | kubectl delete -f -`.

### `manifests-pd-nvlink` — PD over NVLink (experimental)

Unverified, and it bypasses GPU isolation. Apply only with the RDMA variant torn down, then confirm the fast path actually engaged: `nvidia-smi -L` inside an engine must list 8 GPUs, and the logs must show UCX selecting `cuda_ipc` (add `UCX_LOG_LEVEL=info` if needed). If cross-container CUDA IPC is refused, UCX silently falls back to TCP and transfers become very slow — compare a 32-token completion against the RDMA variant before concluding anything.

```bash
cat manifests-pd-rdma/*.yaml   | envsubst '${SGLANG_IMAGE} ${NVME_HOST_PATH}' | kubectl delete -f -  # if running
cat manifests-pd-nvlink/*.yaml | envsubst '${SGLANG_IMAGE} ${NVME_HOST_PATH}' | kubectl apply -f -
kubectl get pod -l app=dsv41flash -o wide      # all on ONE node
```

## Notes

- **Slow RDMA transfers.** A 32-token completion taking tens of seconds with `Decode transfer failed … timed out` in the decode log is the slow NIXL 1.2.0 `LIBFABRIC` GPU-HMEM path documented in [`../kimi2.6-h200-1p1d/README.md`](../kimi2.6-h200-1p1d/README.md): pin a known-good `nixl` wheel in the Dockerfile's commented slot and rebuild. `/dev/gdrdrv` is deliberately not mounted (LIBFABRIC needs no GDRCopy, and the mount fails on nodes without the module); add it back as in that recipe if NIXL logs ask for it.
- Every engine pod exposes `:30000/metrics` with the `sglang-metrics=true` label and `prometheus.io/*` annotations (see the [SGLang README](../README.md#metrics)).
