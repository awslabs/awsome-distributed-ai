# EP-Backend Comparison Results — P5 (H100)

Same harness and matched config as [`RESULTS.md`](RESULTS.md) (B300), run on `p5.48xlarge`
(H100). Use this to compare the three dispatchers **across GPU generations**.

## Environment

| Field | Value |
|---|---|
| Date | 2026-06-22 |
| Hardware | `p5.48xlarge` on EKS (H100, 8 GPU + **32 EFA** / node) |
| EP config | num-tokens=4096 (LL: 128), hidden=7168, num-topk=8, num-experts=256, bf16 (identical to B300) |
| NVSHMEM image | `deepep:efa1.48.0-nvshmem3.7.0-deepep567632d-cuda13` (sm_90) |
| UCCL image | UCCL `0dc87eb`, CUDA 13 (Hopper sm_90 path of the committed `uccl-ep.Dockerfile`) |
| NCCL image | DeepEP image (`/opt/nccl-tests/build/alltoall_perf`) |

> **8-node run pending.** Of the 8 freshly-deployed P5 nodes, 7 came up healthy but 1 had its
> NVSwitch **Fabric Manager stuck in `In Progress`** (every CUDA op on it fails with
> `error 802: system not yet initialized`, including a trivial 1-GPU test). That is a node
> bring-up issue, not a benchmark one — it needs the node recycled. The **4-node (32-rank)**
> matrix below was therefore run on the 7 healthy nodes (pods pinned off the stuck node via
> nodeAffinity). The 8-node (64-rank) run will be added once all 8 nodes are fabric-healthy.

## 4 nodes — 32 ranks

| Backend | Mode | Dispatch (GB/s) | Combine (GB/s) |
|---|---|---:|---:|
| NVSHMEM (DeepEP) | internode (RDMA) | 43.7 | 46.1 |
| NVSHMEM (DeepEP) | low-latency | 9.6 | 20.7 |
| UCCL (UCCL-EP) | internode (RDMA) | 38.2 | 30.2 |
| UCCL (UCCL-EP) | low-latency | 5.0 | 5.0 |

| Reference (NCCL all-to-all) | Metric | GB/s |
|---|---|---:|
| busbw at EP payload (~64 MiB) | matched-size | 45.1 |
| busbw peak (asymptotic) | peak | 53.9 |

## Observations (and the B300 contrast)

- **The winner flips by GPU generation.** On **B300** (see `RESULTS.md`) UCCL matches or beats
  NVSHMEM at 4 nodes and clearly wins at 8. On **P5/H100 the order reverses**: NVSHMEM leads on
  *both* legs of internode (44/46 vs UCCL 38/30 GB/s RDMA) and on low-latency (9.6/20.7 vs UCCL
  5.0/5.0). UCCL-EP's kernels lean on SM90+ features tuned for Blackwell; on H100 they trail
  NVSHMEM here. Pick the dispatcher per target GPU, not globally.
- **Absolute bandwidth is ~half of B300.** P5 4-node internode tops out around 44–46 GB/s (RDMA)
  and the NCCL transport reference around 45 (matched) / 54 (peak), versus ~94 / 117 on B300 — a
  combination of EFA throughput and NVLink-generation differences between the platforms.
- **UCCL low-latency completes here (FP8 check passes at 32 ranks)** — at ~5 GB/s, well below
  NVSHMEM. (On B300 the FP8 low-latency check aborts at 64 ranks; see `RESULTS.md`.)

## Reproduce

Identical to [`README.md`](README.md), with `INSTANCE_TYPE=p5.48xlarge` and `EFA_PER_NODE=32`
(p5.48xlarge exposes 32 EFA NICs vs 16 on p6-b300).
