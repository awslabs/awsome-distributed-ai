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

> **Node bring-up note.** One of the 8 freshly-deployed P5 nodes initially had its NVSwitch
> **Fabric Manager stuck in `In Progress`** (every CUDA op on it failed with `error 802:
> system not yet initialized`, including a trivial 1-GPU test). The 4-node run was done on the
> 7 healthy nodes meanwhile; after that node was recycled the 8-node run completed on all 8.

## 8 nodes — 64 ranks

| Backend | Mode | Dispatch (GB/s) | Combine (GB/s) |
|---|---|---:|---:|
| NVSHMEM (DeepEP) | internode (RDMA) | 39.6 | 39.2 |
| NVSHMEM (DeepEP) | low-latency | 6.5 | 15.6 |
| UCCL (UCCL-EP) | internode (RDMA) | 29.9 | 26.7 |
| UCCL (UCCL-EP) | low-latency | 3.1¹ | 3.7¹ |

| Reference (NCCL all-to-all) | Metric | GB/s |
|---|---|---:|
| busbw at EP payload (~64 MiB) | matched-size | 40.4 |
| busbw peak (asymptotic) | peak | 51.1 |

¹ Same as B300, the standard FP8 low-latency path (`round_scale=False`) **passes** correctness at
64 ranks (max diff 1.07e-4 vs the 9e-4 FP8 tolerance — 8× margin) and gives these numbers. The
unpatched test aborts *first* on the coarser `round_scale=True` FP8 sub-case, which upstream DeepEP
exempts via `if not round_scale`; matching that gating recovers the bandwidth. The per-sub-case
errors are **identical to B300** (the reference is generated from fixed seeds), confirming this is a
quantization-recipe property, not GPU-arch. See [`RESULTS.md`](RESULTS.md) for the sub-case table.

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

- **The winner flips by GPU generation.** On **B300** (see `RESULTS.md`) UCCL matches/beats
  NVSHMEM at 4 nodes and clearly wins at 8. On **P5/H100 the order reverses at both scales**:
  NVSHMEM leads UCCL on internode (4n 44/46 vs 38/30; 8n **40/39 vs 30/27** GB/s RDMA) and on
  low-latency (8n LL 6.5/15.6 vs 3.1/3.7). UCCL-EP's kernels lean on SM90+ features tuned for
  Blackwell; on H100 they trail NVSHMEM here. **Pick the dispatcher per target GPU, not globally.**
- **Absolute bandwidth is ~half of B300.** P5 internode tops out ~40–46 GB/s (RDMA) and the NCCL
  reference ~40–45 (matched) / ~51–54 (peak), versus ~73–96 / ~103–117 on B300 — a combination of
  EFA throughput and NVLink-generation differences.
- **UCCL low-latency's 64-rank abort is a test-gate divergence, not a kernel fault.** The standard
  `round_scale=False` FP8 path passes at 64 ranks (max diff 1.07e-4, identical on P5 and B300); the
  default test aborts only on the coarser `round_scale=True` sub-case that DeepEP exempts via
  `if not round_scale`. Matching that gating recovers the LL bandwidth (P5 3.1/3.7, B300 28.4/24.8).
  See [`RESULTS.md`](RESULTS.md) for the sub-case breakdown.

## Reproduce

Identical to [`README.md`](README.md), with `INSTANCE_TYPE=p5.48xlarge` and `EFA_PER_NODE=32`
(p5.48xlarge exposes 32 EFA NICs vs 16 on p6-b300).
