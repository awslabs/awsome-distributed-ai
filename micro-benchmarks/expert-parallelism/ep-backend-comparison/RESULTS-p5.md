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
| UCCL (UCCL-EP) | low-latency | n/a¹ | n/a¹ |

| Reference (NCCL all-to-all) | Metric | GB/s |
|---|---|---:|
| busbw at EP payload (~64 MiB) | matched-size | 40.4 |
| busbw peak (asymptotic) | peak | 51.1 |

¹ Same as B300: UCCL's low-latency test aborts in its FP8 correctness check at 64 ranks
(`diff ≈ 9e-4 – 1.8e-3` vs the `< 9e-4` FP8 tolerance) before the timed phase — so this is a
**rank-scale** failure in UCCL's FP8 low-latency dispatch, independent of GPU generation
(it also fails on B300 at 64 ranks and passes at 32 on both). See [`RESULTS.md`](RESULTS.md) for
the exact check.

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
  low-latency. UCCL-EP's kernels lean on SM90+ features tuned for Blackwell; on H100 they trail
  NVSHMEM here. **Pick the dispatcher per target GPU, not globally.**
- **Absolute bandwidth is ~half of B300.** P5 internode tops out ~40–46 GB/s (RDMA) and the NCCL
  reference ~40–45 (matched) / ~51–54 (peak), versus ~73–96 / ~103–117 on B300 — a combination of
  EFA throughput and NVLink-generation differences.
- **UCCL low-latency FP8 failure is rank-scale, not arch.** It passes at 32 ranks (P5: ~5 GB/s)
  and aborts at 64 ranks on **both** P5 and B300 — so it tracks rank count (combine sums more
  cross-rank FP8-dequantized contributions), not the GPU.

## Reproduce

Identical to [`README.md`](README.md), with `INSTANCE_TYPE=p5.48xlarge` and `EFA_PER_NODE=32`
(p5.48xlarge exposes 32 EFA NICs vs 16 on p6-b300).
