# EP-Backend Comparison Results

> Populated by running the three configs on the cluster and collating with
> [`collect_results.py`](collect_results.py). Placeholders below are filled after the run.

## Environment

| Field | Value |
|---|---|
| Date | _TBD_ |
| Hardware | 8 × `p6-b300.48xlarge` (Blackwell B300, 8 GPU + 16 EFA / node), EKS |
| World size | 64 ranks (8 nodes × 8 GPU) |
| EP config | num-tokens=4096 (LL: 128), hidden=7168, num-topk=8, num-experts=256, bf16 |
| NVSHMEM image | `deepep:efa1.48.0-nvshmem3.7.0-deepep567632d-cuda13` (CUDA 13) |
| UCCL image | UCCL `0dc87eb`, CUDA 13, Hopper+Blackwell (committed `uccl-ep.Dockerfile`; the run used the equivalent prebuilt dsv3-uccl-nixl image) |
| NCCL image | DeepEP image (reuses `/opt/nccl-tests/build/alltoall_perf`, sm_100) |

## Dispatch / Combine bandwidth (EP backends)

| Backend | Mode | Dispatch (GB/s) | Combine (GB/s) |
|---|---|---:|---:|
| NVSHMEM (DeepEP) | internode (RDMA) | _TBD_ | _TBD_ |
| NVSHMEM (DeepEP) | low-latency | _TBD_ | _TBD_ |
| UCCL (UCCL-EP) | internode (RDMA) | _TBD_ | _TBD_ |
| UCCL (UCCL-EP) | low-latency | _TBD_ | _TBD_ |

## Transport reference (baseline)

| Reference (NCCL all-to-all) | Metric | GB/s |
|---|---|---:|
| busbw at EP payload (~56 MiB) | matched-size | _TBD_ |
| busbw peak (asymptotic) | peak | _TBD_ |

## Notes

- NCCL is the transport ceiling, not a dispatch/combine equal — EP numbers are expected to sit
  below it.
- _Record any config deltas, failures, or retuning here._
