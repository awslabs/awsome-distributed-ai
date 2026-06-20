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
| NVSHMEM image | `deepep:efa1.48.0-nvshmem3.7.0-deepep567632d` (CUDA 13) |
| UCCL image | `uccl-ep:efa1.43.2-uccl0dc87eb-sm100` (CUDA 12.8.1, UCCL `0dc87eb`, sm_100) |
| NCCL image | `nccl-tests:...` (CUDA 13.0.2) |

## Dispatch / Combine bandwidth (EP backends)

| Backend | Mode | Dispatch (GB/s) | Combine (GB/s) |
|---|---|---:|---:|
| NVSHMEM (DeepEP) | internode | _TBD_ | _TBD_ |
| NVSHMEM (DeepEP) | low-latency | _TBD_ | _TBD_ |
| UCCL (UCCL-EP) | internode | _TBD_ | _TBD_ |
| UCCL (UCCL-EP) | low-latency | _TBD_ | _TBD_ |

## Transport reference (baseline)

| Reference | Metric | GB/s |
|---|---|---:|
| NCCL all-to-all (transport ceiling) | peak busbw | _TBD_ |

## Notes

- NCCL is the transport ceiling, not a dispatch/combine equal — EP numbers are expected to sit
  below it.
- _Record any config deltas, failures, or retuning here._
