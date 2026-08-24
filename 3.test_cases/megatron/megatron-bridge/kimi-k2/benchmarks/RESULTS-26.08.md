# Kimi-K2 Megatron-Bridge EP results, NeMo 26.08

Campaign ID: `20260824T030931Z-kimi-k2-megatron-ep-b200-256g`

Qualification date: `2026-08-24 UTC`

The requested 256-GPU B300 headline is `NOT_RUN_INSUFFICIENT_CAPACITY`. The ap-south-1 cluster had 0 B300 nodes, below the 32-node requirement. A separate B200 fallback qualified Kimi-K2 training on 32 p6-b200.48xlarge instances and 256 B200 GPUs. The four-arm scored milestone is `NOT_RUN_CAPACITY_RECLAIMED`: AWS began shutting down all 36 B200 instances at `2026-08-24T10:51:03Z`, before the first scored arm started and 38 minutes before the Capacity Block's listed `2026-08-24T11:30:00Z` end.

## Topology

One Kimi-K2 training replica is 64 GPUs: tensor parallelism uses 8 ranks and pipeline parallelism uses 8 stages. The 256-GPU deployment contains 4 data-parallel replicas, each with expert parallelism across 32 ranks. Qualification used a global batch size of 256 samples, a microbatch size of 4 samples, a sequence length of 4,096 tokens, and 384 experts with top-k 8 routing choices.

## B300 headline

| Arm | Throughput, MB4, 256 B300 GPUs | Reason |
|---|---:|---|
| `nccl-alltoall` | `NOT_RUN_INSUFFICIENT_CAPACITY` | 0 of 32 required B300 nodes existed |
| `uccl` | `NOT_RUN_INSUFFICIENT_CAPACITY` | 0 of 32 required B300 nodes existed |
| `deepep-v1-nvshmem` | `NOT_RUN_INSUFFICIENT_CAPACITY` | 0 of 32 required B300 nodes existed |
| `deepep-v2-gin-gda` | `NOT_RUN_INSUFFICIENT_CAPACITY` | 0 of 32 required B300 nodes existed |

No result from another release, GPU type, or topology is copied into this table.

## B200 256-GPU scored milestone

| Arm | Planned scored shape | Outcome |
|---|---|---|
| `nccl-alltoall` | 12 iterations, 8 warmup iterations, MB4, overlap off | `NOT_RUN_CAPACITY_RECLAIMED` |
| `uccl` | 12 iterations, 8 warmup iterations, MB4, overlap off | `NOT_RUN_CAPACITY_RECLAIMED` |
| `deepep-v1-nvshmem` | 12 iterations, 8 warmup iterations, MB4, overlap off | `NOT_RUN_CAPACITY_RECLAIMED` |
| `deepep-v2-gin-gda` | 12 iterations, 8 warmup iterations, MB4, overlap off | `NOT_RUN_CAPACITY_RECLAIMED` |

There are 0 scored starts and 0 scored iterations for every arm, so this campaign makes no throughput, speedup, variability, or winner claim. The short correctness steps below are not scored performance samples.

## Final image quartet

| Arm | Immutable ap-south-1 image digest |
|---|---|
| `nccl-alltoall` | `sha256:40b21d15e109f3cde56abb8f5feed304535a95d5ae9d041cd233c0077019c076` |
| `uccl` | `sha256:28eded57acffce6b8551fcd05507732f3ff37fe795095e5a0a3fe52a458082d5` |
| `deepep-v1-nvshmem` | `sha256:8645ad603b0ad60afceccea414d34404bfa46c722ff92fd179829894341d1138` |
| `deepep-v2-gin-gda` | `sha256:c42ab1921a66a007b782e8712ec73649dff8969a8a2c00aafc6655f9c1b0c9d7` |

The common stack is Bridge commit `c93251151adeeadbae3ff2a2bf5ee7a1c34cff01`, MCore commit `16ad357ee7973af32916fc1ca39d71065e5f03d4`, Torch `2.13.0a0+8145d630e8.nv26.06`, CUDA 13.3, TransformerEngine `2.17.1+4329ff84`, libfabric `2.6.0amzn1.0`, EFA installer `1.50.0`, and GDRCopy `v2.5.2`. Torch's NCCL build version and `ncclGetVersion()` both report NCCL 2.31.2. Runtime manifests resolve exactly 1 loaded NCCL library, `/opt/nccl/build/lib/libnccl.so.2.31.2`, with SHA-256 `0edb3fea9939b3dc8e91e295e9f89fb3b2bb2404a93eb280e60099fcfe8c5975`.

## Correctness gates

| Gate | Real GPU scope | Outcome |
|---|---|---|
| Gate A, ElasticBuffer contract | 16 ranks, 512 tokens per rank, hidden size 7,168 dimensions, 384 experts, top-k 8 choices | `PASS` for synchronous and asynchronous dispatch/combine, distinct cached handles, 65,536 expected selections, and 0 dropped selections |
| Gate B, Router Replay against NCCL | 2 ranks and 16 ranks, each at 512 tokens per rank and 2,048 tokens per rank | `PASS` for record, forward replay, replay backward, exact global expert counts, 0 dropped selections, and all numeric errors inside independent NCCL BF16 self-repeat envelopes |
| Gate C, pipeline and overlap lifetime | 32 nodes and 256 GPUs | `PASS` with TP 8 ranks, PP 8 stages, EP 32 ranks, 16 microbatches, full recomputation with overlap off, and virtual pipeline size 2 stages with overlap on |
| Gate D, short Kimi-K2 training | 32 nodes and 256 GPUs, 4 optimizer iterations | `PASS` for finite loss, gradient norm, and sampled update norm; exact route hashes where schedules are comparable; and 0 dropped selections |

The overlap-off NCCL baseline and DeepEP v2 runs produced byte-identical top-k route hashes on all 32 node ranks. Each run recorded 61,440 route records, 1,006,632,960 valid selections, and 0 dropped selections. At iteration 4, the baseline gradient norm was `0.9073536396` parameter-gradient units with sampled update L2 norm `0.7729348865` parameter units; DeepEP v2 reported `0.9103103876` parameter-gradient units and `0.7729752397` parameter units.

The final overlap-on DeepEP v2 run completed on all 32 nodes with trainer exit code 0. It recorded 30,720 route records, 503,316,480 valid selections, and 0 dropped selections under the virtual-pipeline schedule. Its fourth gradient norm was `0.9374347925` parameter-gradient units and its sampled update L2 norm was `0.8425297391` parameter units. Runtime inspection found 7 MoE layers, 7 `MoEFlexTokenDispatcher` instances, 7 `_DeepepV2Manager` instances with 32-rank process groups, and 1 shared `ElasticBuffer` workspace. NCCL loaded the `Libfabric_GDAKI` GIN plugin version 14, selected EFA Direct across 8 NICs per node, and reported GPU Direct RDMA enabled on all 8 HCAs sampled on node rank 0.

Earlier overlap attempts are excluded. They exposed that MCore's fine-grained schedule preserved dispatch inputs and restored per-schedule-node probability state only for backend `deepep`, not backend `deepep_v2`. The accepted patch applies the same lifetime rules to `deepep_v2`; the final overlap-on run and the dedicated overlap regression both pass.

## Capacity outcome

The campaign selected 32 free B200 nodes only after the vLLM campaign had moved out of ap-south-1. Four foreign tenant nodes remained explicitly protected and were never selected. After the final correctness harvest, EKS recorded a `Shutdown` event for each of 36 B200 nodes, every B200 node changed to `Ready=Unknown`, and EC2 reported all 36 Capacity Block instances as `shutting-down`. The scored controller's fresh occupancy and readiness check therefore found 0 runnable B200 nodes and refused to create workload pods.

The PR 5 scale matrix remains `NOT_RUN_INSUFFICIENT_CAPACITY` because the cluster had 0 B300 physical domains, below the required 22-domain, 23-domain, and 32-domain cells. No scale verdict is inferred from the normal Kimi topology.

## Raw provenance and custody

Durable artifacts are under:

`/mnt/fsx/ubuntu/workspace/artifacts/adai-kimi-k2-megatron-ep/20260824T030931Z-kimi-k2-megatron-ep-b200-256g`

Key accepted runs are `26.08/kimi-k2/gate-a-16r-final-image`, the four `gate-b-*-mcore-direct` directories, `gate-d-baseline-off-256g-reroute`, `gate-cd-v2-off-256g-eager-fix`, and `gate-cd-v2-on-overlapfix`. The exact reclaim evidence is in `control/aws-capacity-reclaim-20260824T1052Z`; the terminal machine-readable outcome is `results/b200-256gpu-scored.STATUS`. Every accepted raw-pod run has a `STATUS`, a `SHA256SUMS` index, 32 custody verifiers for 32-node runs, pod logs, runtime manifests, route summaries, and telemetry. Failure and diagnostic directories remain preserved but are excluded from accepted results.

## Teardown

The campaign deleted only its owned namespaces and its 2 device-plugin DaemonSets, released its shared lease claim, and did not delete or modify the Capacity Block, EC2 instances, EKS nodes, or foreign workload objects. The terminal teardown census and root custody hash are recorded under the artifact root.
