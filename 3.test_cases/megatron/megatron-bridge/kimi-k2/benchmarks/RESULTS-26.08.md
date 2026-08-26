# Kimi-K2 Megatron-Bridge EP results, NeMo 26.08

Primary scored campaign ID: `20260825T111800Z-k2-b200-scored-clean-3r-r3`

UCCL repeat 3 retry campaign ID: `20260825T142107Z-k2-b200-r3-uccl-retry-r1`

Expanded overlap source campaign ID: `20260825T210749Z-k2-b200-expanded-3c-3r-r5`

DeepEP v2 overlap completion campaign ID: `20260826T215100Z-k2-b200-v2-completion-r1`

Correctness qualification campaign ID: `20260824T030931Z-kimi-k2-megatron-ep-b200-256g`

Performance dates: `2026-08-25 UTC` through `2026-08-26 UTC`

The requested 256-GPU B300 headline remains `NOT_RUN_INSUFFICIENT_CAPACITY`. At `2026-08-25T11:19:24Z`, the ap-south-1 cluster had 0 available B300 nodes, below the requirement of 32 B300 nodes. The primary scored milestone ran on 32 `p6-b200.48xlarge` instances and 256 B200 GPUs from the same Capacity Block.

## Topology

One Kimi-K2 training replica is 64 GPUs: tensor parallelism uses 8 ranks and pipeline parallelism uses 8 stages. The 256-GPU deployment contains 4 data-parallel replicas, each with expert parallelism across 32 ranks and expert tensor parallelism of 1 rank.

| Property | Scored value |
|---|---|
| Hardware | 32 `p6-b200.48xlarge` nodes, 8 B200 GPUs per node, 256 B200 GPUs total |
| Model | Literal Kimi-K2 architecture, random initialization, mock data |
| Architecture | 61 layers, hidden size 7,168 dimensions, 384 experts, top-k 8 routing choices |
| Precision | BF16 |
| Parallelism | TP 8 ranks, PP 8 stages, EP 32 ranks, ETP 1 rank, DP 4 replicas |
| Batch | Global batch 256 samples, microbatch 4 samples |
| Sequence | 4,096 tokens per sample |
| Communication cell | Expert-parallel overlap off, CUDA graphs off, forced load balancing on |
| Network | 8 EFA devices per node, EFA kernel module `3.3.0g` |
| Job length | 40 optimizer iterations per job, with 8 warmup iterations discarded and 32 steady iterations scored |
| Replication | 3 independent fresh job starts per arm |
| Random seed | 1,234 dimensionless |

The selected node list was byte-identical for the primary campaign and the UCCL retry. The campaign selected the 32 free B200 nodes only after the concurrent vLLM campaign moved out of ap-south-1. Four foreign tenant nodes remained explicitly protected and were never selected.

## B300 headline

| Arm | Throughput result, MB4, 256 B300 GPUs | Capacity evidence |
|---|---:|---|
| `nccl-alltoall` | `NOT_RUN_INSUFFICIENT_CAPACITY` | 0 available B300 nodes; 32 B300 nodes required |
| `uccl` | `NOT_RUN_INSUFFICIENT_CAPACITY` | 0 available B300 nodes; 32 B300 nodes required |
| `deepep-v1-nvshmem` | `NOT_RUN_INSUFFICIENT_CAPACITY` | 0 available B300 nodes; 32 B300 nodes required |
| `deepep-v2-gin-gda` | `NOT_RUN_INSUFFICIENT_CAPACITY` | 0 available B300 nodes; 32 B300 nodes required |

No result from another GPU type, release, or topology is copied into the B300 table.

## B200 primary scored cell

Each job contributes the median of its 32 steady iteration times. The per-arm latency, throughput, and TFLOPS/GPU values below are medians across 3 independent job starts. Run-to-run CV is the sample standard deviation of the 3 run medians divided by their mean. Iteration-time reduction is paired by repeat against `nccl-alltoall` and uses `(baseline_ms - arm_ms) / baseline_ms * 100%`. The interval is a paired repeat-level bootstrap with 10,000 resamples, a 1,234 dimensionless seed, and a 95% percentile interval.

| Arm | Repeat 1 median, ms | Repeat 2 median, ms | Repeat 3 median, ms | Median of run medians, ms | Median throughput, tokens/s | Median performance, TFLOPS/GPU | Run-to-run CV, % | Paired time reduction vs NCCL, mean [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nccl-alltoall` | 7,647.75 | 7,645.80 | 7,653.10 | 7,647.75 | 137,109 | 104.825 | 0.049 | Reference |
| `uccl` | 6,348.45 | 6,338.95 | 6,307.95 | 6,338.95 | 165,418 | 125.709 | 0.334 | 17.22% [16.99%, 17.58%] |
| `deepep-v1-nvshmem` | 6,908.90 | 6,931.70 | 6,906.85 | 6,908.90 | 151,772 | 115.594 | 0.199 | 9.58% [9.34%, 9.75%] |
| `deepep-v2-gin-gda` | 5,980.70 | 5,977.90 | 5,952.95 | 5,977.90 | 175,409 | 133.147 | 0.256 | 21.94% [21.80%, 22.22%] |

`nccl-alltoall` is the end-to-end legacy dispatcher baseline, not a transport-only control. The three flex-dispatcher arms provide the native EP implementation comparison:

| Treatment relative to flex-arm reference | Paired mean iteration-time reduction, % | Bootstrap 95% CI, % |
|---|---:|---:|
| `deepep-v2-gin-gda` relative to `uccl` | 5.71 | [5.63, 5.79] |
| `deepep-v2-gin-gda` relative to `deepep-v1-nvshmem` | 13.67 | [13.43, 13.81] |
| `uccl` relative to `deepep-v1-nvshmem` | 8.44 | [8.11, 8.67] |

All run-to-run CV values are at most 0.335%, and every paired repeat favors the same arm as its aggregate interval. `deepep-v2-gin-gda` is the winner for this B200, MB4, overlap-off cell. This is not a B300 result and does not establish a winner for overlap-on or MB1 cells. With only 3 independent starts per arm, the bootstrap intervals describe this campaign rather than a broad population.

### First iteration

The first measured optimizer step is reported separately because it can include cold runtime and JIT work. No first-iteration value contributes to the steady-state score.

| Arm | Repeat 1 first iteration, s | Repeat 2 first iteration, s | Repeat 3 first iteration, s |
|---|---:|---:|---:|
| `nccl-alltoall` | 259.66 | 260.47 | 260.68 |
| `uccl` | 239.97 | 240.50 | 240.36 |
| `deepep-v1-nvshmem` | 266.57 | 267.92 | 266.37 |
| `deepep-v2-gin-gda` | 521.13 | 525.15 | 522.15 |

### Acceptance and exclusion

The accepted dataset contains exactly 1 `PASS` result for each `(cell, repeat, arm)` key, totaling 12 accepted jobs. All required validity gates pass. The accepted jobs preserve 384 trainer logs, 384 runtime manifests, 384 route summaries, and 384 node custody files. Every accepted DeepEP v2 job additionally proves `_DeepepV2Manager`, `ElasticBuffer`, `NCCL_GIN_TYPE=5`, and a GDAKI context at runtime.

The original repeat 3 UCCL attempt entered the training loop but emitted no iteration metric for more than 720 seconds. The runner recorded `FAIL_TRAINER_EXIT` at `2026-08-25T14:14:51Z`. That attempt was excluded before aggregation and remains durably preserved with 32 of 32 node directories and 32 of 32 custody logs. A same-condition UCCL-only retry used the identical 32-node list, image digest, model configuration, and training entrypoint SHA-256. It completed 40 of 40 iterations and recorded `PASS` at `2026-08-25T14:35:02Z`.

## DeepEP v2 overlap completion

DeepEP v2 now has 3 accepted fresh job starts for each overlap-on cell. Repeats 1 and 2 come from the expanded overlap source campaign, and repeat 3 comes from the DeepEP v2 completion campaign. Both campaigns used the same 32 B200 nodes, immutable image digest, training entrypoint SHA-256, model topology, batch configuration, 1,234 dimensionless random seed, and timing protocol. Each job completed 40 optimizer iterations; the first 8 iterations were discarded and the remaining 32 steady iterations were scored.

| Cell | Microbatch, samples | Repeat 1 median, ms | Repeat 2 median, ms | Repeat 3 median, ms | Median of run medians, ms | Median throughput, tokens/s | Median performance, TFLOPS/GPU | Run-to-run CV, % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `throughput-overlap` | 4 | 3,451.55 | 3,469.55 | 3,450.85 | 3,451.55 | 303,799 | 226.866 | 0.307 |
| `small-message-overlap` | 1 | 9,730.80 | 9,748.15 | 9,694.60 | 9,730.80 | 107,758 | 83.066 | 0.281 |

All 6 DeepEP v2 jobs are `PASS`. Each job passes all 14 required validity gates, including `_DeepepV2Manager`, `ElasticBuffer`, `NCCL_GIN_TYPE=5`, a GDAKI context, EFA, finite loss and gradient norm, 0 dropped token selections, and exactly 1 loaded NCCL library. Each job preserves 32 trainer logs, 32 runtime manifests, 32 route summaries, and 32 node custody files. Cross-arm ranking is not reported for these cells until the remaining arm and repeat keys have accepted results under the same conditions.

## Final image quartet

| Arm | Immutable ap-south-1 image digest |
|---|---|
| `nccl-alltoall` | `sha256:40b21d15e109f3cde56abb8f5feed304535a95d5ae9d041cd233c0077019c076` |
| `uccl` | `sha256:28eded57acffce6b8551fcd05507732f3ff37fe795095e5a0a3fe52a458082d5` |
| `deepep-v1-nvshmem` | `sha256:8645ad603b0ad60afceccea414d34404bfa46c722ff92fd179829894341d1138` |
| `deepep-v2-gin-gda` | `sha256:c42ab1921a66a007b782e8712ec73649dff8969a8a2c00aafc6655f9c1b0c9d7` |

The common stack is Bridge `0.6.0` at commit `c93251151adeeadbae3ff2a2bf5ee7a1c34cff01`, MCore `0.19.0` at commit `16ad357ee7973af32916fc1ca39d71065e5f03d4`, Torch `2.13.0a0+8145d630e8.nv26.06`, CUDA `13.3`, TransformerEngine `2.17.1+4329ff84`, aws-ofi-nccl `1.21.1`, libfabric `2.6.0amzn1.0`, EFA installer `1.50.0`, and GDRCopy `v2.5.2`. Torch's NCCL build version and `ncclGetVersion()` both report NCCL `2.31.2`. Runtime manifests resolve exactly 1 loaded NCCL library, `/opt/nccl/build/lib/libnccl.so.2.31.2`, with SHA-256 `0edb3fea9939b3dc8e91e295e9f89fb3b2bb2404a93eb280e60099fcfe8c5975`.

The backend source pins are UCCL commit `a8c94e485da302b7cc963d2f74edb5b4e9115748`, DeepEP v1 commit `567632dd59810d77b3cc05553df953cc0f779799`, and DeepEP v2 synthetic commit `0ff264eb08d196621f49c7d011270c62f8716996` with tree `ab43791defa8961b88fdee1a63e8877e317da9af`.

## Correctness gates

| Gate | Real GPU scope | Outcome |
|---|---|---|
| Gate A, ElasticBuffer contract | 16 ranks, 512 tokens per rank, hidden size 7,168 dimensions, 384 experts, top-k 8 choices | `PASS` for synchronous and asynchronous dispatch/combine, distinct cached handles, 65,536 expected selections, and 0 dropped selections |
| Gate B, Router Replay against NCCL | 2 ranks and 16 ranks, each at 512 tokens per rank and 2,048 tokens per rank | `PASS` for record, forward replay, replay backward, exact global expert counts, 0 dropped selections, and all numeric errors inside independent NCCL BF16 self-repeat envelopes |
| Gate C, pipeline and overlap lifetime | 32 nodes and 256 GPUs | `PASS` with TP 8 ranks, PP 8 stages, EP 32 ranks, 16 microbatches, full recomputation with overlap off, and virtual pipeline size 2 stages with overlap on |
| Gate D, short Kimi-K2 training | 32 nodes and 256 GPUs, 4 optimizer iterations | `PASS` for finite loss, gradient norm, and sampled update norm; exact route hashes where schedules are comparable; and 0 dropped selections |

The overlap-off NCCL baseline and DeepEP v2 correctness runs produced byte-identical top-k route hashes on all 32 node ranks. Each run recorded 61,440 route records, 1,006,632,960 valid selections, and 0 dropped selections. At optimizer iteration 4, the baseline gradient norm was 0.9073536396 parameter-gradient units with sampled update L2 norm 0.7729348865 parameter units; DeepEP v2 reported 0.9103103876 parameter-gradient units and 0.7729752397 parameter units.

The final overlap-on DeepEP v2 correctness run completed on all 32 nodes with trainer exit code 0 dimensionless. It recorded 30,720 route records, 503,316,480 valid selections, and 0 dropped selections under the virtual-pipeline schedule. Its fourth gradient norm was 0.9374347925 parameter-gradient units and its sampled update L2 norm was 0.8425297391 parameter units. Runtime inspection found 7 MoE layers, 7 `MoEFlexTokenDispatcher` instances, 7 `_DeepepV2Manager` instances with 32-rank process groups, and 1 shared `ElasticBuffer` workspace. NCCL loaded the `Libfabric_GDAKI` GIN plugin version 14, selected EFA Direct across 8 NICs per node, and reported GPU Direct RDMA enabled on all 8 HCAs sampled on node rank 0.

Earlier overlap attempts are excluded. They exposed that MCore's fine-grained schedule preserved dispatch inputs and restored per-schedule-node probability state only for backend `deepep`, not backend `deepep_v2`. The accepted patch applies the same lifetime rules to `deepep_v2`; the final overlap-on run and the dedicated overlap regression both pass.

## PR 5 scale gate

The PR 5 scale matrix remains `NOT_RUN_INSUFFICIENT_CAPACITY` because the cluster had 0 B300 physical domains, below the required cells of 22 domains, 23 domains, and 32 domains. No PR 5 scale verdict is inferred from the normal Kimi topology.

## Raw provenance and custody

The correctness qualification artifacts are under:

`/mnt/fsx/ubuntu/workspace/artifacts/adai-kimi-k2-megatron-ep/20260824T030931Z-kimi-k2-megatron-ep-b200-256g`

Its root `SHA256SUMS` file has SHA-256 `350f6d6d6cc96f82ce6cf622cbd9ffc317db310ea86c123431de453a12a3b70d`.

The primary scored artifacts are under:

`/mnt/fsx/ubuntu/workspace/artifacts/adai-kimi-k2-megatron-ep/20260825T111800Z-k2-b200-scored-clean-3r-r3`

Its root `SHA256SUMS` file has SHA-256 `54cfa374a52982ef780863e0ceed1879a9a0f8e57baf11711cbed8565277d96a`, and `results/index.json` has SHA-256 `817ffc83d3ce31987969f587f937a0ffc10dd5cbaf144b68608634230572c63c`. The excluded UCCL attempt's `SHA256SUMS` file has SHA-256 `9cfcf7ec29d30fa6e73484c4f2714a35bbc1d736dd278e2eb050058d32179e80`.

The UCCL repeat 3 retry artifacts are under:

`/mnt/fsx/ubuntu/workspace/artifacts/adai-kimi-k2-megatron-ep/20260825T142107Z-k2-b200-r3-uccl-retry-r1`

Its root `SHA256SUMS` file has SHA-256 `0866a88491255838a3920d03cd9e9169d0ca94554f8fa7bcabb4b51f9f38171b`, `results/index.json` has SHA-256 `6995fa4ef5cb5ff84708b0dd519d7ef10427228c3383e91d61912fcd2cd3e52d`, and the accepted UCCL arm's `SHA256SUMS` file has SHA-256 `bcb33b8a5bb89c374c986d0f2c6e7ac284a91ca89780122dec5462934f55a542`. Full root checksum verification passes for both scored artifact roots.

The expanded overlap source artifacts are under:

`/mnt/fsx/ubuntu/workspace/artifacts/adai-kimi-k2-megatron-ep/20260825T210749Z-k2-b200-expanded-3c-3r-r5`

Its root `SHA256SUMS` file has SHA-256 `591d53da34a3f672c219a161315d960565823f0e4d329e53493b0f6d3013f940`, and `results/index.json` has SHA-256 `5d4566e2ea1a830f91deb7e5815b557e1ec5e68144874160af62a166c06902c8`.

The DeepEP v2 overlap completion artifacts are under:

`/mnt/fsx/ubuntu/workspace/artifacts/adai-kimi-k2-megatron-ep/20260826T215100Z-k2-b200-v2-completion-r1`

Its root `SHA256SUMS` file has SHA-256 `68e0017f0bf577c031fc7fef42613689d698a5cf0c8595fd08e7cf07af2e8a37`, and `results/index.json` has SHA-256 `a1026bb415090284a66a4dc65b14c9561248e68e08fa78ddc6b6bdf459a63566`. Full root checksum verification passes.

## Teardown

Both scored campaign controllers deleted only their owned Kubernetes resources, released the shared Lease claim, and preserved the Capacity Block, EC2 instances, EKS nodes, and foreign workload objects. Finalization recorded controller, parser, census, and teardown exit codes. The primary controller exit code is 1 dimensionless because the excluded UCCL attempt failed; parser, census, and teardown exit codes are each 0 dimensionless. All retry exit codes are 0 dimensionless. The terminal live census found both owned namespaces absent, 0 owned GPU pods, and an empty Lease holder with a 1-second lease duration.

The DeepEP v2 overlap completion controller also recorded controller, parser, census, and teardown exit codes of 0 dimensionless. Its terminal live census found the owned namespace absent, 0 owned pods, 0 active GPU pods, 0 requested GPUs, and an empty Lease holder with a 1-second lease duration.
