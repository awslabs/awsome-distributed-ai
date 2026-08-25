# EP Backend Comparison Results on B200

The replacement measures one synthetic decode dispatch-plus-combine workload with one external CUDA timing boundary and one logical payload definition. It is not an end-to-end training or serving benchmark. EP32 means 32 GPU ranks on 4 `p6-b200.48xlarge` nodes, not 32 instances.

## Result

UCCL has the lowest latency in 3 of the 4 measured workload cells. DeepEP V2 has the lowest latency at EP16 with BF16 dispatch. DeepEP V2 is faster than DeepEP V1 in all 4 cells. These observations apply only to the configuration and scales in this report.

Each value is the median across 3 independent process starts. Each process start contributes the median of 100 slowest-rank measured iterations after 20 warmup iterations.

| EP size | Dispatch dtype | Backend | Latency | 95% bootstrap CI | Run-to-run CV | Input throughput | Logical throughput | Scale-out logical throughput |
|---:|:---:|:---|---:|:---:|---:|---:|---:|---:|
| 16 ranks | FP8 | UCCL | 0.5213 ms | [0.5190, 0.5277] ms | 0.86% | 3,928,790.51 tokens/s | 42.68 GB/s/rank | 21.34 GB/s/rank |
| 16 ranks | FP8 | DeepEP V1 NVSHMEM | 1.0179 ms | [1.0167, 1.0203] ms | 0.18% | 2,011,977.59 tokens/s | 21.86 GB/s/rank | 10.93 GB/s/rank |
| 16 ranks | FP8 | DeepEP V2 NCCL GIN | 0.6219 ms | [0.6180, 0.6375] ms | 1.65% | 3,293,197.58 tokens/s | 35.78 GB/s/rank | 17.89 GB/s/rank |
| 16 ranks | BF16 | UCCL | 0.5920 ms | [0.5918, 0.5978] ms | 0.58% | 3,459,365.90 tokens/s | 49.59 GB/s/rank | 24.80 GB/s/rank |
| 16 ranks | BF16 | DeepEP V1 NVSHMEM | 1.0297 ms | [1.0251, 1.0304] ms | 0.28% | 1,988,936.56 tokens/s | 28.51 GB/s/rank | 14.26 GB/s/rank |
| 16 ranks | BF16 | DeepEP V2 NCCL GIN | 0.4850 ms | [0.4826, 0.5043] ms | 2.42% | 4,222,332.23 tokens/s | 60.53 GB/s/rank | 30.27 GB/s/rank |
| 32 ranks | FP8 | UCCL | 0.7655 ms | [0.7640, 0.7714] ms | 0.51% | 5,350,835.03 tokens/s | 29.07 GB/s/rank | 21.80 GB/s/rank |
| 32 ranks | FP8 | DeepEP V1 NVSHMEM | 1.5542 ms | [1.5513, 1.5628] ms | 0.38% | 2,635,371.65 tokens/s | 14.32 GB/s/rank | 10.74 GB/s/rank |
| 32 ranks | FP8 | DeepEP V2 NCCL GIN | 0.9537 ms | [0.9217, 0.9675] ms | 2.48% | 4,294,725.45 tokens/s | 23.33 GB/s/rank | 17.50 GB/s/rank |
| 32 ranks | BF16 | UCCL | 0.8686 ms | [0.8683, 0.8696] ms | 0.08% | 4,715,590.93 tokens/s | 33.80 GB/s/rank | 25.35 GB/s/rank |
| 32 ranks | BF16 | DeepEP V1 NVSHMEM | 1.5589 ms | [1.5526, 1.5714] ms | 0.61% | 2,627,446.53 tokens/s | 18.83 GB/s/rank | 14.13 GB/s/rank |
| 32 ranks | BF16 | DeepEP V2 NCCL GIN | 0.9509 ms | [0.9482, 0.9606] ms | 0.68% | 4,307,661.17 tokens/s | 30.88 GB/s/rank | 23.16 GB/s/rank |

The maximum run-to-run CV is 2.48%. The 95% intervals use 20,000 bootstrap resamples of the 3 independent process-start medians. With only 3 independent starts, the intervals describe this campaign but should not be read as precise estimates of production variability.

## Paired DeepEP V2 latency deltas

Starts are paired by EP size, dtype, input, route, named nodes, and rotation index. A positive value means DeepEP V2 had lower latency than the baseline. A direction is marked supported only when the paired bootstrap interval excludes 0% and both arms have no more than 5% run-to-run CV.

| EP size | Dispatch dtype | Baseline | Median V2 latency reduction | 95% bootstrap CI | Direction supported |
|---:|:---:|:---|---:|:---:|:---:|
| 16 ranks | FP8 | UCCL | -19.30% | [-22.84%, -17.11%] | Yes |
| 16 ranks | FP8 | DeepEP V1 NVSHMEM | 39.05% | [37.37%, 39.22%] | Yes |
| 16 ranks | BF16 | UCCL | 18.44% | [14.82%, 18.87%] | Yes |
| 16 ranks | BF16 | DeepEP V1 NVSHMEM | 52.68% | [51.06%, 53.13%] | Yes |
| 32 ranks | FP8 | UCCL | -24.83% | [-25.41%, -20.40%] | Yes |
| 32 ranks | FP8 | DeepEP V1 NVSHMEM | 38.64% | [37.64%, 41.02%] | Yes |
| 32 ranks | BF16 | UCCL | -9.35% | [-10.63%, -9.16%] | Yes |
| 32 ranks | BF16 | DeepEP V1 NVSHMEM | 38.87% | [38.76%, 39.18%] | Yes |

"Direction supported" is a campaign-level reproducibility rule, not a universal performance claim or a formal significance test.

## Common comparison boundary

The harness holds the semantic workload and measurement rules constant across the 3 backends:

| Control | Value |
|---|---|
| Input | Deterministic BF16 tensor, 128 tokens/rank |
| Model shape | Hidden size 7,168, 256 experts, top-k 8 experts/token |
| Operation | FP8 or BF16 dispatch followed by BF16 combine |
| Expert work | Identity-expert semantics; expert compute is outside the timed boundary |
| Timing | CUDA Events from input preparation through dispatch and combine completion |
| Rank reduction | Maximum elapsed time across all ranks for every measured iteration |
| Warmup and measurement | 20 warmup iterations and 100 measured iterations per dtype and process start |
| Replication | 3 independent process starts per backend and workload cell |
| Order control | Backend order and dtype order rotate across process starts |
| Hardware control | The same named nodes are used for every backend at each EP size |
| Input control | Input and routing SHA-256 hashes must match across all backends and starts in each EP size |
| Runtime control | GPU model, PyTorch, CUDA, and NCCL versions must match across every scored result |
| Image control | Every backend image must be digest-pinned and invariant across the matrix |
| Correctness | Every rank must pass the common identity-expert output check before timing |

DeepEP V2 uses non-expanded dispatch, which sends a token once per destination rank. The correctness path applies the local gated identity-expert reduction before combine so V2 and the expanded backends return the same weighted token. This normalizes semantic output, not the internal algorithm.

## Logical throughput definition

The cross-backend throughput columns use the same useful-payload numerator. Each valid expert assignment contributes its dispatch tensor, FP8 scales when FP8 is selected, and its BF16 combine tensor. Backend-specific metadata is excluded. Scale-out logical bytes count only assignments whose destination expert is on a different node.

```text
logical GB/s/rank = average logical bytes/rank / median slowest-rank latency
scale-out logical GB/s/rank = average remote logical bytes/rank / median slowest-rank latency
```

These values are effective logical throughput, not observed wire bandwidth. They intentionally replace the retired side-by-side native bandwidth columns. In DeepEP V2, for example, SO bandwidth uses `num_scaleout_bytes / t`, SU bandwidth uses `num_scaleup_bytes / t`, and the trailing `bytes` field is the SU numerator. DeepEP V1 and UCCL use their own byte accounting and aggregation boundaries. Those native metrics remain useful for diagnosing a backend, but comparing them as one GB/s metric is not valid.

## Correctness and admission

All 36 scored dtype results passed on every rank. The normalized-difference tolerance was `9e-4` dimensionless for FP8 dispatch and `1e-5` dimensionless for BF16 dispatch. The matrix contains 18 distributed process starts: 2 EP sizes multiplied by 3 backends and 3 independent starts, with 2 dtype results from each start.

Before scoring, the DeepEP V2 EP16 admission run also verified:

- `uvm_disable_hmm=1` on all 4 selected hosts;
- `/dev/gdrdrv` present as a character device on all 4 selected hosts; and
- an INFO-level GDAKI context in the DeepEP V2 log.

## Environment and provenance

| Field | Value |
|---|---|
| Campaign | `fair-ep-b200-20260824t195145z` |
| Date | 2026-08-24 UTC |
| AWS Region and Availability Zone | `ap-south-1`, `ap-south-1c` |
| EKS cluster | `ml-clusters-shared-ap-south-1` |
| Instance type | `p6-b200.48xlarge` |
| Node topology | 4 nodes, 8 NVIDIA B200 GPUs/node, 8 allocatable EFA devices/node |
| Runtime | PyTorch `2.13.0+cu130`, CUDA `13.0`, NCCL `2.29.7` |

Digest-pinned images:

| Backend | Image |
|---|---|
| UCCL | `159553542841.dkr.ecr.ap-south-1.amazonaws.com/adai/dsv3-ep-backend-comparison-uccl@sha256:a703a0d35916bbd51b2d34d968626d9617cbceb328d64b920f21ad159d93c91a` |
| DeepEP V1 NVSHMEM | `159553542841.dkr.ecr.ap-south-1.amazonaws.com/adai/dsv3-ep-backend-comparison-deepep-v1-nvshmem@sha256:175a8470e11bc7832024941d1775923d5e8ecae2c6dc06183d5f1653d66b4fac` |
| DeepEP V2 NCCL GIN | `159553542841.dkr.ecr.ap-south-1.amazonaws.com/adai/dsv3-ep-backend-comparison-deepep-v2-gin-gda@sha256:4d07367ea290c5d6ec3c02b223ac819feed5240a46fd4a6492421e9c0853dbeb` |

Selected nodes:

```text
ip-10-6-124-201.ap-south-1.compute.internal
ip-10-6-126-97.ap-south-1.compute.internal
ip-10-6-127-248.ap-south-1.compute.internal
ip-10-6-71-166.ap-south-1.compute.internal
```

## Concurrent campaign isolation

The campaign ran in coordinated observe mode beside the vLLM campaign identified by Lease holder `dsv3-b200-parallel-r7-20260824t2122z`. The vLLM campaign had 12 protected nodes. The selected-node overlap was 0 nodes. The runner did not mutate the shared Lease and rechecked its exact holder before each arm and before aggregation.

Node disjointness prevents direct GPU and host contention. It does not prove absence of shared network-fabric effects, so these results remain specific to the observed cluster conditions.

## Artifact custody and teardown

The durable artifact contains 19 canonical result files and 38 result records, including 36 scored records and 2 admission records. It retains the rendered manifests, all-rank logs, Pod descriptions, image references, input and route hashes, admission evidence, cluster snapshots, teardown evidence, and a `SHA256SUMS` manifest covering 523 files.

The detached runner returned a nonzero status after all 18 scored jobs passed because one backend diagnostic was appended to a JSON result line. The parser was corrected, canonical records were re-extracted from the preserved rank-zero logs, and the complete matrix was revalidated. No benchmark value was reconstructed or rerun during recovery.

Final teardown verification passed: the owned namespace was absent, 0 owned resources remained, all 4 selected nodes were Ready, and 0 selected-node GPU Pods remained. The concurrent vLLM Lease holder was unchanged.

The committed machine-readable summary is [`results/b200-ap-south-1-2026-08-24.json`](results/b200-ap-south-1-2026-08-24.json), SHA-256 `129877f60412e4a87fe1b8dd29074bcf803457d9c8b5c816276616fd09700f66`.

## Limits

This benchmark does not measure prefill, expert compute, communication/computation overlap, memory footprint, end-to-end training, serving throughput, TTFT, TPOT, or end-to-end latency. It covers only EP16 and EP32 on B200 in one Availability Zone, with 3 independent starts per cell. It does not establish behavior on B300, H100, larger EP domains, different routing distributions, or production Kimi K2 training replicas.
