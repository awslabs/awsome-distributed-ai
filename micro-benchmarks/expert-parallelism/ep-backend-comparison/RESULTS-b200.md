# EP-Backend Comparison Results: B200 with DeepEP V2

This page records 2 separate B200 campaigns in `ap-south-1`. The fair common-boundary decode rerun completed on 24 August 2026 is the primary cross-backend comparison. The earlier backend-native campaign is retained below as historical diagnostic evidence only. These results are separate from the historical [B300 results](RESULTS.md) and [P5/H100 results](RESULTS-p5.md). Results from different campaigns, timing boundaries, byte-accounting conventions, or GPU generations are not combined.

## Fair common-boundary decode rerun

The rerun passed all 36 scored dtype results from 18 distributed jobs. UCCL had the lowest median latency in 3 of 4 workload cells: EP16 FP8, EP32 FP8, and EP32 BF16. DeepEP V2 had the lowest median latency in EP16 BF16. DeepEP V2 was faster than DeepEP V1 in all 4 workload cells.

These conclusions apply only to the exact synthetic dispatch-plus-combine workload and timing boundary below. They do not establish end-to-end training or serving throughput, prefill performance, or a universal backend winner. Each arm and workload cell has 3 independent process starts. The 95% bootstrap intervals therefore summarize these 3 starts and should not be treated as a high-power population estimate.

### Common workload and method

| Field | Value |
|---|---|
| Measurement period | 24 August 2026 |
| Region and Availability Zone | `ap-south-1`, `ap-south-1c` |
| EKS cluster | `ml-clusters-shared-ap-south-1` |
| Hardware | `p6-b200.48xlarge`, 8 B200 GPUs and 8 EFA devices per node |
| EP16 topology | 2 nodes, 16 ranks |
| EP32 topology | 4 nodes, 32 ranks |
| Workload | 128 tokens/rank, hidden size 7,168, top-k 8, experts 256 |
| Dispatch dtype | FP8 and BF16 |
| Combine dtype | BF16 |
| Timing boundary | BF16 input ready through dispatch and combine completion, measured with CUDA Events and reduced to the slowest rank |
| Logical payload | Useful dispatch tensor, FP8 scales when selected, and BF16 combine tensor for every valid expert assignment; backend metadata excluded |
| Iterations | 20 warmup iterations and 100 measured iterations per dtype and start |
| Replication | 3 independent process starts per arm and workload cell |
| Order control | Backend order rotated across starts; dtype order also rotated |
| Input control | Deterministic BF16 input, exact top-k route, and weights; matching SHA-256 input and route hashes required within each EP size |
| Correctness | 36 of 36 scored dtype results passed; FP8 tolerance `9e-4` dimensionless and BF16 tolerance `1e-5` dimensionless |

All 3 images used PyTorch `2.13.0+cu130`, CUDA 13.0, NCCL 2.29.7, and digest-pinned images:

| Backend | Scored image digest |
|---|---|
| UCCL | `sha256:a703a0d35916bbd51b2d34d968626d9617cbceb328d64b920f21ad159d93c91a` |
| DeepEP V1 NVSHMEM | `sha256:175a8470e11bc7832024941d1775923d5e8ecae2c6dc06183d5f1653d66b4fac` |
| DeepEP V2 NCCL GIN | `sha256:4d07367ea290c5d6ec3c02b223ac819feed5240a46fd4a6492421e9c0853dbeb` |

The 3 backends ran serially on the same named nodes for each EP size. A concurrent vLLM campaign used 12 disjoint nodes, with 0 selected-node overlap. This campaign observed the shared Lease and did not mutate the vLLM campaign's Lease, nodes, namespace, images, or artifacts. Because the campaigns shared the cluster fabric, unobserved background-fabric effects cannot be excluded completely; order rotation and paired starts reduce but do not eliminate that risk.

### Common-boundary results

Values are medians across 3 independent starts. Logical throughput uses the common logical-byte numerator defined above, not a backend-native bandwidth field. Lower latency is better.

| EP size | Dispatch dtype | Backend | Median latency | 95% bootstrap CI | Run-to-run CV | Input throughput | Logical throughput | Scale-out logical throughput |
|---:|:---:|:---|---:|:---:|---:|---:|---:|---:|
| 16 ranks | FP8 | UCCL | **0.5213 ms** | [0.5190, 0.5277] ms | 0.86% | 3,928,790.51 tokens/s | 42.68 GB/s/rank | 21.34 GB/s/rank |
| 16 ranks | FP8 | DeepEP V1 NVSHMEM | 1.0179 ms | [1.0167, 1.0203] ms | 0.18% | 2,011,977.59 tokens/s | 21.86 GB/s/rank | 10.93 GB/s/rank |
| 16 ranks | FP8 | DeepEP V2 NCCL GIN | 0.6219 ms | [0.6180, 0.6375] ms | 1.65% | 3,293,197.58 tokens/s | 35.78 GB/s/rank | 17.89 GB/s/rank |
| 16 ranks | BF16 | UCCL | 0.5920 ms | [0.5918, 0.5978] ms | 0.58% | 3,459,365.90 tokens/s | 49.59 GB/s/rank | 24.80 GB/s/rank |
| 16 ranks | BF16 | DeepEP V1 NVSHMEM | 1.0297 ms | [1.0251, 1.0304] ms | 0.28% | 1,988,936.56 tokens/s | 28.51 GB/s/rank | 14.26 GB/s/rank |
| 16 ranks | BF16 | DeepEP V2 NCCL GIN | **0.4850 ms** | [0.4826, 0.5043] ms | 2.42% | 4,222,332.23 tokens/s | 60.53 GB/s/rank | 30.27 GB/s/rank |
| 32 ranks | FP8 | UCCL | **0.7655 ms** | [0.7640, 0.7714] ms | 0.51% | 5,350,835.03 tokens/s | 29.07 GB/s/rank | 21.80 GB/s/rank |
| 32 ranks | FP8 | DeepEP V1 NVSHMEM | 1.5542 ms | [1.5513, 1.5628] ms | 0.38% | 2,635,371.65 tokens/s | 14.32 GB/s/rank | 10.74 GB/s/rank |
| 32 ranks | FP8 | DeepEP V2 NCCL GIN | 0.9537 ms | [0.9217, 0.9675] ms | 2.48% | 4,294,725.45 tokens/s | 23.33 GB/s/rank | 17.50 GB/s/rank |
| 32 ranks | BF16 | UCCL | **0.8686 ms** | [0.8683, 0.8696] ms | 0.08% | 4,715,590.93 tokens/s | 33.80 GB/s/rank | 25.35 GB/s/rank |
| 32 ranks | BF16 | DeepEP V1 NVSHMEM | 1.5589 ms | [1.5526, 1.5714] ms | 0.61% | 2,627,446.53 tokens/s | 18.83 GB/s/rank | 14.13 GB/s/rank |
| 32 ranks | BF16 | DeepEP V2 NCCL GIN | 0.9509 ms | [0.9482, 0.9606] ms | 0.68% | 4,307,661.17 tokens/s | 30.88 GB/s/rank | 23.16 GB/s/rank |

### Paired DeepEP V2 latency deltas

Positive values mean DeepEP V2 had lower latency than the named baseline. A directional winner is supported for this workload only when the paired 95% bootstrap interval excludes 0% and both arms have at most 5% run-to-run CV.

| EP size | Dispatch dtype | Baseline | Median V2 latency reduction | 95% bootstrap CI | Direction supported |
|---:|:---:|:---|---:|:---:|:---:|
| 16 ranks | FP8 | UCCL | -19.30% | [-22.84, -17.11]% | UCCL lower |
| 16 ranks | FP8 | DeepEP V1 NVSHMEM | 39.05% | [37.37, 39.22]% | V2 lower |
| 16 ranks | BF16 | UCCL | 18.44% | [14.82, 18.87]% | V2 lower |
| 16 ranks | BF16 | DeepEP V1 NVSHMEM | 52.68% | [51.06, 53.13]% | V2 lower |
| 32 ranks | FP8 | UCCL | -24.83% | [-25.41, -20.40]% | UCCL lower |
| 32 ranks | FP8 | DeepEP V1 NVSHMEM | 38.64% | [37.64, 41.02]% | V2 lower |
| 32 ranks | BF16 | UCCL | -9.35% | [-10.63, -9.16]% | UCCL lower |
| 32 ranks | BF16 | DeepEP V1 NVSHMEM | 38.87% | [38.76, 39.18]% | V2 lower |

The common logical throughput columns are the inverse-latency view of the same common payload and timing boundary, so they produce the same ranking as latency. Backend-native UCCL aggregate bandwidth, DeepEP V1 bandwidth, and DeepEP V2 SO/SU bandwidth use different numerators or aggregation boundaries and remain diagnostic only.

### Fair-rerun provenance and custody

The durable campaign ID is `fair-ep-b200-20260824t195145z`. Its machine-readable summary validates the complete 2-EP-size by 2-dtype by 3-backend by 3-start matrix, matching input and route hashes, digest-pinned images, and per-result correctness. The DeepEP V2 admission additionally required HMM disabled, `/dev/gdrdrv` present as a character device, and a logged GDAKI context before scoring.

The 4 selected nodes were Ready after the run, each with 8 allocatable GPUs and 8 allocatable EFA devices. The benchmark namespace and all campaign-labeled pods, Jobs, and MPIJobs were absent after teardown. The shared Lease still belonged to the concurrent vLLM campaign.

## Earlier backend-native campaign

The remaining sections preserve the earlier campaign's backend-native measurements. Their timing boundaries and byte numerators differ by backend, so the latency and GB/s columns must not be used for a direct cross-backend ranking. The numerically lower V2 values reported in that campaign remain observations of the native benchmark outputs, not evidence that V2 won a fair common-boundary comparison.

### Native campaign summary

Using each backend's native fields, the DeepEP V2 dispatch-plus-combine sum was numerically lower in all 4 B200 cells. Relative to the UCCL native aggregate, the differences were 19.04% for EP16 FP8, 44.70% for EP16 BF16, 6.82% for EP32 FP8, and 18.88% for EP32 BF16. Relative to the DeepEP V1 native aggregate, the differences ranged from 52.22% to 56.34%. Because the measurement boundaries differ, these percentages are retained as historical diagnostics and are not fair cross-backend speedups.

Each historical cell has 1 independent benchmark start, so run-to-run variance and coefficient of variation are unavailable. This campaign contains direct EP tests only; it produced no DeepEP V2 serving throughput, TTFT, TPOT, E2E latency, or NIXL result. DeepEP V2 also used substantially more HBM for prefill than the other 2 backends.

The synthetic DeepEP source contains PR 5, but EP16 and EP32 do not validate its greater-than-20-NVLink-domain scale path. The independent 256-rank PR 5 gate was not completed, and no PR 5 scale-success claim is inferred from these performance cells.

### Native campaign environment and provenance

| Field | Value |
|---|---|
| Measurement period | 23 August 2026 through 24 August 2026 |
| Region and Availability Zone | `ap-south-1`, `ap-south-1c` |
| EKS cluster | `ml-clusters-shared-ap-south-1` |
| Hardware | `p6-b200.48xlarge`, 8 B200 GPUs and 8 EFA devices per node |
| Fleet census | 36 Ready B200 nodes |
| EP16 topology | 2 nodes, 16 ranks |
| EP32 topology | 4 nodes, 32 ranks |
| Prefill shape | 4,096 tokens/rank, hidden size 7,168, top-k 8, experts 256 |
| Decode shape | 128 tokens/rank, hidden size 7,168, top-k 8, experts 256 |
| Selected operations | FP8 dispatch, BF16 dispatch, BF16 combine |
| Independent starts | 1 start per cell |
| Correctness | 12 of 12 backend cells passed |

All 3 scored images used the same vLLM wheel with SHA-256 `60800409bd2ede00aed65ac7b463541cbdc382716715e21c5bbef87d6f2423c4`. The common stack was vLLM commit `185cada36bb25aa55f762d004d54c5ca1e3fc753`, CUDA 13.0.3, PyTorch `2.13.0+cu130`, NCCL tag `v2.31.2-1` with runtime version 23,102 dimensionless, aws-ofi-nccl 1.21.1, libfabric API 2.6.0, and NIXL 1.3.2.

| Backend | Backend revision | Scored image digest |
|---|---|---|
| UCCL | `0dc87eb3b40c372a16b70ef320f37daaa5299ca7` | `sha256:d2e3e500524e168a3d577f059325df34872dde850bf77505c080daabfb9f8288` |
| DeepEP V1 NVSHMEM | DeepEP `567632dd59810d77b3cc05553df953cc0f779799`, NVSHMEM 3.7.0 | `sha256:175a8470e11bc7832024941d1775923d5e8ecae2c6dc06183d5f1653d66b4fac` |
| DeepEP V2 NCCL GIN | Synthetic commit `b56ebf8bb4ece24cd78aa8c12550b24e35ac255b` | `sha256:4d07367ea290c5d6ec3c02b223ac819feed5240a46fd4a6492421e9c0853dbeb` |

The DeepEP V2 source was synthesized reproducibly from base `02efc268a37802fc00812ede8f5ad7f535ceea0e`, PR 3 head `dd0f87261a80cf0ce8aa66e4ab2041843851d810`, and PR 5 head `2542d9641f2ec280213e875feb04be7862dda57c`. The final Git tree was `4da3a118b3b316de79b5af2b4770646035dc5802`. The image contained B200-specific `sm_100` and `sm_100a` code.

Before measurement, an INFO-level EP16 admission ran on both nodes with `NCCL_GIN_TYPE=5` and `NCCL_SYM_GIN_KERNELS_ENABLE=0`. Logs showed the aws-ofi-nccl `Libfabric_GDAKI` version 14 dimensionless plugin, GPU Direct RDMA enabled on EFA HCAs, nonzero GIN layouts, and balanced cluster EFA TX and RX deltas of 235,008 bytes each. Every scored cell also recorded nonzero EFA RDMA writes, so the cross-node path was measured rather than inferred from topology.

## Historical native decode diagnostics

These values come from backend-native reports, not the common-boundary harness. UCCL and DeepEP V1 print an aggregate dispatch-plus-combine latency. DeepEP V2 prints dispatch and combine separately, so the V2 value below is their sum for the same dispatch dtype. The boundaries differ, and the table does not support a direct latency ranking.

| EP ranks | Dispatch dtype | UCCL latency | DeepEP V1 latency | DeepEP V2 latency | V2 vs UCCL | V2 vs V1 |
|---:|---|---:|---:|---:|---:|---:|
| 16 ranks | FP8 | 475.540 us | 881.700 us | 384.994 us | 19.04 percent lower | 56.34 percent lower |
| 16 ranks | BF16 | 710.180 us | 886.920 us | 392.738 us | 44.70 percent lower | 55.72 percent lower |
| 32 ranks | FP8 | 693.840 us | 1,403.720 us | 646.522 us | 6.82 percent lower | 53.94 percent lower |
| 32 ranks | BF16 | 844.520 us | 1,433.780 us | 685.084 us | 18.88 percent lower | 52.22 percent lower |

In the native outputs, UCCL had the smallest numerical latency increase when scaling from EP16 to EP32, while the DeepEP V2 sum remained numerically smallest. For FP8, the increases were 45.91% for UCCL, 59.21% for DeepEP V1, and 67.93% for DeepEP V2. This is a single-start diagnostic observation across different timing boundaries, not a scaling-efficiency claim.

The native decode bandwidth reports are included for completeness. UCCL and DeepEP V1 report one dispatch-plus-combine bandwidth; DeepEP V2 reports separate SO legs. The V2 dispatch/combine pair is not an aggregate and is not directly comparable to the first 2 columns.

| EP ranks | Dispatch dtype | UCCL aggregate | DeepEP V1 aggregate | DeepEP V2 dispatch / combine SO |
|---:|---|---:|---:|---:|
| 16 ranks | FP8 | 46.37 GB/s | 25.01 GB/s | 5 GB/s / 9 GB/s |
| 16 ranks | BF16 | 40.94 GB/s | 32.78 GB/s | 9 GB/s / 9 GB/s |
| 32 ranks | FP8 | 31.78 GB/s | 15.71 GB/s | 10 GB/s / 14 GB/s |
| 32 ranks | BF16 | 34.43 GB/s | 20.28 GB/s | 16 GB/s / 14 GB/s |

## Historical native prefill diagnostics

The table reports each backend's native latency and cross-node bandwidth fields. For UCCL, latency is `transmit + notify` and bandwidth is the RDMA leg. For DeepEP V1, latency is the sum of the 2 printed components and bandwidth is the RDMA leg. For DeepEP V2, latency is the operation latency and bandwidth is the scale-out, or SO, leg. These measurement boundaries and bandwidth accounting conventions differ, so the table is directional. Do not rank the backends by comparing the GB/s columns alone.

| EP ranks | Operation | UCCL latency / RDMA bandwidth | DeepEP V1 latency / RDMA bandwidth | DeepEP V2 latency / SO bandwidth |
|---:|---|---:|---:|---:|
| 16 ranks | FP8 dispatch | 1,161.120 us / 55.59 GB/s | 1,014.000 us / 65.95 GB/s | 864.688 us / 35 GB/s |
| 16 ranks | BF16 dispatch | 1,637.970 us / 74.42 GB/s | 1,639.000 us / 78.14 GB/s | 1,501.000 us / 39 GB/s |
| 16 ranks | BF16 combine | 2,096.580 us / 58.81 GB/s | 1,633.220 us / 78.62 GB/s | 1,615.000 us / 36 GB/s |
| 32 ranks | FP8 dispatch | 2,229.950 us / 51.38 GB/s | 2,369.000 us / 50.95 GB/s | 2,041.000 us / 41 GB/s |
| 32 ranks | BF16 dispatch | 3,852.970 us / 56.69 GB/s | 4,125.000 us / 56.94 GB/s | 3,753.000 us / 43 GB/s |
| 32 ranks | BF16 combine | 3,950.790 us / 55.52 GB/s | 4,106.520 us / 55.80 GB/s | 3,745.000 us / 43 GB/s |

## Historical DeepEP V2 operation diagnostics

SO is scale-out bandwidth over the internode path and SU is scale-up bandwidth over the intranode path. FP8 combine is included because the selected V2 test prints it, but the matched headline operation set uses BF16 combine.

| Topology and workload | Dispatch dtype | Operation | SO bandwidth | SU bandwidth | Latency | Payload |
|---|---|---|---:|---:|---:|---:|
| EP16 decode | FP8 | dispatch | 5 GB/s | 27 GB/s | 185.637 us | 4,972,032 bytes |
| EP16 decode | FP8 | combine | 9 GB/s | 48 GB/s | 199.357 us | 9,540,352 bytes |
| EP16 decode | BF16 | dispatch | 9 GB/s | 49 GB/s | 195.257 us | 9,582,848 bytes |
| EP16 decode | BF16 | combine | 9 GB/s | 48 GB/s | 197.481 us | 9,540,352 bytes |
| EP32 decode | FP8 | dispatch | 10 GB/s | 19 GB/s | 275.659 us | 5,106,816 bytes |
| EP32 decode | FP8 | combine | 14 GB/s | 26 GB/s | 370.863 us | 9,798,976 bytes |
| EP32 decode | BF16 | dispatch | 16 GB/s | 31 GB/s | 314.156 us | 9,842,624 bytes |
| EP32 decode | BF16 | combine | 14 GB/s | 26 GB/s | 370.928 us | 9,798,976 bytes |
| EP16 prefill | FP8 | dispatch | 35 GB/s | 203 GB/s | 864.688 us | 175,106,880 bytes |
| EP16 prefill | FP8 | combine | 36 GB/s | 208 GB/s | 1,614.000 us | 335,995,680 bytes |
| EP16 prefill | BF16 | dispatch | 39 GB/s | 225 GB/s | 1,501.000 us | 337,492,320 bytes |
| EP16 prefill | BF16 | combine | 36 GB/s | 208 GB/s | 1,615.000 us | 335,995,680 bytes |
| EP32 prefill | FP8 | dispatch | 41 GB/s | 95 GB/s | 2,041.000 us | 194,860,224 bytes |
| EP32 prefill | FP8 | combine | 43 GB/s | 100 GB/s | 3,749.000 us | 373,898,464 bytes |
| EP32 prefill | BF16 | dispatch | 43 GB/s | 100 GB/s | 3,753.000 us | 375,563,936 bytes |
| EP32 prefill | BF16 | combine | 43 GB/s | 100 GB/s | 3,745.000 us | 373,898,464 bytes |

The DeepEP V2 EFA RDMA write deltas were 19,708,762,240 bytes for EP16 decode, 111,413,870,476 bytes for EP32 decode, 507,290,310,848 bytes for EP16 prefill, and 2,742,751,387,856 bytes for EP32 prefill.

## Historical native memory tradeoff

All 12 cells reached 100 percent sampled peak GPU utilization. Peak HBM use was materially higher for DeepEP V2 prefill.

| Backend | EP16 decode | EP32 decode | EP16 prefill | EP32 prefill |
|---|---:|---:|---:|---:|
| UCCL | 9,627 MiB/GPU | 9,501 MiB/GPU | **13,697 MiB/GPU** | **20,735 MiB/GPU** |
| DeepEP V1 NVSHMEM | 10,597 MiB/GPU | 11,239 MiB/GPU | 17,047 MiB/GPU | 23,793 MiB/GPU |
| DeepEP V2 NCCL GIN | **7,041 MiB/GPU** | **7,271 MiB/GPU** | 39,657 MiB/GPU | 44,315 MiB/GPU |

## Separate EFA 3.3.0g revalidation

On 24 August 2026, a separate revalidation updated the current 32-node B200 fleet to the EFA 3.3.0g kernel module and rdma-core 64 on kernel `6.12.100-125.179.amzn2023.x86_64`. The post-update audit passed on 32 of 32 nodes. The first targeted DeepEP V2 admission then triggered a kernel panic in the Linux device-memory mapping path called by NVIDIA UVM, with the first preserved frame at `__init_zone_device_page`. The remaining runs were stopped to avoid risking more nodes.

That trace does not establish EFA 3.3.0g as the root cause. The historical native tables are from the earlier B200 campaign and are not presented as post-update measurements. The fair common-boundary rerun used the post-update hosts only after the required HMM and GDRCopy-device admission checks passed. The failed revalidation's benchmark resources were harvested and removed; all 32 nodes returned Ready.

## Reproduce and collate

The standalone DeepEP V2 build and launch harness is being added in [PR 1234](https://github.com/awslabs/awsome-distributed-ai/pull/1234). Pin the image and source revisions recorded above, use `NCCL_GIN_TYPE=5` and `NCCL_SYM_GIN_KERNELS_ENABLE=0`, and first run an INFO-level admission that proves the `Libfabric_GDAKI` context and nonzero GIN layout. Return `NCCL_DEBUG` to `WARN` for measurement.

The collector accepts rank-zero DeepEP V2 logs from `tests/elastic/test_ep.py`:

```bash
python3 collect_results.py \
    --nvshmem-internode nvshmem_prefill.log \
    --nvshmem-lowlat nvshmem_decode.log \
    --uccl-internode uccl_prefill.log \
    --uccl-lowlat uccl_decode.log \
    --deepep-v2-prefill deepep_v2_prefill.log \
    --deepep-v2-decode deepep_v2_decode.log \
    --nccl nccl_alltoall.log
```

Preserve every rank's log, rendered launch manifest, immutable image reference, command line, correctness result, and before-and-after EFA counters. `collect_results.py` summarizes rank zero, but correctness and path validation must cover every rank and every node.
