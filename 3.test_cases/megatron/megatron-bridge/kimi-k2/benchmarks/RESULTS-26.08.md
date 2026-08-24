# Kimi-K2 Megatron-Bridge EP results, NeMo 26.08

Campaign ID: `20260823T223611Z-kimi-k2-megatron-ep-2608`

Status timestamp: `2026-08-24T00:58:05Z`

The 256-GPU B300 headline is `NOT_RUN_INSUFFICIENT_CAPACITY`. The live ap-south-1 cluster had 0 B300 nodes and 36 p6-b200.48xlarge nodes. The only Capacity Block was 36 p6-b200.48xlarge instances in ap-south-1c, ending at `2026-08-24T11:30:00Z`. The concurrent vLLM campaign retained the active shared lease through the reclaim window, so this campaign did not claim any B200 node protected by that lease.

## Headline matrix

| Arm | throughput-no-overlap, MB4, 256 GPUs | Reason |
|---|---:|---|
| `nccl-alltoall` | `NOT_RUN_INSUFFICIENT_CAPACITY` | 0 of 32 required B300 nodes existed |
| `uccl` | `NOT_RUN_INSUFFICIENT_CAPACITY` | 0 of 32 required B300 nodes existed |
| `deepep-v1-nvshmem` | `NOT_RUN_INSUFFICIENT_CAPACITY` | 0 of 32 required B300 nodes existed |
| `deepep-v2-gin-gda` | `NOT_RUN_INSUFFICIENT_CAPACITY` | 0 of 32 required B300 nodes existed |

No 26.04 value is copied into this table.

## Image and stack qualification

The exact supplied base digest was pulled and probed. It contains Bridge 0.6.0 at `c93251151adeeadbae3ff2a2bf5ee7a1c34cff01`, MCore 0.19.0 at `16ad357ee7973af32916fc1ca39d71065e5f03d4`, Torch `2.13.0a0+8145d630e8.nv26.06`, TransformerEngine `2.17.1+4329ff84`, CUDA toolkit 13.3.33, NCCL build 2.30.5, NCCL runtime 2.30.7, libfabric 2.4.0amzn3.0, and GDRCopy 2.5.

The common replacement image built successfully as local image ID `sha256:97df9472170b90cd23c27511416dda056c925bd6231b0a0b611c32ac2670746a`. It has NCCL 2.31.2 at `/opt/nccl/build/lib/libnccl.so.2.31.2`, NCCL SHA-256 `28bf309dbc620a1bce54c33ad21e69623f555209d537c099a1d7f015728d3051`, aws-efa-installer 1.50.0, libfabric 2.6.0amzn1.0, aws-ofi-nccl 1.21.1, and GDRCopy v2.5.2. The process map contained one resolved NCCL library.

Image acceptance failed because `torch.cuda.nccl.version()` remained 2.30.5 while `ncclGetVersion()` returned 2.31.2. The exact Torch source commit was not shipped in the image, `torch.version.git_version` returned `Unknown`, and public PyTorch fetch of `8145d630e8` returned `fatal: couldn't find remote ref`. Rebuilding from another Torch revision would change the common training stack and invalidate the requested isolation. The final targets therefore were not tagged or pushed, and no image digest is claimed.

## Correctness and performance status

Desk gates passed for patch application against the exact Bridge and MCore commits, Python compilation, MCore `deepep_v2` config construction, `_DeepepV2Manager` inheritance, a mocked ElasticBuffer forward/backward contract, shell syntax, result-schema parsing, and deterministic DeepEP patch hashes. These are implementation checks, not GPU correctness results.

Gates A through D are `NOT_RUN_STACK_IDENTITY_GATE`. A real ElasticBuffer/GDAKI run would violate the required NCCL build/runtime identity gate. The strongest smaller B200 qualification is also `NOT_RUN_CONCURRENT_CAMPAIGN`: the shared lease holder was `dsv3-ep-backend-comparison-20260823`, with lease duration 57,600 seconds from `2026-08-23T20:36:29Z`, and its four named nodes and all resources remained untouched.

The PR #5 scale matrix is `NOT_RUN_INSUFFICIENT_CAPACITY`: the cluster had 0 B300 physical domains, below the required 22-domain, 23-domain, and 32-domain cells. No scale verdict is inferred from the normal Kimi topology.

## Raw provenance

Durable artifacts are under:

`/mnt/fsx/ubuntu/workspace/artifacts/adai-kimi-k2-megatron-ep/20260823T223611Z-kimi-k2-megatron-ep-2608`

Key files are `build/common-build.log`, `build/common-build-manifest.json`, `build/nccl-identity-gate.log`, `provenance/nemo2608-probe.txt`, `provenance/torch-source-fetch.log`, `qualification/desk-gates.log`, and the timestamped directories under `census/`. Machine-readable terminal outcomes are under `status/`. Each artifact directory carries SHA-256 custody files.

## Teardown and custody

The terminal live census found 0 resources owned by this campaign and no campaign namespace. This campaign never acquired or mutated the shared lease. At the same timestamp, the lease holder remained `dsv3-ep-backend-comparison-20260823`, and its 4 running GPU pods remained on its 4 named B200 nodes. No teardown action was needed. The machine-readable report is `teardown/custody.json` under the artifact root.
