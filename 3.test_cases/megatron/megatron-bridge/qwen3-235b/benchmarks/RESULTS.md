<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Qwen3-235B-A22B — NCCL vs DeepEP+UCCL vs DeepEP+NVSHMEM — Benchmark Results

Three-way MoE token-dispatcher comparison on **8× p6-b300 (64× B300)**: NCCL all-to-all
(baseline) vs DeepEP over **UCCL** (EFA-native) vs DeepEP over **NVSHMEM-libfabric** (NVIDIA
DeepEP, EFA-patched). The only toggle that changes between arms is the dispatcher / image;
everything else (Qwen3-235B-A22B 128-expert recipe, mock data, seq 4096, global batch,
bf16, parallelism, seed, EFA env, `MOE_A2A_OVERLAP`) is held fixed within a run. Metric of
record: **mean training-iteration time** over the steady state (first 4 iters dropped),
plus analytical `MODEL TFLOP/s/GPU` and derived `tok/s = global_batch × seq_len / iter_time`.

| Field | Value |
|---|---|
| Date | _TBD_ |
| Hardware | 8 × `p6-b300.48xlarge` (B300, 8 GPU + 16 EFA / node), EKS `ml-shared` (us-west-2) |
| Model | Qwen3-235B-A22B — 128 experts, top-8, 94 layers, hidden 4096, MoE FFN 1536, bf16 |
| World | 64 ranks (8 nodes × 8 GPU), TP8, ETP1, balanced routing |
| EP sweep | EP16 (PP4/DP2) · EP32 (PP2/DP4) |
| UCCL image | `megatron-bridge-uccl:nemo-26.04.01-uccl-0dc87eb` (UCCL `0dc87eb`) |
| NVSHMEM image | `megatron-bridge-uccl:nemo-26.04.01-deepep-nvshmem-567632d-cu13` (DeepEP `567632d`, NVSHMEM v3.7.0-0, libfabric/EFA) |
| Bridge / Core | megatron-bridge 0.4.2 / megatron-core 0.17.1 (nemo:26.04.01) |

> Status: harness + both images validated end-to-end on a 2-node smoke (NUM_LAYERS=8);
> full 64-GPU campaign numbers below are _TBD_ pending the run.

## Measured result — mean iter time (lower = better), overlap=off (dispatcher-isolation)

`delta` columns are vs the NCCL all-to-all baseline (negative = DeepEP faster).

### EP32 (TP8/PP2/DP4)

| micro_batch | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM | UCCL Δ | NVSHMEM Δ |
|------------:|----------------:|------------:|---------------:|-------:|----------:|
| 1 | _TBD_ s | _TBD_ s | _TBD_ s | _TBD_ | _TBD_ |
| 4 | _TBD_ s | _TBD_ s | _TBD_ s | _TBD_ | _TBD_ |

### EP16 (TP8/PP4/DP2)

| micro_batch | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM | UCCL Δ | NVSHMEM Δ |
|------------:|----------------:|------------:|---------------:|-------:|----------:|
| 1 | _TBD_ s | _TBD_ s | _TBD_ s | _TBD_ | _TBD_ |
| 4 | _TBD_ s | _TBD_ s | _TBD_ s | _TBD_ | _TBD_ |

## Deployment-realistic overlap=on (mb=4) — separate within-regime A/B

`overlap=on` enables `overlap_moe_expert_parallel_comm` (1F1B hides the EP all-to-all). On
Core 0.17.1 with PP>1 this forces VPP=2 + recompute off, and Qwen3's 94 layers are rounded
to **96** for VPP divisibility — so these numbers are an independent within-regime A/B, **not
comparable cell-for-cell to overlap=off** (do not subtract across regimes).

| EP | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM | UCCL Δ | NVSHMEM Δ |
|---:|----------------:|------------:|---------------:|-------:|----------:|
| 16 | _TBD_ s | _TBD_ s | _TBD_ s | _TBD_ | _TBD_ |
| 32 | _TBD_ s | _TBD_ s | _TBD_ s | _TBD_ | _TBD_ |

## TFLOP/s/GPU and tok/s (steady state)

| arm | EP | mb | overlap | mean iter (s) | MODEL TFLOP/s/GPU | tok/s | transport |
|-----|---:|---:|---------|--------------:|------------------:|------:|-----------|
| _TBD (filled from bench/last-campaign-index.csv)_ | | | | | | | |

## Validity gates (every run)

- **EFA active on all 64 ranks**: each rank logged `NET/OFI Selected provider is efa`
  (true EFA RDMA, no socket fallback). _TBD: confirmed N/N._
- **Transport-active per arm**: UCCL arm logs the UCCL-EP proxy banner; NVSHMEM arm logs
  NVSHMEM-over-libfabric init. _TBD._
- **Work-equivalence (no token dropping)**: `LOSS_PROBE=1` iteration-1 loss of the DeepEP
  arms matches the NCCL `alltoall` arm to ~5 significant figures (bf16 round-off only).
  _TBD: alltoall=____, deepep-uccl=____, deepep-nvshmem=____._

## Caveats

1. **Mock data + random-init + forced balancing** — measures the dispatcher on a balanced
   token distribution (step time, not loss); same regime as the DSV3/Kimi-K2 A/Bs.
2. **overlap=on uses 96 layers** (94 rounded up for VPP). Within-regime only.
3. **NVSHMEM DeepEP = NVIDIA DeepEP `567632d`** patched for EFA (NVSHMEM-libfabric host-proxy,
   IBGDA off) — see [`../../deepep/`](../../deepep/). Not DeepEP's IB/IBGDA fast path.
4. Single run per cell (within-run σ reported by `parse-runs.py`; between-run variance not bounded).

## How to reproduce

See [`../README.md`](../README.md). One command:
`CTX=… UCCL_IMG=… NVSHMEM_IMG=… bash run-qwen3-campaign.sh`, then
`python3 ../bench/parse-runs.py <campaign_dir> --warmup 4`.
