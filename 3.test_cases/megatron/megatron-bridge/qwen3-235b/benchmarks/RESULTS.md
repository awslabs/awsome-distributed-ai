<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Qwen3-235B-A22B — NCCL vs DeepEP+UCCL vs DeepEP+NVSHMEM — Benchmark Results

Three-way MoE token-dispatcher comparison on **`p6-b300`**: NCCL all-to-all (baseline) vs
DeepEP over **UCCL** (EFA-native) vs DeepEP over **NVSHMEM-libfabric** (NVIDIA DeepEP,
EFA-patched). The only toggle that changes between arms is the dispatcher / image; everything
else (Qwen3-235B-A22B 128-expert recipe, mock data, seq 4096, global batch, bf16, parallelism,
seed, EFA env, recompute) is held fixed within a run. Metric of record: **mean
training-iteration time** over the steady state (first 4 of 12 iters dropped), plus analytical
`MODEL TFLOP/s/GPU` and derived `tok/s = global_batch × seq_len / iter_time`.

| Field | Value |
|---|---|
| Date | 2026-06-21 |
| Hardware | **4 × `p6-b300.48xlarge`** (32× B300, 8 GPU + 16 EFA / node), EKS `ml-shared` (us-west-2) |
| Model | Qwen3-235B-A22B — 128 experts, top-8, 94 layers, hidden 4096, MoE FFN 1536, bf16 |
| World | 32 ranks, TP8, ETP1, balanced routing, `RECOMPUTE=full`, overlap=off |
| EP sweep | EP16 (PP2/DP2) · EP32 (PP1/DP4) — `PP = WORLD/EP`, WORLD=32 |
| Global batch | 128 (12 iters/run, 8 steady) |
| UCCL image | `megatron-bridge-uccl:nemo-26.04.01-uccl-0dc87eb` (UCCL `0dc87eb`) |
| NVSHMEM image | `megatron-bridge-uccl:nemo-26.04.01-deepep-nvshmem-567632d-cu13` (DeepEP `567632d`, NVSHMEM v3.7.0-0, libfabric/EFA) |
| Bridge / Core | megatron-bridge 0.4.2 / megatron-core 0.17.1 (nemo:26.04.01) |
| Campaign | `qwen3final-20260621T0215Z` (raw `index.csv`: [`last-campaign-index.csv`](last-campaign-index.csv)) |

> **Why 4 nodes, not 8.** The plan targeted 8× p6-b300 (EP32 at TP8/PP2/DP4). At run time
> the shared `ml-shared` cluster had only 6 of its 8 B300 nodes free (2 held by another
> team), and EP32 cannot tile 6 nodes with integer PP. The whole sweep was therefore run on
> **4 nodes (32 GPU)**, where both EP16 (PP4/DP2) and EP32 (PP1/DP4) tile cleanly and the
> EP degree — the variable the all-to-all turns on — is preserved. EP32 at PP1 needs
> `RECOMPUTE=full` to fit; that recompute is applied to **all** arms and **both** EPs so the
> per-cell 3-way comparison stays clean (see caveats).

## Measured result — mean iter time (lower = better), overlap=off, recompute=full

`Δ` = vs the NCCL all-to-all baseline (negative = DeepEP faster). Each cell: mean iter s ·
MODEL TFLOP/s/GPU · derived tok/s.

### EP16 (TP8/PP2/DP2) — internode EP across 2 nodes

| mb | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM | UCCL Δ | NVSHMEM Δ |
|---:|----------------:|------------:|---------------:|-------:|----------:|
| 4 | 15.18 s · 159.8 · 34.5k | 17.85 s · 135.9 · 29.4k | **14.91 s · 162.7 · 35.2k** | +17.6% | **−1.8%** |
| 1 | **46.81 s · 51.8 · 11.2k** | 56.00 s · 43.5 · 9.4k | 50.09 s · 48.4 · 10.5k | +19.6% | +7.0% |

### EP32 (TP8/PP1/DP4) — internode EP across all 4 nodes

| mb | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM | UCCL Δ | NVSHMEM Δ |
|---:|----------------:|------------:|---------------:|-------:|----------:|
| 4 | 22.20 s · 109.3 · 23.6k | **20.96 s · 117.2 · 25.0k** | 23.76 s · 103.1 · 22.1k | **−5.6%** | +7.1% |
| 1 | **45.45 s · 53.4 · 11.5k** | 57.94 s · 42.2 · 9.0k | 55.35 s · 44.2 · 9.5k | +27.5% | +21.8% |

## What the numbers say

1. **At the efficient operating point (mb=4) the three dispatchers are within ~±18%, and the
   winner depends on EP degree.** At **EP16**, DeepEP+NVSHMEM is fastest (−1.8% vs NCCL) and
   UCCL is slowest (+17.6%). At **EP32** the order flips among the DeepEP backends: **UCCL is
   fastest (−5.6% vs NCCL)** while NVSHMEM falls behind (+7.1%). UCCL degrades least going
   EP16→EP32 (17.85→20.96 s), whereas NCCL and NVSHMEM degrade more (15.18→22.20 and
   14.91→23.76) — UCCL scales to higher EP best here, NVSHMEM worst.
2. **At mb=1 NCCL all-to-all wins at both EP degrees** (+7–28% over the DeepEP arms). This is
   the smallest per-dispatch granularity (64/EP16 and 32/EP32 microbatches per iter), where
   DeepEP's per-dispatch proxy/kernel-launch overhead is unamortized — the same crossover the
   DSV3 A/B found. A throughput-tuned run uses mb≥4.
3. **The deltas are far smaller than the DSV3 256-GPU result (−36% for UCCL at mb4).** The
   dominant reason is `RECOMPUTE=full` (forced to fit EP32 on 4 nodes): recomputation roughly
   doubles compute per step, so the all-to-all becomes a smaller fraction of the iteration and
   any dispatcher delta is compressed. These numbers are therefore a **conservative,
   recompute-on** comparison, not the dispatcher-isolation upper bound.

## Validity gates (every run)

- **EFA active on all 32 ranks**: every run logged `NET/OFI Selected provider is efa`
  (`efa_ok=True`, 4/4 node logs) — true EFA RDMA, no socket fallback. ✓ all 12 runs.
- **Transport-active per arm**: UCCL arms logged the UCCL-EP proxy banner (`uccl_ok=True`);
  NVSHMEM arms logged `NVSHMEM v3.7.0` init (`nvshmem_ok=True`); NCCL arms neither. ✓
- **No stalls**: 0/8 steady iters > 3× median in every run (balanced routing held).
- **Work-equivalence (no token dropping)** — verified on a 2-node smoke (NUM_LAYERS=8) with
  `LOSS_PROBE=1`: iteration-1 mean loss **deepep-nvshmem 12.701515 vs alltoall 12.702237**
  (identical num_tokens=2048; relative diff 5.7e-5 = bf16 round-off). The NVSHMEM arm's
  IBGDA→host-proxy + put_signal patches dispatch/combine the same tokens to numerically
  equivalent output as NCCL.

## Caveats

1. **`RECOMPUTE=full` on all arms** compresses the dispatcher delta (see point 3). To measure
   the dispatcher-isolation upper bound, re-run with recompute off — which needs ≥8 nodes for
   EP32 (more pipeline stages) to fit without it.
2. **4 nodes / 32 GPU, not 8** — shared-cluster capacity at run time (see the note above). The
   EP degree (16/32) is preserved; absolute scale is half the plan.
3. **EP16 uses PP2, EP32 uses PP1** — within each EP the three arms are identical (clean 3-way),
   but the cross-EP comparison also changes PP, so read EP16-vs-EP32 as indicative, not isolated.
4. **`overlap=on` not run** here (it requires recompute off → OOM at PP1 on 4 nodes, plus
   Qwen3's 94 layers need a 96-layer VPP round-up). Deferred to an 8-node run.
5. **NVSHMEM arm exits non-zero (1) at process teardown** (NVSHMEM finalize after
   `[after training is done]`); all 8 steady training iters are recorded and valid (tight
   median≈mean), so the measurement is unaffected. STATUS shows `exit=1` for those runs.
6. **Mock data + random-init + forced balancing** — measures the dispatcher on a balanced
   token distribution (step time, not loss), same regime as the DSV3/Kimi-K2 A/Bs.
7. Single run per cell (within-run σ small; between-run variance not bounded).

## How to reproduce

See [`../README.md`](../README.md). The exact campaign:

```bash
CTX=ml-shared NS=kimi-k2-bench NNODES=4 EPS="16 32" CELLS="4:off 1:off" RECOMPUTE=full \
TRAIN_ITERS=12 GLOBAL_BATCH=128 \
UCCL_IMG=<…>:nemo-26.04.01-uccl-0dc87eb \
NVSHMEM_IMG=<…>:nemo-26.04.01-deepep-nvshmem-567632d-cu13 \
  bash run-qwen3-campaign.sh
# then: python3 ../bench/parse-runs.py <campaign_dir> --warmup 4
```

**Confounder guard:** assert `NET/OFI Selected provider is efa` appears for every rank of
every run (the parser's `efa_ok`); discard and re-run any run where it does not.
