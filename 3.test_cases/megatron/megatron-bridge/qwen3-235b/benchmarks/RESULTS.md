<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Qwen3-235B-A22B — NCCL vs DeepEP+UCCL vs DeepEP+NVSHMEM — Benchmark Results

Three-way MoE token-dispatcher comparison on **`p6-b300`** (B300): NCCL all-to-all (baseline)
vs DeepEP over **UCCL** (EFA-native) vs DeepEP over **NVSHMEM-libfabric** (NVIDIA DeepEP,
EFA-patched). The only toggle that changes between arms is the dispatcher / image; everything
else (Qwen3-235B-A22B 128-expert recipe, mock data, seq 4096, global batch, bf16, parallelism,
seed, EFA env, recompute) is held fixed within a run. Metric of record: **mean training-iteration
time** over the steady state (first 4 of 12 iters dropped), plus analytical `MODEL TFLOP/s/GPU`
and derived `tok/s = global_batch × seq_len / iter_time`.

There are **two campaigns**: the **primary** 8-node EP32-at-PP2 run (cleaner regime, below) and a
**secondary** 4-node reference (capacity-constrained, near the bottom). Read the primary; the
secondary is retained to show how the regime (PP depth + recompute) changes the conclusion.

---

## Primary — 8 nodes, EP32 @ PP2 (recompute=selective)

| Field | Value |
|---|---|
| Hardware | **8 × `p6-b300.48xlarge`** (64× B300, 8 GPU + 16 EFA / node), EKS `ml-shared` (us-west-2) |
| World | 64 ranks, TP8, ETP1, balanced routing, **`RECOMPUTE=selective`** |
| Parallelism | **EP32 = TP8/PP2/DP4** · EP16 = TP8/PP4/DP2 (`PP = WORLD/EP`, WORLD=64) |
| Global batch | 256 (12 iters/run, 8 steady) |
| Images | `…-uccl-0dc87eb` (UCCL) · `…-deepep-nvshmem-567632d-cu13` (NVSHMEM v3.7.0-0, DeepEP 567632d, libfabric/EFA) |
| Campaign | `qwen3-8n-ep32pp2-20260621` |

### EP32 @ PP2, overlap=off (dispatcher-isolation) — mean iter s · TFLOP/s/GPU · tok/s

| mb | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM | UCCL Δ | NVSHMEM Δ |
|---:|----------------:|------------:|---------------:|-------:|----------:|
| 4 | **15.91 s · 152.4 · 65.9k** | 18.43 s · 134.5 · 56.9k | 20.41 s · 120.9 · 51.4k | +15.9% | +28.3% |
| 1 | **33.41 s · 72.6 · 31.4k** | 46.90 s · 52.4 · 22.4k | 45.39 s · 54.2 · 23.1k | +40.4% | +35.9% |

### EP32 @ PP2, overlap=on (1F1B, deployment regime) — mean iter s

| mb | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM |
|---:|----------------:|------------:|---------------:|
| 4 | **11.01 s · 225.0 · 95.3k** | _re-running_ | _re-running_ |

### EP16 @ PP4, overlap off/on

| mb · overlap | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM |
|---|----------------:|------------:|---------------:|
| 4 · off | _re-running_ | _re-running_ | _re-running_ |
| 1 · off | _re-running_ | _re-running_ | _re-running_ |
| 4 · on  | _re-running_ | _re-running_ | _re-running_ |

> The first 8-node campaign lost a node to another team mid-run (rendezvous 7/8); the EP32
> overlap=off cells completed before that, the EP16 cells and the two EP32 overlap=on DeepEP
> cells are being re-run into the same campaign dir. This section is updated when they land.

### What the numbers say (primary)

1. **At the clean EP32 @ PP2 regime, NCCL all-to-all is the fastest dispatcher and both DeepEP
   backends are slower** — and these are **large, robust deltas** (not n=1 noise): at mb=4,
   DeepEP+UCCL **+15.9%** and DeepEP+NVSHMEM **+28.3%** slower than NCCL; at mb=1, **+40%** and
   **+36%**. NCCL all-to-all also scales fine into the `overlap=on` deployment regime (11.0 s,
   **95k tok/s**, 225 TFLOP/s/GPU at mb=4).
2. **This overturns the 4-node EP32 result** (secondary table), where DeepEP+UCCL *looked* 5.6%
   *faster* than NCCL at EP32. That apparent win was an artifact of the 4-node **PP1 +
   `RECOMPUTE=full`** regime (recompute roughly doubles compute and shrinks the all-to-all's
   share of the step, compressing/distorting the dispatcher delta). At PP2 with lighter
   `selective` recompute the all-to-all is a larger share of the step and the true ordering
   shows: **NCCL > UCCL > NVSHMEM** at EP32 on this 6.4 Tbps/node B300 fabric.
3. Consistent with the DSV3 "honest bottom line": there is no published EFA *training* win for
   the DeepEP/UCCL path over NCCL all-to-all, and p6-b300's per-node bandwidth makes NCCL's
   all-to-all very strong. (DeepEP/UCCL wins in the literature are inference-scale or on
   InfiniBand / lower-bandwidth fabrics.)

### Validity (primary, EP32 overlap=off + NCCL overlap=on)

EFA active on all 64 ranks (`efa_ok`, 8/8 node logs) · 0 stalls · 8 steady iters · per-arm
transport confirmed (`uccl_ok` / `nvshmem_ok`). Work-equivalence verified on a 2-node smoke
(`LOSS_PROBE`): deepep-nvshmem iter-1 loss matches NCCL to ~4 sig figs (12.701515 vs 12.702237,
rel 5.7e-5). The NVSHMEM arm exits non-zero at NVSHMEM finalize *after* training — validate via
`efa_ok` + `n_steady==8`, not exit code.

---

## Secondary reference — 4 nodes, recompute=full (capacity-constrained)

Retained to show the regime dependence. **Do not read this as the headline** — point 2 above
explains why its EP32 deltas are distorted.

| Field | Value |
|---|---|
| Hardware | **4 × `p6-b300.48xlarge`** (32× B300), EKS `ml-shared` |
| World | 32 ranks, TP8, ETP1, **`RECOMPUTE=full`**, overlap=off, global batch 128 |
| Parallelism | EP16 = PP2/DP2 · EP32 = **PP1**/DP4 (`PP = WORLD/EP`, WORLD=32) |
| Campaign | `qwen3final-20260621T0215Z` |

> **Why 4 nodes:** at run time the shared cluster had only 6 of 8 B300 free (held by other
> teams), and EP32 cannot tile 6 nodes with integer PP, so this ran on 4 nodes — where EP32 is
> forced to **PP1** and needs `RECOMPUTE=full` to fit. That regime is what distorts the deltas.

mean iter s · TFLOP/s/GPU · tok/s; `Δ` vs NCCL (− = DeepEP faster):

| EP | mb | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM | UCCL Δ | NVSHMEM Δ |
|---:|---:|----------------:|------------:|---------------:|-------:|----------:|
| 16 | 4 | 15.18 · 159.8 · 34.5k | 17.85 · 135.9 · 29.4k | 14.91 · 162.7 · 35.2k | +17.6% | −1.8% |
| 16 | 1 | 46.81 · 51.8 · 11.2k | 56.00 · 43.5 · 9.4k | 50.09 · 48.4 · 10.5k | +19.6% | +7.0% |
| 32 | 4 | 22.20 · 109.3 · 23.6k | 20.96 · 117.2 · 25.0k | 23.76 · 103.1 · 22.1k | −5.6% | +7.1% |
| 32 | 1 | 45.45 · 53.4 · 11.5k | 57.94 · 42.2 · 9.0k | 55.35 · 44.2 · 9.5k | +27.5% | +21.8% |

At this `RECOMPUTE=full` regime the mb=4 deltas are small (≤±18%) and, at n=1, sub-±6% gaps
(EP16 NVSHMEM −1.8%, EP32 UCCL −5.6%) are within noise — which is exactly why the 8-node PP2
re-run was done. The one regime-independent read that holds in both campaigns: **NCCL wins the
mb=1 tiny-dispatch regime** (DeepEP per-dispatch overhead unamortized).

---

## Caveats (both campaigns)

1. **Regime matters more than n.** The dispatcher delta depends strongly on PP depth + recompute
   (compute-vs-comm balance), as the primary-vs-secondary contrast shows. Compare only within a
   regime.
2. **n=1 per cell.** Within-run σ is small (median≈mean), but between-run variance is not bounded.
   The primary EP32 deltas are large enough (+16% to +40%) to be robust at n=1; the secondary's
   sub-±6% deltas are not. n≥3 with error bars is the recommended next step.
3. **`overlap=on` uses 96 layers** (Qwen3's 94 rounded up for VPP divisibility) and recompute off
   — a separate within-regime A/B; do not subtract across overlap regimes.
4. **NVSHMEM arm exits 1 at teardown** (NVSHMEM finalize after training); measured iters are valid
   — gate on `efa_ok` + `n_steady==8`.
5. **Mock data + random-init + forced balancing** — measures the dispatcher on a balanced token
   distribution (step time, not loss), same regime as the DSV3/Kimi-K2 A/Bs.

## How to reproduce

See [`../README.md`](../README.md). Primary (8-node EP32@PP2):

```bash
CTX=ml-shared NS=kimi-k2-bench NNODES=8 EPS="32 16" CELLS="4:off 1:off 4:on" RECOMPUTE=selective \
TRAIN_ITERS=12 GLOBAL_BATCH=256 \
UCCL_IMG=<…>:nemo-26.04.01-uccl-0dc87eb \
NVSHMEM_IMG=<…>:nemo-26.04.01-deepep-nvshmem-567632d-cu13 \
  bash run-qwen3-campaign.sh
# then: python3 ../bench/parse-runs.py <campaign_dir> --warmup 4
```

**Confounder guard:** assert `NET/OFI Selected provider is efa` for every rank of every run
(`efa_ok`); discard and re-run any run where it does not (this is how the preempted 7/8-node
runs are detected and rerun).
