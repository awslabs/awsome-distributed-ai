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
| Global batch | 256 — EP32: 16 iters/run (12 steady) · EP16: 12 iters/run (8 steady) |
| Images | `…-uccl-0dc87eb` (UCCL) · `…-deepep-nvshmem-567632d-cu13` (NVSHMEM v3.7.0-0, DeepEP 567632d, libfabric/EFA) |
| Campaign | EP32 `qwen3-ep32-remeasure-20260624` (git b73a2d7) · EP16 `qwen3-8n-ep32pp2-20260621` (git 633c791) — both 8-node, recompute=selective |

### EP32 @ PP2, overlap=off (dispatcher-isolation) — mean iter s · TFLOP/s/GPU · tok/s

| mb | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM | UCCL Δ | NVSHMEM Δ |
|---:|----------------:|------------:|---------------:|-------:|----------:|
| 4 | **15.76 s · 153.9 · 66.5k** | 19.16 s · 130.8 · 54.7k | 21.12 s · 117.9 · 49.6k | +21.5% | +34.0% |
| 1 | **33.36 s · 72.7 · 31.4k** | 46.88 s · 52.3 · 22.4k | 44.81 s · 54.9 · 23.4k | +40.5% | +34.3% |

### EP32 @ PP2, overlap=on (1F1B, deployment regime) — mean iter s · TFLOP/s/GPU · tok/s

| mb | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM | UCCL Δ | NVSHMEM Δ |
|---:|----------------:|------------:|---------------:|-------:|----------:|
| 4 | **10.90 s · 227.1 · 96.2k** | 18.31 s · 143.6 · 57.3k | 17.81 s · 148.0 · 58.9k | +68.0% | +63.4% |

### EP16 @ PP4 (TP8/PP4/DP2) — internode EP across 2 nodes

| mb · overlap | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM | UCCL Δ | NVSHMEM Δ |
|---|----------------:|------------:|---------------:|-------:|----------:|
| 4 · off | 12.21 s · 198.6 · 85.9k | 13.06 s · 185.8 · 80.3k | **11.57 s · 209.6 · 90.6k** | +6.9% | **−5.2%** |
| 1 · off | 34.17 s · 71.0 · 30.7k | 40.00 s · 60.7 · 26.2k | _preempted_ | +17.1% | — |
| 4 · on  | **10.53 s · 235.0 · 99.5k** | 13.24 s · 187.1 · 79.2k | 11.32 s · 218.7 · 92.6k | +25.7% | +7.5% |

> **Provenance.** All nine EP32 cells above are from a single clean re-measure campaign
> (`qwen3-ep32-remeasure-20260624`, git b73a2d7) and are fully reproducible from the retained
> rank logs (`efa_ok` 8/8, transport confirmed per arm). This re-run replaces an earlier 8-node
> pass whose EP32 overlap=off timing-rank logs were clobbered by a same-`CAMPAIGN_ID` re-run
> (the all-rank skip-guard now fixed in `run-ab-rawpods.sh` prevents that). The re-measured
> overlap=off numbers land within ~1–4% of the earlier pass — the ordering and conclusion are
> unchanged — and the overlap=on row, which previously showed both DeepEP backends at a
> suspicious digit-identical `+55.1%`, now resolves into two distinct runs (UCCL +68.0%,
> NVSHMEM +63.4%). One EP16 cell (`nvshmem-ep16-mb1-off`) remains lost to mid-run preemption.
> NCCL overlap=on logged 7 steady iters (teardown truncation) vs 12 for the DeepEP cells; mean
> iter time is period-stable and matches the earlier pass (10.90 vs 11.01 s), so the large
> +60-something% deltas are unaffected.

### What the numbers say (primary)

1. **At the clean EP32 @ PP2 regime, NCCL all-to-all is the fastest dispatcher in every cell and
   both DeepEP backends are slower** — **large, robust deltas** (not n=1 noise): at mb=4,
   DeepEP+UCCL **+21.5%** and DeepEP+NVSHMEM **+34.0%** slower than NCCL; at mb=1, **+40.5%** and
   **+34.3%**. NCCL all-to-all also scales fine into the `overlap=on` deployment regime (10.9 s,
   **96k tok/s**, 227 TFLOP/s/GPU at mb=4), where the DeepEP backends sit at **+68.0%** (UCCL) /
   **+63.4%** (NVSHMEM).
1b. **The DeepEP penalty grows with EP degree.** At the lower **EP16** the gap nearly closes —
   DeepEP+NVSHMEM is actually *fastest* at mb=4 overlap=off (**−5.2%** vs NCCL) and UCCL only
   +6.9%; going EP16→EP32 the NVSHMEM delta swings from −5% to **+34%** and UCCL from +7% to
   **+22%**. The same EP-scaling trend appears far more violently on H100/30B
   ([`../../qwen3-30b`](../../qwen3-30b/benchmarks/RESULTS.md)). So the all-to-all fan-out, not
   just the model, decides whether DeepEP is competitive — and on this B300/EFA fabric it only
   is at the lower EP degree.
2. **This overturns the 4-node EP32 result** (secondary table), where DeepEP+UCCL *looked* 5.6%
   *faster* than NCCL at EP32. That apparent win was an artifact of the 4-node **PP1 +
   `RECOMPUTE=full`** regime (recompute roughly doubles compute and shrinks the all-to-all's
   share of the step, compressing/distorting the dispatcher delta). At PP2 with lighter
   `selective` recompute the all-to-all is a larger share of the step and **NCCL is unambiguously
   fastest**. Between the two DeepEP backends the order is close and regime-dependent: UCCL edges
   NVSHMEM at mb=4 overlap=off (+21.5% vs +34.0%), while NVSHMEM edges UCCL at mb=1 (+34.3% vs
   +40.5%) and at overlap=on (+63.4% vs +68.0%) — NVSHMEM is the marginally stronger DeepEP
   backend in the comm-heavier cells, but neither comes close to NCCL at EP32 on this 6.4
   Tbps/node B300 fabric.
3. Consistent with the DSV3 "honest bottom line": there is no published EFA *training* win for
   the DeepEP/UCCL path over NCCL all-to-all, and p6-b300's per-node bandwidth makes NCCL's
   all-to-all very strong. (DeepEP/UCCL wins in the literature are inference-scale or on
   InfiniBand / lower-bandwidth fabrics.)

### Validity (primary)

17 of 18 cells valid: EFA active on all 64 ranks (`efa_ok`, 8/8 node logs) · 0 stalls ·
per-arm transport confirmed (`uccl_ok` / `nvshmem_ok`). Steady iters: **12** for all nine EP32
cells (16 iters, drop 4) except NCCL overlap=on at 7 (teardown truncation; period-stable mean);
**8** for EP16. The one invalid cell is `nvshmem-ep16-mb1-off` (mid-run preemption).
Work-equivalence is confirmed directly in-campaign: at EP32 overlap=on, iter-1 `lm loss` matches
across all three arms to ~5 sig figs (NCCL 12.751420 · UCCL 12.751430 · NVSHMEM 12.751400,
rel ≤2e-6) — the dispatchers compute the same thing, so the timing deltas are not a silent
mis-route. The NVSHMEM arm exits non-zero at NVSHMEM finalize *after* training — validate via
`efa_ok` + `n_steady`, not exit code.

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
   The primary EP32 deltas are large enough (+21% to +68%) to be robust at n=1 — corroborated by
   the re-measure landing within ~1–4% of the earlier pass; the secondary's sub-±6% deltas are
   not. n≥3 with error bars is the recommended next step.
3. **`overlap=on` uses 96 layers** (Qwen3's 94 rounded up for VPP divisibility) and recompute off
   — a separate within-regime A/B; do not subtract across overlap regimes.
4. **NVSHMEM arm exits 1 at teardown** (NVSHMEM finalize after training); measured iters are valid
   — gate on `efa_ok` + `n_steady`, not exit code.
5. **Mock data + random-init + forced balancing** — measures the dispatcher on a balanced token
   distribution (step time, not loss), same regime as the DSV3/Kimi-K2 A/Bs.

## How to reproduce

See [`../README.md`](../README.md). Primary (8-node EP32@PP2):

```bash
CTX=ml-shared NS=kimi-k2-bench NNODES=8 EPS="32 16" CELLS="4:off 1:off 4:on" RECOMPUTE=selective \
TRAIN_ITERS=16 GLOBAL_BATCH=256 \
UCCL_IMG=<…>:nemo-26.04.01-uccl-0dc87eb \
NVSHMEM_IMG=<…>:nemo-26.04.01-deepep-nvshmem-567632d-cu13 \
  bash run-qwen3-campaign.sh
# then: python3 ../bench/parse-runs.py <campaign_dir> --warmup 4
```

**Confounder guard:** assert `NET/OFI Selected provider is efa` for every rank of every run
(`efa_ok`); discard and re-run any run where it does not (this is how the preempted 7/8-node
runs are detected and rerun).
