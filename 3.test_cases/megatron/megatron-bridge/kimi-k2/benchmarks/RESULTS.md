<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Kimi-K2 UCCL-EP vs NCCL all-to-all — Benchmark Results

Results template for the MoE token-dispatcher A/B: UCCL's EFA-native DeepEP
drop-in (expert-parallel all-to-all over EFA) vs the stock NCCL all-to-all token
dispatcher, for 256-GPU (32x p6-b300) Kimi-K2 / DSV3-class MoE SFT.

The only toggle that changes between
arms is `MOE_DISPATCHER` (`alltoall` -> baseline, `deepep` -> UCCL-EFA treatment);
everything else (model, data, seq length, global batch, precision, parallelism
TP8/EP32/PP8/DP4, seed, image, FSx mounts, EFA env, and the
`MOE_A2A_OVERLAP` flags) is held fixed within a run.

---

## Measured result — 2026-06-01 (256× B300, balanced routing, overlap off & on)

**Headline — the dispatcher winner DEPENDS ON micro-batch granularity (a crossover):**
- At **micro_batch=4** (16 dispatches/iter, the throughput-efficient operating point):
  **UCCL `deepep` is ~36% FASTER** than NCCL all-to-all (6.263 s vs 9.774 s/iter;
  170.5 vs 109.2 MODEL TFLOP/s/GPU). **✓ Work-equivalence CONFIRMED** by a loss-match
  check (caveat 7): iteration-1 loss agrees to ~5 significant figures (deepep 11.897349
  vs alltoall 11.897517, relative diff 1.4e-5 — bf16 round-off; identical num_tokens),
  so the speedup is **equal-work, not token-dropping**.
- At **micro_batch=1** (64 tiny dispatches/iter): NCCL all-to-all is ~12.6% faster
  (12.538 s vs 14.117 s) — UCCL-EP's per-dispatch overhead is unamortized here.

The two points above are `overlap=off` (dispatcher fully exposed). **The
deployment-realistic `overlap=on` regime was also measured** (mb=4) and the UCCL win
**holds at −35.8%** (deepep 3.839 s vs alltoall 5.978 s) — overlap does **not** close
the gap; see [Deployment-realistic overlap=on](#deployment-realistic-overlapon-mb4--the-gap-does-not-close).
The mb=4 row is the practically-relevant one: mb=4 is the faster operating point for
**both** dispatchers, so a tuned run uses mb≥4, and there UCCL deepep wins decisively
in **both** overlap regimes. Full granularity sweep:
[Micro-batch granularity sweep](#micro-batch-granularity-sweep).

The per-arm detail below is the **mb=1** point (kept because it is the worst case for
UCCL and shows the crossover); the mb=4 point is in the sweep table.

| arm (mb=1) | dispatcher | mean iter (s) | median (s) | σ (s) | MODEL TFLOP/s/GPU | derived tok/s | derived tok/s/GPU |
|-----|-----------|--------------:|-----------:|------:|------------------:|--------------:|------------------:|
| baseline  | `alltoall` (NCCL A2A over EFA) | **12.538** | 12.440 | 0.323 | **85.1** | 83,631 | 327 |
| treatment | `deepep` (UCCL `deep_ep` over EFA) | **14.117** | 13.930 | 0.397 | **75.6** | 74,278 | 290 |
| **delta** | deepep − alltoall | **+1.579 (+12.6%)** | +12.0% | — | **−11.2%** | **−11.2%** | −11.2% |

- At mb=1 **deepep is slower** by +12.6% in step time (~12 SE — not noise; σ≈0.3–0.4 s,
  0 stalls in either arm). Reproducible: a separate 8-iter `deepep` run measured
  13.9 s mean, matching this run's 14.1 s. At mb=4 the sign flips (deepep −36%).
- Internal consistency: at each mb, the MODEL-TFLOP/s ratio equals the iter-time ratio
  (mb=1: 85.1/75.6 = 14.117/12.538 = 1.126), confirming identical model FLOPs across
  arms — the delta is purely the dispatcher.
- Both arms ran **identically**: DeepSeek-V3 256-expert MoE (L=61, h=7168, top-8,
  `moe_ffn_hidden_size=2048`), TP8/PP8/EP32/DP4 = 256 GPU, seq 4096, GBS 256
  (64 microbatches/iter), bf16, `moe_router_force_load_balancing=True`,
  `MOE_A2A_OVERLAP=off`, image `megatron-bridge-uccl:nemo-26.04.01-uccl-0dc87eb`
  (bridge 0.4.2 / core 0.17.1). 20 iters/arm, first 4 (incl. the ~300 s compile/init
  warmup) dropped, mean over the remaining 16.
- **Run validity:** every rank logged `NET/OFI Selected provider is efa, fabric is
  efa-direct (found 16 nics)` — true EFA RDMA, no socket fallback. UCCL arm logged
  `Registered proxies … (high-throughput mode)` (UCCL-EP EFA proxy active).

### Why "balanced routing" (and the stall finding it fixed)

With random-init weights + mock data and the router **untrained**, routing is
degenerate: tokens pile onto a few experts, one EP rank's all-to-all floods while the
rest idle, producing **bimodal step times with ~18× stalls** (steady ~14 s iters
punctuated by ~290–310 s near-hangs at ~3 TFLOP/s/GPU — GPUs *waiting*, not
computing). That is an artifact of the untrained router, **not** the dispatcher or
the fabric. Setting `moe_router_force_load_balancing=True` (representative of how a
real run stays balanced via the aux loss) removes the stalls entirely (0/16 in both
arms) and is the regime the table above measures. Median-of-fast-iters was rejected
as the metric because it would hide exactly this stall behavior; the metric of record
is mean iter time over the balanced steady state.

### Caveats (do not over-read this number)

1. **Both overlap regimes were measured (mb=4); the gap held in both.** `overlap=off`
   is the dispatcher-isolation regime; `overlap=on` (1F1B hides the A2A) is the
   deployment regime. The common expectation is that overlap compresses the dispatcher
   delta toward parity — here it did **not**: deepep was −36% (off) and −35.8% (on).
   `overlap=on` required `VPP=2` (the recipe's `(8,2)` layout) + recompute OFF on both
   arms, so it is a separate within-regime A/B (see that section). Remaining overlap
   follow-up: a finer micro-batch sweep under overlap, and `delay_wgrad_compute=on`.
2. **256-expert DSV3 shape**, not literally Kimi-K2's 384 experts (overriding experts
   without recomputing the recipe's node-group routing breaks the build). This is an
   architecture-**family** result.
3. **Random-init + mock data + forced balancing** measures the dispatcher on a
   *balanced* token distribution; real data carries some residual imbalance.
4. Single 20-iter run per arm (within-run σ tiny; between-run variance not bounded,
   though deepep reproduced across two launches).
5. **Message-size regime — this is `micro_batch=1`, the smallest per-dispatch
   granularity.** In 1F1B the token dispatch fires once per microbatch, so mb=1 is
   the *smallest* per-dispatch token volume — and that is exactly the regime where
   the UCCL/DeepEP path is weakest (small-message EFA firmware limits; UCCL-EP's
   per-op proxy/kernel-launch cost amortizes only over larger dispatches — see
   reference table line "EP32 … latency vs PPLX": PPLX wins at ≤128 tokens). Every
   published DeepEP/UCCL win is inference-scale or large-batch. So "NCCL +12%" is the
   **mb=1 (worst-case-for-UCCL) point**, not a granularity-general claim. See the
   [micro-batch sweep](#micro-batch-granularity-sweep) for mb=4.
6. **UCCL at default config**, not its ceiling: no sweep of UCCL's EFA NIC/QP/proxy
   settings was done. This is "UCCL as-shipped at commit `0dc87eb`," not tuned.
7. **✓ Work-equivalence VERIFIED (token-dropping ruled out).** A 36% swing from
   swapping only the dispatcher makes MoE token capacity-dropping a live confound —
   dropped tokens → fewer real FLOPs while the *analytical* FLOP/s number stays
   unchanged, faking a speedup. Two independent checks rule it out:
   - **Config (drop-free by construction):** both arms' full config dumps are
     byte-identical except the dispatcher selector (`flex`+deepep vs `alltoall`); there
     is **no `moe_expert_capacity_factor` / `moe_token_drop_policy` /
     `moe_pad_expert_input_to_capacity`** set (all Mcore-default None), so neither
     dispatcher drops tokens — DSV3's drop-free regime.
   - **Loss-match (numerical equivalence):** re-ran a short mb=4 pair with a loss probe
     (`LOSS_PROBE=1`, prints per-microbatch loss on the last PP stage). Iteration-1
     loss: **deepep 11.897349 vs alltoall 11.897517** (loss_sum 194926.17 vs 194928.92,
     **identical num_tokens=16384**), agreeing to ~5 significant figures (relative diff
     **1.4e-5** — the expected bf16 summation-order round-off). The flex/deepep
     dispatcher routes/combines the same tokens to numerically-equivalent output.
   Conclusion: the mb=4 **−36%** is a genuine equal-work speedup. (mb=1, where deepep
   *lost*, was never subject to this confound — dropping would only have helped it.)

This is, to our knowledge, the **first end-to-end DeepEP-on-EFA-vs-NCCL-alltoall
*training* number on B300** — the reference table below confirms no such measurement
existed in the literature. It is directionally consistent with that table: there is
no published EFA *training* win for the DeepEP/UCCL path over NCCL, and on p6-b300's
6.4 Tbps/node fabric the NCCL all-to-all is already very strong.

### Micro-batch granularity sweep

Per-dispatch token volume scales with `micro_batch`. mb=1 is the worst case for
UCCL-EP (smallest dispatches, per-op overhead unamortized); larger mb makes each
dispatch bigger, where UCCL/DeepEP is expected to close the gap. Same config
otherwise (EP=32, GBS=256, overlap=off, balanced, 256× B300). `delta` = deepep vs
alltoall mean iter time (positive = deepep slower).

| micro_batch | µbatches/iter | alltoall mean (s) | deepep mean (s) | delta | TFLOP/s/GPU (a2a → deepep) |
|------------:|--------------:|------------------:|----------------:|-------|----------------------------|
| 1           | 64            | 12.538            | 14.117          | **deepep +12.6% (slower)** | 85.1 → 75.6 |
| 4           | 16            | 9.774             | 6.263           | **deepep −36.0% (FASTER)** | 109.2 → 170.5 |

**Crossover.** The dispatcher winner flips with granularity. At mb=1 (64 tiny
dispatches) NCCL all-to-all wins by 12.6% — UCCL-EP's per-op proxy/kernel-launch
cost is unamortized. At mb=4 (16 dispatches of 4× volume) UCCL `deepep` wins by 36%
(170.5 vs 109.2 TFLOP/s/GPU). Note mb=4 is the **better absolute operating point for
both** arms (deepep 14.12→6.26 s, alltoall 12.54→9.77 s) — a throughput-tuned run
would use mb≥4, so the **practically-relevant comparison is the mb=4 row, where UCCL
deepep is ~36% faster**. deepep is far more granularity-sensitive (2.25× speedup
mb1→mb4 vs alltoall's 1.28×): it pays a fixed per-dispatch overhead that dominates at
mb=1 and is dwarfed by its higher large-dispatch bandwidth at mb=4. The crossover
lies between mb=1 and mb=4 (not localized further; a finer sweep is follow-up).

### Deployment-realistic overlap=on (mb=4) — the gap does NOT close

The earlier rows are `overlap=off` (dispatcher fully exposed). The deployment regime
enables `overlap_moe_expert_parallel_comm` (1F1B hides the EP all-to-all behind
compute). Enabling it on core 0.17.1 forces a **separate config** for BOTH arms —
virtual pipeline `VPP=2` (the recipe's shipped `(8,2)` 16-chunk layout) and
recomputation OFF — so these numbers are an **independent within-regime A/B, NOT
comparable cell-for-cell to overlap=off** (do not subtract across regimes; recompute
flipped too). Both arms verified in the config dump:
`overlap_moe_expert_parallel_comm=True`, `VPP=2`, recompute disabled, 16-chunk layout.

| overlap=on, mb=4 | mean iter (s) | median | σ | MODEL TFLOP/s/GPU | tok/s | stalls |
|------------------|--------------:|-------:|----:|------------------:|------:|--------|
| `alltoall` (NCCL)  | 5.978 | 5.860 | 0.237 | 178.6 | 175,402 | 0/16 |
| `deepep` (UCCL)    | **3.839** | 3.740 | 0.215 | **278.5** | 273,111 | 0/16 |
| **delta** (deepep) | **−35.8% (FASTER)** | −36% | — | **+55.9%** | +55.7% | — |

**The headline finding: overlap does NOT erase the dispatcher gap.** UCCL `deepep` is
~36% faster than NCCL all-to-all *with* deployment-realistic overlap — essentially the
same delta as overlap=off (−36%). This is **contrary to the usual expectation** that
1F1B overlap compresses the dispatcher delta toward parity: here both arms sped up
~1.6× from overlap+recompute-off, **preserving the ratio**. The implication is that
the difference is not merely *exposed* all-to-all latency (which overlap would hide)
but the dispatch/combine **execution throughput** itself — at EP=32 over EFA, NCCL's
all-to-all is the bottleneck whether or not it is overlapped. So on this 256× B300 /
EFA DSV3-class setup, UCCL `deepep` is the better dispatcher at the efficient operating
point (mb≥4) in **both** the isolation and the deployment-overlap regimes.

## Expected / reference table (verbatim from spec)

Every reference number is labeled by hardware + transport + source tier. None of
them is a DeepEP-on-EFA-vs-NCCL-alltoall *training* number — that is exactly what
this benchmark produced (see the Measured result above).

<!-- markdownlint-disable MD013 -->

> **All reference numbers below are labeled by hardware + transport + source tier. None of them is a DeepEP-on-EFA-vs-NCCL-alltoall *training* number — that measurement did not exist in the literature and is exactly what this benchmark produced.** EFA numbers come only from UCCL (the drop-in); DeepEP's own tables are InfiniBand. Tier: T1 = official docs/source/paper, T2 = blog cross-checkable.

| metric | without-DeepEP / baseline | with-DeepEP / UCCL-EFA | delta | hardware / transport | source + tier | caveat |
|--------|---------------------------|------------------------|-------|----------------------|---------------|--------|
| EP32 internode dispatch BW (normal kernels) | — | 53 GB/s (2072 us) | — | B200 + 8x400G **EFAv4** (p6-b200) | UCCL ep/README @0dc87eb3 — **T1** | closest published EFA training proxy; B200 not B300; 1 NIC/GPU vs target 2 |
| EP32 internode combine BW (normal kernels) | — | 57 GB/s (3724 us) | — | B200 + EFAv4 | UCCL ep/README @0dc87eb3 — **T1** | BF16 combine, ~1.8x dispatch latency; forward only (no backward published) |
| EP32 internode dispatch / combine | — | 54 / 43 GB/s | — | H200 + 16x200G **EFAv3** (p5en) | UCCL blog 2025-10-27 — **T1** | combine collapses to 18 GB/s @ EP16; 2-NIC/GPU topology like target but older GPU |
| dispatch+combine throughput | **PPLX** (best EFA EP solution) | up to **2.1x** PPLX | 2.1x | H200/B200 + EFA | UCCL-EP arXiv:2512.19849v2 — **T1** | **baseline is PPLX, NOT NCCL**; per-kernel, not training; "up to"/batch-dependent |
| EP32 dispatch / combine latency vs PPLX | PPLX | 2.3x / 1.1–1.5x lower | — | H200 + EFAv3 | arXiv:2512.19849v2 — **T1** | medium/large batch only; PPLX wins at ≤128 tokens (EFA small-msg firmware limit) |
| EP32 internode dispatch / combine | — | 59 / 60 GB/s | — | H800 + CX7 **InfiniBand** | DeepEP README @e0eaaf94 — **T1** | **IB NOT EFA**; the canonical "DeepEP internode" cite; stock DeepEP can't run on EFA |
| intranode dispatch / combine | — | 153 / 158 GB/s | — | H800 **NVLink** | DeepEP README @e0eaaf94 — **T1** | **NVLink**; ~99% of 160 GB/s ceiling; not internode, not EFA |
| V2 intranode dispatch / combine (Blackwell) | — | 726 / 740 GB/s | — | SM100 **NVLink** (IB testbed) | DeepEP README @b306af06 — **T1** | closest Blackwell proxy but **logical BW** (incl. local traffic), NVLink not EFA |
| SGLang prefill throughput @ EP32 (Qwen3-235B) | 44K tok/s (**NCCL**) | 62K tok/s (UCCL) | **+40%** | H200 + EFAv3 | arXiv:2512.19849v2 Fig13 — **T1** | **inference, not training**; only NVIDIA+EFA-vs-NCCL point; large partly because NCCL can't scale at large EP |
| DeepSeek-V3 training tok/s (DeepEP A/B) | 651 (BF16 EP) | 859 (+DeepEP); 918 (+MXFP8) | **+32%** (DeepEP-only) | B200, NVLink + **IB** | TorchTitan blog — **T2** | **IB not EFA; TorchTitan not Megatron**; +41% headline includes MXFP8 |
| DeepSeek-V3 training tok/s vs RCCL | RCCL | +7% to +45% tok/s | — | AMD MI300X + Broadcom Thor-2 | arXiv:2512.19849v2 Fig14 — **T1** | **AMD, not EFA/NVIDIA/NCCL**; 379B/32-layer downscaled; only end-to-end training A/B that exists |
| A2A share of step (unoptimized) | — | — | up to 60% | DSV3 cross-node EP (NVIDIA) | Megatron-Core arXiv:2603.07685 — **T1** | **overlap-OFF ceiling**, not a realized gain; sets max headroom only |
| A2A latency hidden by 1F1B overlap | — | — | up to 93% | Megatron-Core | Megatron-LM README — **T1** | the decisive bound: shrinks the realized end-to-end training delta to the *exposed* residual |
| compute-to-comm ratio (cross-node EP A2A) | — | — | ~1:1 before overlap | 2048x H800, NVLink+IB | DeepSeek-V3 arXiv:2412.19437 §3.2.1 — **T1** | A2A ≈ compute pre-overlap; DualPipe then drives it to near-zero exposed |

Pins: DeepEP V1 README `e0eaaf94` (2025-04-21); V2 `b306af06` (2026-04-29); UCCL ep/README `0dc87eb3`; UCCL-EP arXiv:2512.19849v2 (2026-01-22); UCCL blog 2025-10-27; Megatron-Core MoE arXiv:2603.07685; DeepSeek-V3 arXiv:2412.19437; AWS p6-b300 blog 2025-11-18.

<!-- markdownlint-enable MD013 -->

## Honest bottom line

The *pre-run* expectation was that the communication-layer delta from swapping the NCCL
all-to-all dispatcher for the DeepEP/UCCL-over-EFA drop-in can be large in microbenchmarks,
but the end-to-end training delta is most likely modest — plausibly single-digit to
low-double-digit percent on a well-overlapped baseline. **The measurement overturned this**
(see Measured result above). Key caveats baked into the reference table that made it
surprising:

- DeepEP's own headline numbers (153/158 GB/s NVLink, ~59 GB/s EP32 internode, V2's
  726/740 GB/s on Blackwell) are H800/SM100 + ConnectX-7 **InfiniBand**, and stock
  DeepEP cannot run on EFA at all.
- The only EFA evidence is **UCCL's** EFA-native drop-in (B200+EFAv4 ~53/57 GB/s
  dispatch/combine at EP32; H200+EFAv3 54/43 GB/s) — measured on B200/H200, **not
  B300**.
- There is **no published DeepEP-on-EFA-vs-NCCL-alltoall training number anywhere**:
  the "2.1x" is UCCL vs PPLX (not NCCL), "+40%" is SGLang inference (not training),
  "+32%" is TorchTitan on InfiniBand (not EFA/Megatron), and "+7–45%" is AMD
  MI300X+Broadcom vs RCCL (wrong platform).
- Megatron's 1F1B overlap hides up to ~93% of all-to-all latency; the "up to 60% of
  step is A2A" figure is the overlap-OFF ceiling. p6-b300's 16x400G EFA
  (6.4 Tbps/node) is 2x the per-GPU bandwidth of every published testbed, which
  further compresses the exposed (post-overlap) comm.

The pre-run rule of thumb — "report overlap=on as the deployment number (expected small),
overlap=off as the dispatcher-isolation upper bound" — assumed overlap would shrink the
gap. It did not (see below); both regimes were measured and reported.

**What we actually measured (2026-06-01):** see
[Measured result](#measured-result--2026-06-01-256-b300-overlapoff-balanced-routing).
The dispatcher-isolation delta is **not** a fixed number — it **crosses over** with
micro-batch: NCCL +12.6% at mb=1, UCCL `deepep` **−36%** (faster) at mb=4. At the
efficient operating point (mb≥4) UCCL deepep wins clearly on this 256× B300 / EFA
DSV3-class setup. The deployment-realistic `overlap=on` regime (mb=4) was **also
measured** and the win **held at −35.8%** — contrary to the usual expectation that
1F1B overlap compresses the dispatcher delta toward parity, here it did not, implying
the gap is dispatch/combine *execution* throughput (UCCL EFA-native kernels vs NCCL
all-to-all at EP=32), not merely exposed comm latency that overlap could hide.

## How to reproduce

Run the A/B on a 32-node (256-GPU) p6-b300 block with `run-ab-rawpods.sh` (see
[`README.md`](README.md) for the exact commands). One arm per call; delete the arm's
Pods between runs. Drop the warmup iters and take the **mean** of `Step Time` and
`MODEL_TFLOP/s/GPU` over the steady-state iters; derive tokens/s as
`global_batch × seq_len / iter_time_s` (it is not a printed Megatron label). For the
deployment-realistic number set `MOE_A2A_OVERLAP=on` (the launcher auto-adds VPP=2 +
recompute-off); for the dispatcher-isolation number set `MOE_A2A_OVERLAP=off`.

**Confounder guard:** assert `NET/OFI Selected provider is efa` appears in the logs
for every rank of every run. Discard and re-run any run where it does not.
