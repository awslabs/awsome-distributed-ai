<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Kimi-K2 (literal 384-expert MoE) — UCCL-EP vs NCCL all-to-all — Benchmark Results

MoE token-dispatcher A/B on **literal Kimi-K2** (`moonshotai/Kimi-K2-Base`: 61 layers,
384 routed experts, top-8, MLA, 64 attention heads, `n_group=1`, no MTP), built via
`AutoBridge.from_hf_pretrained(..., trust_remote_code=True).to_megatron_provider()` —
**not** the DeepSeek-V3 256-expert stand-in used in [`../../dsv3/`](../../dsv3/benchmarks/RESULTS.md).
This is the first dispatcher A/B measured on the real K2 architecture.

The only toggle that changes between arms is `MOE_DISPATCHER`
(`alltoall` → NCCL baseline, `deepep` → UCCL-EP over EFA); everything else (model,
mock data, seq 4096, GBS 256, bf16, seed, image, EFA env, `MOE_A2A_OVERLAP` flags)
is held fixed within a cell. 24 iterations per run, `log_interval=1`; the first 4
iterations (incl. the ~300 s compile) are dropped from perf stats.

- Image: `megatron-bridge-uccl:nemo-26.04.01-uccl-0dc87eb` (Bridge 0.4.2, Core 0.17.1)
- Raw logs (no overwrite, all ranks): `/fsx/megatron-bridge-bench/<campaign>/kimi-k2/<arm>-mb<m>-ovl<on|off>/`
- Per-run gate: `NET/OFI Selected provider is efa, fabric is efa-direct` on **all** nodes;
  deepep arms additionally show the UCCL-EP proxy registration (high-throughput mode).

---

## Headline — 32-node PP8 (256× B300), campaign `20260604T083049Z-uccl-ab-pp8-32n-v2`

Parallelism **TP8 / PP8 / EP32 / DP4** (the canonical layout, same as the published
DSV3 numbers). Steady-state mean over iters 5–24, zero stalls, EFA active 32/32 in
every run.

| cell | UCCL deepep | NCCL alltoall | UCCL delta (iter time) |
|---|---|---|---|
| **mb=4, overlap=on** | **3.834 s/iter · 219.1 TF/GPU · 273.5k tok/s** | 5.793 s/iter · 144.8 TF/GPU · 181.0k tok/s | **−33.8%** |
| **mb=4, overlap=off** | **6.041 s/iter · 138.8 TF/GPU** | 9.303 s/iter · 90.1 TF/GPU | **−35.1%** |
| **mb=1, overlap=off** | 14.98 s/iter · 56.0 TF/GPU | **12.88 s/iter · 65.1 TF/GPU** | +16.3% (NCCL faster) |
| mb=1, overlap=on | *not measured* (see coverage note) | *not measured* | — |

**Takeaways**

- **At mb≥4 — the throughput-efficient operating point for both dispatchers — UCCL-EP
  wins decisively on literal K2: −34/−35% iteration time in both overlap regimes.**
  Overlap does not close the gap (it helps both arms; the relative win is unchanged).
- At **mb=1 no-overlap**, NCCL all-to-all is ~16% faster — the same unamortized
  per-dispatch-overhead regime seen on DSV3 (~12.6%) and on the K2 16-node run (~12%).
  A tuned run operates at mb≥4, where UCCL wins.
- The mb4+overlap UCCL advantage is **larger at 32-node/PP8 (−34%) than at
  16-node/PP4 (−21%)** (table below). EP width is **identical in both (EP32)** —
  the difference tracks the doubled node count (wider inter-node all-to-all span)
  and deeper pipeline, not expert-parallel width.

### Numerical (work) equivalence — confirmed

Per-iteration `lm loss` curves (24 iters, `log_interval=1`, last-PP-stage rank) for
deepep vs alltoall within each cell:

| cell | max relative divergence, iters 1–10 |
|---|---|
| mb4, overlap=on | 4.1e-4 |
| mb4, overlap=off | 2.2e-4 |
| mb1, overlap=off | 2.6e-4 |

Curves start ~1e-6 apart at iteration 1 (e.g. 13.438850 vs 13.438960) and diverge
slowly with no systematic offset — bf16 round-off accumulation, not token-dropping or
different routing. **The two dispatchers do the same numerical work**; per-run
`loss_curve.csv` files sit next to the raw logs.

### Coverage note (Capacity Block expiry)

The 32-node Capacity Block ended at 2026-06-04 ~10:51Z (EC2 reclaims instances
~30 min before the reservation end), killing the campaign after 6 of 8 K2 cells:

- **mb1+overlap pair missing at 32-node**: the deepep run was reclaimed mid-flight
  (partial logs preserved at `.../deepep-mb1-ovlon-ABORTED-blockend/`); the alltoall
  counterpart never started. The 16-node campaign measured deepep mb1+ovl (20.95 s)
  but not its alltoall pair, so **no mb1+overlap A/B exists at either scale**.
- The planned same-campaign **DSV3 re-run never started** — DSV3 reference numbers
  remain the separate 2026-06-01 campaign in [`../../dsv3/benchmarks/RESULTS.md`](../../dsv3/benchmarks/RESULTS.md)
  (same image/layout, but not same-campaign apples-to-apples).

## Appendix — 16-node PP4 (128× B300), campaign `20260604T051726Z-uccl-ab-pp4-16n-v2`

Parallelism **TP8 / PP4 / EP32 / DP2** — *not directly comparable to the PP8 table*
(half the GPUs, shallower pipeline); valid for within-cell A/B deltas only.

| cell | UCCL deepep | NCCL alltoall | UCCL delta |
|---|---|---|---|
| mb=4, overlap=on | **6.420 s · 261.3 TF/GPU** | 8.090 s · 207.2 TF/GPU | **−20.6%** |
| mb=4, overlap=off | **9.852 s · 170.1 TF/GPU** | 15.381 s · 108.9 TF/GPU | **−35.9%** |
| mb=1, overlap=off | 26.59 s · 63.0 TF/GPU | **23.76 s · 70.5 TF/GPU** | +11.9% (NCCL faster) |
| mb=1, overlap=on | 20.95 s · 80.0 TF/GPU | *not run* (block handover) | — |

Loss equivalence at this scale was also confirmed (curves match to 3–5 significant
figures through the 13.44 → 0.03 mock-data collapse).

## Reproduce / parse

```bash
# launch one cell (see ../../run-ab-rawpods.sh for all knobs)
MODEL=kimi-k2 ARM=deepep MICRO_BATCH=4 MOE_A2A_OVERLAP=on NNODES=32 \
  CAMPAIGN_ID=<id> bash ../../run-ab-rawpods.sh

# full matrix, serial, one campaign root
MODELS="kimi-k2" CELLS="4:on 4:off 1:off 1:on" ARMS="deepep alltoall" \
  NNODES=32 bash ../../bench/run-campaign.sh

# parse any campaign root → index.csv + per-run loss_curve.csv
python3 ../../bench/parse-runs.py /fsx/megatron-bridge-bench/<campaign-id>
```

Requires `KIMI_K2_HF_PATH` pointing at the K2 HF checkpoint/config directory
(config + tokenizer only; weights are not loaded for the mock-data benchmark) and
`trust_remote_code=True` (K2 ships a custom `configuration_deepseek.py`).
