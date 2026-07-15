<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Kimi-K2 (literal 384-expert MoE) — EP dispatcher results (NCCL / DeepEP+UCCL / DeepEP+NVSHMEM)

MoE token-dispatcher A/B on **literal Kimi-K2** (`moonshotai/Kimi-K2-Base`: 61 layers,
384 routed experts, top-8, MLA, 64 attention heads, `n_group=1`, no MTP), built via
`AutoBridge.from_hf_pretrained(..., trust_remote_code=True).to_megatron_provider()` —
**not** the DeepSeek-V3 256-expert stand-in used in [`../../dsv3/`](../../dsv3/benchmarks/RESULTS.md).
This is the first dispatcher A/B measured on the real K2 architecture.

The only toggle that changes between arms is `MOE_DISPATCHER`
(`alltoall` → NCCL baseline, `deepep` → DeepEP-over-EFA, whose *transport* — UCCL or
NVSHMEM — is fixed by the image; see the arm table in [`../README.md`](../README.md));
everything else (model, mock data, seq 4096, GBS 256, bf16, seed, EFA env,
`MOE_A2A_OVERLAP` flags) is held fixed within a cell. 24 iterations per run,
`log_interval=1`; the first 4 iterations (incl. the ~300 s compile) are dropped from
perf stats. The headline table merges two campaigns — NCCL+UCCL (same-campaign A/B,
2026-06-04) and DeepEP+NVSHMEM (2026-07-14) — with per-column provenance marked there.

- UCCL/NCCL image: `megatron-bridge-uccl:nemo-26.04.01-uccl-0dc87eb` (Bridge 0.4.2, Core 0.17.1)
- Raw logs (no overwrite, all ranks): `/fsx/megatron-bridge-bench/<campaign>/kimi-k2/<arm>-mb<m>-ovl<on|off>/`
- Per-run gate: `NET/OFI Selected provider is efa, fabric is efa-direct` on **all** nodes;
  deepep arms additionally show the UCCL-EP proxy registration (high-throughput mode).

---

## Headline — 32-node PP8 (256× B300), all three arms

Parallelism **TP8 / PP8 / EP32 / DP4** — the reference layout shared by all 256-GPU
dispatcher numbers in this library (chosen for the published DSV3 A/B and reused here
so cells are comparable across model cases; not a tuned-optimal layout for K2). Steady-state mean over iters 5–24, zero stalls, EFA active 32/32 in
every run. Two measurement campaigns feed this table:

- **¹ NCCL + UCCL**: same-campaign A/B, `20260604T083049Z-uccl-ab-pp8-32n-v2`
  (2026-06-04, EKS us-west-2, FSx). The UCCL Δ is a rigorous same-campaign delta.
- **² DeepEP+NVSHMEM**: `20260714T0515Z-k2-nvshmem-pp8-32n-ue1` (2026-07-14, EKS
  `ml-clusters-shared-us-east-1`, 32× p6-b300 **us-east-1-atl-2a local-zone Capacity
  Block**, node-local NVMe via `STORAGE=hostpath`, `GDRCOPY_DEV=on` — see
  [`../README.md`](../README.md)). Image `megatron-bridge-uccl:nemo-26.04.01-deepep-nvshmem-567632d-cu13`
  (NVSHMEM v3.7.0-0, DeepEP `567632d`, the
  [`deepep-benchmark`](../../../../../micro-benchmarks/expert-parallelism/deepep-benchmark)
  NVSHMEM-over-libfabric/EFA build). Same bench entrypoint, layout, GBS, seq, mock
  data, iters, warmup. **The NVSHMEM Δ is cross-campaign** (different cluster/date/
  storage) and carries noise the UCCL Δ does not — read it as indicative, not exact.

| cell | NCCL alltoall¹ | DeepEP+UCCL¹ | DeepEP+NVSHMEM² | UCCL Δ¹ | NVSHMEM Δ² |
|---|---|---|---|---|---|
| **mb=4, overlap=on** | 5.793 s/iter · 144.8 TF/GPU · 181.0k tok/s | 3.834 s/iter · 219.1 TF/GPU · 273.5k tok/s | **3.742 s/iter · 224.6 TF/GPU · 280.2k tok/s** | **−33.8%** | ~−35.4% |
| **mb=4, overlap=off** | 9.303 s/iter · 90.1 TF/GPU | **6.041 s/iter · 138.8 TF/GPU** | 6.632 s/iter · 126.4 TF/GPU · 158.1k tok/s | **−35.1%** | ~−28.7% |
| **mb=1, overlap=off** | **12.88 s/iter · 65.1 TF/GPU** | 14.98 s/iter · 56.0 TF/GPU | 13.538 s/iter · 61.9 TF/GPU · 77.5k tok/s | +16.3% (NCCL faster) | ~+5.1% (NCCL faster) |
| mb=1, overlap=on | *not measured* (see coverage note) | *not measured* | *not measured* | — | — |

**Takeaways**

- **At mb≥4 — the throughput-efficient operating point — both DeepEP transports win
  decisively on literal K2**: UCCL −34/−35% iteration time in both overlap regimes
  (same-campaign), NVSHMEM ~−35/−29% (cross-campaign). Between the two DeepEP
  transports there is **no ranking to read here**: at mb=4+overlap NVSHMEM lands ~2.4%
  under the June UCCL number (within cross-campaign noise), while at mb=4 no-overlap
  it trails UCCL by ~10% — call them **competitive**.
- At **mb=1 no-overlap**, NCCL all-to-all stays fastest — the same unamortized
  per-dispatch-overhead regime seen on DSV3 (~12.6%) and on the K2 16-node run (~12%).
  A tuned run operates at mb≥4, where the DeepEP arms win.
- The mb4+overlap UCCL advantage is **larger at 32-node/PP8 (−34%) than at
  16-node/PP4 (−21%)** (table below). EP width is **identical in both (EP32)** —
  the difference tracks the doubled node count (wider inter-node all-to-all span)
  and deeper pipeline, not expert-parallel width.

**NVSHMEM-arm validity** — every cell: `efa_ok` on all 32 nodes, 0 stalls, `n_steady`
20/20, `parse-runs.py` transport = **nvshmem** (NVSHMEM-over-libfabric init banner on
the 8 EP-group leader nodes; no UCCL proxy lines). Iteration-1 `lm loss` 13.43885 /
13.43207 / 13.43359 (mb4-ovlon / mb4-ovloff / mb1-ovloff); mb4-ovlon matches the June
arms' 13.438850/13.438960 to 5 significant figures — same numerical work, bf16
round-off apart. As designed, every NVSHMEM run wrote `STATUS` then exited 1 at
NVSHMEM finalize (benign; see `../run-ab-rawpods.sh`).

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
# launch one cell — the arm (alltoall|deepep) is the POSITIONAL arg, NNODES the second
# (see ../../run-ab-rawpods.sh for all knobs)
MODEL=kimi-k2 MICRO_BATCH=4 MOE_A2A_OVERLAP=on \
  CAMPAIGN_ID=<id> bash ../../run-ab-rawpods.sh deepep 32

# full matrix, serial, one campaign root
MODELS="kimi-k2" CELLS="4:on 4:off 1:off 1:on" ARMS="deepep alltoall" \
  NNODES=32 bash ../../bench/run-campaign.sh

# parse any campaign root → index.csv + per-run loss_curve.csv
python3 ../../bench/parse-runs.py /fsx/megatron-bridge-bench/<campaign-id>
```

Requires `KIMI_K2_HF_PATH` pointing at the K2 HF checkpoint/config directory
(config + tokenizer only; weights are not loaded for the mock-data benchmark) and
`trust_remote_code=True` (K2 ships a custom `configuration_deepseek.py`).
