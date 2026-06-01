<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Kimi-K2 MoE: DeepEP/UCCL-over-EFA vs NCCL all-to-all — Benchmark

This harness measures the performance of UCCL's EFA-native **DeepEP drop-in**
(expert-parallel dispatch/combine over AWS EFA) against the standard **NCCL
all-to-all** token dispatcher, for Kimi-K2 / DeepSeek-V3-class fine-grained MoE
training with NVIDIA Megatron-Bridge / Megatron-Core on **32x p6-b300.48xlarge**
(256x B300, 8 GPU/node, 16x 400 Gbps EFAv4 = 6.4 Tbps/node).

> ## ✅ Measured (2026-06-01) — results are in
>
> This A/B has been **run** on a live 256× B300 block. Full numbers, methodology, and
> caveats: **[`RESULTS.md`](RESULTS.md)**.
>
> **Headline:** at the throughput-efficient operating point (micro-batch ≥ 4), **UCCL
> `deep_ep` is ~36% faster than NCCL all-to-all**, and the advantage **holds under
> deployment-realistic 1F1B overlap** (−35.8% overlap-on, −36.0% overlap-off). NCCL wins
> only at micro-batch 1 (64 tiny dispatches, an operating point no tuned run uses). This
> **supersedes the "honest bottom line" prediction below**, which expected a *modest*
> delta that overlap would compress toward parity — it did not.
>
> **How it ran:** this cluster has **no kubeflow PyTorchJob CRD**, so the A/B ran as **raw
> ranked Pods** via [`run-ab-rawpods.sh`](run-ab-rawpods.sh) driving
> [`bench_dsv3_pretrain.py`](bench_dsv3_pretrain.py). The substrate is the recipe-native
> **DSV3 256-expert** MoE, not Kimi-K2's 384 (overriding the expert count breaks the
> recipe's node-group routing). Metric of record is **mean** steady-state iter time (not
> median — see RESULTS.md on why), with forced router load-balancing to avoid
> degenerate-routing stalls.

## Overview — the A/B (only the dispatcher changes)

The experiment is a strict A/B in which **the only thing that changes is the MoE
token dispatcher**. Everything else — model, data, parallelism, precision, image,
EFA env, and the A2A/EP overlap configuration — is held identical, so any measured
delta is attributable to the dispatcher and not to a confounder.

| arm | dispatcher | what carries dispatch/combine |
|-----|------------|-------------------------------|
| **WITHOUT DeepEP** (baseline) | `moe_token_dispatcher_type="alltoall"` | NCCL all-to-all over EFA (via aws-ofi-nccl) |
| **WITH DeepEP** (treatment) | `moe_token_dispatcher_type="flex"`, `moe_flex_dispatcher_backend="deepep"` | DeepEP kernels; on AWS the `deep_ep` module imported is **UCCL's EFA-native drop-in** (stock NVIDIA DeepEP is NVSHMEM/IBGDA-bound and cannot run on EFA) |

- Model (as run): recipe-native **DSV3 256-expert** MoE (Kimi-K2 family — top-8, MLA,
  hidden 7168, 61 layers). Kimi-K2's native 384 routed experts is **not** used: overriding
  the expert count without recomputing the recipe's node-group routing breaks the build,
  and the dispatcher A/B does not depend on the exact count.
- Parallelism: **TP=8** (intra-node), **EP=32** (spans 4 nodes), **PP=8**, **DP=4** = 256 GPUs (32 nodes x 8 B300).
- Image: `<account>.dkr.ecr.us-west-2.amazonaws.com/megatron-bridge-uccl:nemo-26.04.01-uccl-0dc87eb`
- FSx layout: `/fsx/kimi-k2/{hf,mcore,sft-data,sft-output}`; benchmark outputs -> `/fsx/kimi-k2/bench`
  (rank-0 logs at `/fsx/kimi-k2/bench/logs/`, preserved results as `RESULT-*-rank0.log`).
- Cluster: an EKS cluster with a `p6-b300.48xlarge` capacity-block node group, namespace
  `kimi-k2-bench` (override the context and namespace via `CTX` / `NS`). The validation
  cluster had **no kubeflow PyTorchJob CRD**, so runs launch as raw ranked Pods (see
  `run-ab-rawpods.sh`).

Scripts in this directory:

| file | what it does |
|------|--------------|
| `run-ab-rawpods.sh` | **the launcher** — one arm per call (`./run-ab-rawpods.sh <alltoall\|deepep> [NNODES]`); creates raw ranked Pods + a headless Service + static torchrun rendezvous (no PyTorchJob CRD needed). Env knobs: `EXPERT_PARALLEL`, `MICRO_BATCH`, `GLOBAL_BATCH`, `TRAIN_ITERS`, `MOE_A2A_OVERLAP`, `MOE_FORCE_BALANCE`, `LOSS_PROBE` |
| `bench_dsv3_pretrain.py` | **the entrypoint** torchrun runs — builds the recipe's DSV3 256-expert config, applies the single dispatcher toggle + overlap/VPP/recompute handling, launches `pretrain()`. Staged to FSx at `/fsx/kimi-k2/bench_dsv3_pretrain.py` |
| `RESULTS.md` | results sheet (sweep + overlap=on + work-equivalence + caveats) |

---

## Prior expectation (now SUPERSEDED by the measurement — see the banner / `RESULTS.md`)

> ⚠️ The expectation below was the **pre-run** hypothesis. The actual 256× B300 result
> contradicts it: at micro-batch ≥ 4 UCCL `deep_ep` is ~36% faster and overlap did **not**
> compress the gap. The literature caveats and reference table remain valid context for
> *why this was surprising*, but do not read the "modest / small overlap delta" framing as
> the finding — the finding is in [`RESULTS.md`](RESULTS.md).

For our 256-GPU (32x p6-b300) Kimi-K2 / DSV3-class MoE training A/B, the *pre-run*
expectation was: the *communication-layer* delta from swapping the NCCL all-to-all
token dispatcher for the DeepEP/UCCL-over-EFA drop-in can be large in
microbenchmarks, but the *end-to-end training* delta is most likely modest —
plausibly single-digit to low-double-digit percent on a well-overlapped baseline —
and must be measured, not quoted. (It was measured; the modest-delta expectation did
not hold — the *execution-throughput* difference of the dispatch/combine kernels, not
just exposed comm latency, is what overlap could not hide.)

The critical caveats, all of which the reference table below preserves:

- **DeepEP's own headline numbers are InfiniBand, not EFA.** 153/158 GB/s NVLink,
  ~59 GB/s EP32 internode, and V2's 726/740 GB/s on Blackwell are all H800/SM100 +
  ConnectX-7 **InfiniBand**, and stock DeepEP cannot run on EFA at all. The EFA
  delta depends entirely on **UCCL's** EFA-native drop-in.
- **The only EFA evidence is UCCL's own B200/H200 numbers** (B200 + EFAv4:
  ~53 GB/s dispatch / ~57 GB/s combine at EP32; H200 + EFAv3: 54/43 GB/s) — on
  B200/H200, **not B300**. No published B300 EFA number exists.
- **There is no published DeepEP-on-EFA-vs-NCCL-alltoall training number
  anywhere.** The "2.1x" is UCCL vs PPLX (not NCCL); "+40%" is SGLang inference
  (not training); "+32%" is TorchTitan on InfiniBand (not EFA/Megatron); "+7–45%"
  is AMD MI300X + Broadcom vs RCCL (wrong platform).
- **The decisive bound:** Megatron's 1F1B overlap hides up to ~93% of all-to-all
  latency, and the "up to 60% of step is A2A" figure is the overlap-OFF ceiling.
  The realized training gain is bounded by the *exposed* (post-overlap) comm,
  further compressed because p6-b300's 16x400G EFA (6.4 Tbps/node) is 2x the
  per-GPU bandwidth of every published testbed.

The pre-run rule of thumb was to report **overlap=on** as the deployment number (expected
small) and **overlap=off** as the dispatcher-isolation upper bound. The measurement
overturned the "expected small" part — both were measured and reported (see the banner /
`RESULTS.md`).

---

## Expected / reference table

> **All reference numbers below are labeled by hardware + transport + source tier.
> None of them is a DeepEP-on-EFA-vs-NCCL-alltoall *training* number — that
> measurement did not exist in the literature and is exactly what this benchmark
> produced.** EFA numbers come only from UCCL (the drop-in); DeepEP's own tables
> are InfiniBand. Tier: T1 = official docs/source/paper, T2 = blog cross-checkable.

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

Pins: DeepEP V1 README `e0eaaf94` (2025-04-21); V2 `b306af06` (2026-04-29); UCCL
ep/README `0dc87eb3`; UCCL-EP arXiv:2512.19849v2 (2026-01-22); UCCL blog
2025-10-27; Megatron-Core MoE arXiv:2603.07685; DeepSeek-V3 arXiv:2412.19437; AWS
p6-b300 blog 2025-11-18.

---

## Methodology

### The single toggle — `MOE_DISPATCHER`

`bench_dsv3_pretrain.py` (the validated entrypoint; the un-run PyTorchJob path used
`conf/kimi_k2_sft.py` instead) honors one env var, `MOE_DISPATCHER`:

```python
# conf/kimi_k2_sft.py reads env MOE_DISPATCHER in {"alltoall", "deepep"}:
import os
_disp = os.environ.get("MOE_DISPATCHER", "alltoall")
if _disp == "deepep":
    moe_token_dispatcher_type   = "flex"
    moe_flex_dispatcher_backend = "deepep"   # on AWS the deep_ep module is UCCL's EFA build
else:
    moe_token_dispatcher_type   = "alltoall"  # NCCL all-to-all over EFA — baseline
```

> **Resolved (nemo-26.04.01 / core 0.17.1):** the recipe sets
> `moe_flex_dispatcher_backend="deepep"` and `apply_flex_dispatcher_backend()` sets
> `moe_token_dispatcher_type="flex"`. `moe_enable_deepep` is **deprecated** on this image
> (it emits *"moe_enable_deepep is deprecated. Please use --moe-flex-dispatcher-backend=deepep"*)
> — do **not** set it. `apply_flex_dispatcher_backend` only flips the type to `flex` on a
> B300-allowlisted device (the 0.4.x `startswith("NVIDIA B300")` fix), so
> `bench_dsv3_pretrain.py` hard-asserts `moe_token_dispatcher_type=="flex"` afterward — a
> silent fall-through to alltoall (which would null out the A/B) aborts the run instead.

> On AWS the `deep_ep` Python module imported by the `deepep` backend **must be
> UCCL's EFA-native build**, not stock NVIDIA DeepEP (NVSHMEM/IBGDA-bound, won't
> run on EFA). Megatron only sees the backend string `"deepep"`; which library it
> imports is an image/install concern baked into the `nemo-26.04.01-uccl-0dc87eb`
> image. **Verified:** the image build step asserts `import deep_ep` resolves to
> `/opt/venv/lib/python3.12/site-packages/deep_ep` (UCCL), **not** the base's NVIDIA
> `deep_ep` in `dist-packages`; the single-node gate further confirms `deep_ep.Buffer`.

### Hold fixed across both arms (confounder control)

Everything except the dispatcher is identical — otherwise you measure a
confounder, not the dispatcher: model, data, seq length, global batch, precision
(FP8 dispatch / BF16 combine), parallelism (TP8/EP32/PP8/DP4), random seed, image,
FSx mounts, EFA env (`FI_PROVIDER=efa`, `FI_EFA_FORK_SAFE=1`), and — **most
decisive** — the **A2A/EP overlap flags** (`--overlap-moe-expert-parallel-comm`,
`--delay-wgrad-compute`). Those flags are gated by `MOE_A2A_OVERLAP`, set
**identically for both arms within a run**, so overlap never differs between
`alltoall` and `deepep`.

Env contracts the harness sets (all **verified** in `bench_dsv3_pretrain.py`):

- `MOE_A2A_OVERLAP` in `{on,off}` — gates the overlap path; identical across arms in a
  mode. Treated as ON iff the value == `"on"`. On core 0.17.1, ON additionally requires a
  virtual pipeline + recompute OFF, so the bench sets
  `virtual_pipeline_model_parallel_size=2` (the recipe's shipped `(8,2)` 16-chunk layout)
  and disables recomputation on **both** arms — making overlap=on a separate within-regime
  A/B (see RESULTS.md).
- `TRAIN_ITERS` — total iterations (warmup + measure). Also honored: `MICRO_BATCH`,
  `GLOBAL_BATCH`, `EXPERT_PARALLEL`, `MOE_FORCE_BALANCE` (default on; forces balanced
  routing to avoid degenerate-router stalls), `LOSS_PROBE` (work-equivalence check).

### overlap=on vs overlap=off — and why overlap=on is the deployment number

Run **both** overlap modes and report both:

- **`overlap=on` — realistic deployment.** A2A is overlapped with compute (Megatron
  1F1B). Report as the deployment speedup. *(Measured: the gap did **not** shrink —
  deepep −35.8% at mb=4. The "expect it small" prior was wrong; see RESULTS.md.)*
- **`overlap=off` — dispatcher isolation.** A2A fully exposed (the "up to 60% of
  step" regime). Upper bound on the dispatcher's contribution; never present it as the
  deployment delta, but here it nearly equals the deployment delta (−36.0%).

> **Resolved:** `overlap_moe_expert_parallel_comm` is supported on **both** `alltoall`
> and `flex` dispatchers (core `transformer_config.py` validation), so `overlap=on` runs
> symmetrically on both arms — no need to fall back to overlap=off-only. Both overlap modes
> were measured. Note overlap=on forces VPP=2 + recompute-off on both arms, so its numbers
> are a separate within-regime A/B (do not subtract against overlap=off).

---

## What the benchmark measures

A strict swap of **only** the MoE token dispatcher (NCCL `alltoall` vs UCCL `deepep`),
run end-to-end through a real Megatron-Bridge training step on the full 256-GPU config.
Everything else (model, data, parallelism, precision, image, EFA env, overlap mode) is
held identical, so the iter-time delta is attributable to the dispatcher, not a
confounder. `run-ab-rawpods.sh` launches one arm at a time (changing only
`MOE_DISPATCHER`); verify the nodes are Ready first:

```bash
kubectl --context "$CTX" get nodes -l node.kubernetes.io/instance-type=p6-b300.48xlarge
```

---

## How to run

**Validated path — raw ranked Pods (`run-ab-rawpods.sh`).** One arm per invocation;
each arm holds all 32 nodes, so delete the previous arm's Pods before the next. Stage
`bench_dsv3_pretrain.py` to FSx first (`/fsx/kimi-k2/bench_dsv3_pretrain.py`).

```bash
cd 3.test_cases/megatron/megatron-bridge/kimi-k2/benchmarks
export CTX=<your-kubectl-context>          # required by run-ab-rawpods.sh
export IMG=<your-account>.dkr.ecr.<region>.amazonaws.com/megatron-bridge-uccl:nemo-26.04.01-uccl-0dc87eb

# Headline A/B at the throughput-efficient operating point (mb=4, overlap off), 256 GPU.
# Run deepep, wait for completion, delete its Pods, then run alltoall with the SAME knobs.
EXPERT_PARALLEL=32 GLOBAL_BATCH=256 MICRO_BATCH=4 TRAIN_ITERS=20 \
  MOE_A2A_OVERLAP=off bash run-ab-rawpods.sh deepep 32
EXPERT_PARALLEL=32 GLOBAL_BATCH=256 MICRO_BATCH=4 TRAIN_ITERS=20 \
  MOE_A2A_OVERLAP=off bash run-ab-rawpods.sh alltoall 32

# Deployment-realistic overlap=on (the bench auto-adds VPP=2 + recompute-off on both arms):
EXPERT_PARALLEL=32 GLOBAL_BATCH=256 MICRO_BATCH=4 TRAIN_ITERS=20 \
  MOE_A2A_OVERLAP=on  bash run-ab-rawpods.sh deepep 32      # then alltoall, same as above

# Cheap pre-flight: 8-node smoke (EP capped at 8 on 8 nodes = 64 GPU, DP=1):
EXPERT_PARALLEL=8 GLOBAL_BATCH=8 TRAIN_ITERS=8 MOE_A2A_OVERLAP=off bash run-ab-rawpods.sh deepep 8
```

Read each arm's rank-0 log from FSx (`/fsx/kimi-k2/bench/logs/abrun-<arm>-0.log`) via a
reader pod (`kubectl exec`/`logs` stdout garbles at this scale); scrape
`Step Time : <s>s ... MODEL_TFLOP/s/GPU`, drop the warmup iters, take the **mean**.

Common env overrides for `run-ab-rawpods.sh`: `CTX` (required), `IMG` (required), `NS`,
`EXPERT_PARALLEL`, `MICRO_BATCH`, `GLOBAL_BATCH`, `TRAIN_ITERS`, `SEQ_LEN`,
`MOE_A2A_OVERLAP`, `MOE_FORCE_BALANCE`, `LOSS_PROBE`. Outputs go to `/fsx/kimi-k2/bench`
(rank-0 logs under `logs/`).

### Capacity-block scheduling contract

The capacity-block managed node group carries taints `nvidia.com/gpu`,
`workload=bench`, and `capacity-reservation` (plus matching labels). `run-ab-rawpods.sh`
sets the matching `nodeSelector` + `tolerations` and requests `vpc.amazonaws.com/efa: 16`
per node on every Pod it creates.

---

## How to read the results

1. **Validate the run (EFA-provider assertion).** Both arms must log
   `NET/OFI Selected Provider is efa` on every rank. If a run fell back to the
   sockets provider, it is comparing a different fabric — **discard it**. Grep the
   per-arm rank-0 logs at `/fsx/kimi-k2/bench/logs/abrun-<arm>-0.log`.
2. **Drop warmup, report the MEAN (not median).** Drop the first few iters (iter-1 is a
   ~300 s compile/NCCL/UCCL-proxy-init outlier; the runs used `TRAIN_ITERS=20`, dropping 4).
   Report the **mean** over the steady-state iters. **Median was rejected:** with bimodal
   stalls it reports best-case latency and could hide "deepep stalls 2× as often" as "no
   difference." With forced load-balancing the steady state is unimodal (σ ≈ 0.2–0.4 s,
   0 stalls), so mean is clean — see RESULTS.md.
3. **The per-iteration line (this image).** Megatron-Bridge's logger prints
   `Step Time : <s>s GPU utilization: <tf>MODEL_TFLOP/s/GPU` each `log_interval` (the bench
   sets `cfg.logger.log_throughput=True`, `log_interval=1`). `Step Time` is the primary A/B
   metric; `MODEL_TFLOP/s/GPU` is model-FLOP throughput (MFU = ÷ B300 BF16 peak). This is
   **not** the stock Megatron-LM `elapsed time per iteration (ms)` / `lm loss` line — the
   recipe's logger replaces it, so there is no per-iter loss line (the `LOSS_PROBE=1` hook
   adds one on the last pipeline stage for the work-equivalence check).
4. **tokens/s is DERIVED, not printed.** `tokens/sec` is **not** a Megatron label
   (it appears only on the RL path). The harness derives it:
   `tokens/s = global_batch × seq_len ÷ iter_time_s`. Grepping for a tokens/s
   label scrapes nothing.
5. **There is no per-op dispatch/combine timer.** The MoE layer only wraps
   dispatch/combine in **NVTX ranges** (Nsight). `--timing-log-level` does not
   break them out and `--moe-per-layer-logging` covers only aux/z-loss. The
   per-iteration A/B relies on iter-time + TFLOP/s/GPU; for a per-op breakdown,
   profile with Nsight.
6. **Report both overlap modes for what they are.** **`overlap=on`** is the
   deployment delta — report it as *the* number; **`overlap=off`** is the
   dispatcher-isolation upper bound — report alongside, labeled. When comparing
   measured EFA deltas against the reference table above, flag any case where a
   measured EFA number is placed next to an IB number without a transport label.

---

## Caveats

- **p6-b300 has 2x the per-GPU bandwidth of every published testbed — yet the
  overlap=on delta was NOT small.** 16x400G EFA = 6.4 Tbps/node is double the per-GPU
  bandwidth of UCCL's B200 (8x400G) and H200 (16x200G) testbeds, which we expected would
  shrink the exposed-comm fraction and bias the `overlap=on` delta toward zero. It did
  not: deepep measured **−35.8% with overlap on**, essentially equal to the −36.0%
  overlap-off delta. The dispatcher swap's value here is the dispatch/combine **execution
  throughput** (UCCL EFA-native kernels vs NCCL all-to-all at EP=32), not merely exposed
  comm latency — so overlap, which hides latency, does not erase it. See RESULTS.md.
- **B300 has no published EFA number.** Every EFA reference here is UCCL on B200 or
  H200; there is no B300 + EFA dispatch/combine number in the literature. Treat
  UCCL's B200 + EFAv4 numbers as the proxy and remember the second NIC per GPU on
  p6-b300 is unmodelled by any published measurement.
- **Every reference number is transport-labeled; only the measured rows are
  measured-EFA.** Do not copy an InfiniBand (or inference, or PPLX/RCCL-baseline,
  or non-Megatron) number into a measured-EFA row.
