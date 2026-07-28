<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# DeepSeek-R1 on SGLang + DeepEP-over-EFA — measured results

End-to-end serving results for the image in this directory, all on **H200** (`p5en.48xlarge`), in
two topologies:

- **Colocated 2-node TP16/EP16** — what [`recipe/serve.sh`](../recipe/serve.sh) launches.
- **4-node 2P2D prefill/decode-disaggregated** — what [`recipe/serve-pd.sh`](../recipe/serve-pd.sh)
  launches, with the KV cache over EFA RDMA via Mooncake.

Different topologies with different conclusions, so each is labelled and they are not read across
as one experiment. For the **kernel-level** dispatch/combine bandwidth of the same DeepEP-EFA build
— including the 8-to-32-node scaling runs and the NCCL/UCCL cross-backend comparison — see
[`micro-benchmarks/expert-parallelism/deepep-benchmark`](../../../../../micro-benchmarks/expert-parallelism/deepep-benchmark)
and [`ep-backend-comparison`](../../../../../micro-benchmarks/expert-parallelism/ep-backend-comparison).

An earlier revision of this page also carried H100 (`p5.48xlarge`) serving tables. They have been
removed: they reached the same conclusions on a cluster that no longer exists, and one of them (the
H100 2P2D run) predated the `MOONCAKE_PROTOCOL=efa` fix, so its absolute numbers were depressed by
a KV path on TCP. The one H100 finding worth keeping is preserved below under
[UCCL-EP is instance-type sensitive](#uccl-ep-is-instance-type-sensitive).

## Read this first

**The best MoE backend depends on the stage, and no single backend wins both.**

| Stage | Winner | Margin |
|---|---|---|
| **Prefill** (large batches, throughput) | **DeepEP** | 161.5k input tok/s at 1K×conc256 — **3.0×** the best any other backend reaches at 1K, **3.6×** at matched concurrency |
| **Decode** (small batches, latency) | **UCCL-EP + DP-attention** | 4094 output tok/s at conc128, TPOT 28 ms — **1.9×** DeepEP, **1.13×** the best non-DP config |

Because prefill and decode run on *separate* node groups in a PD deployment, you can act on that:
**run DeepEP on the prefill role and UCCL-EP + DP-attention on the decode role.** That split is the
most useful result in this document and it is only available because the roles are disaggregated.

**On decode at 16 GPUs, DeepEP loses to everything** — SGLang's ordinary all-to-all, pure TP, and
UCCL-EP. That is the expected result at this scale, not a defect in the EFA port:

- With 16 ranks and 256 experts, each GPU already owns 16 experts. The MoE fan-out is small and
  largely intra-node NVLink, which the fused-MoE path handles at near-zero overhead. DeepEP's
  per-layer dispatch/combine launches, NVSHMEM RDMA setup and grouped-GEMM cost more than they save
  when so little traffic crosses the network.
- DeepEP targets **large-scale** expert parallelism — tens of nodes, experts spread thin enough
  that every token must cross the fabric, where an ordinary all-to-all's message count and lack of
  compute/comm overlap make it collapse. That crossover is not reachable on 2 nodes for decode.

**But roughly half of DeepEP's decode deficit is a configuration artefact, not the kernels.** Every
DeepEP decode row in Part 1 is `--deepep-mode auto`. Pinning `low_latency` doubles conc-32 throughput
(274 → 554 tok/s), halves TPOT (115.9 → 54.5 ms) and cuts p99 ITL 5.5× (298 → 54 ms). DeepEP still
loses to pure TP at 16 ranks, but by 2.2× rather than 4.5×. **If you benchmark DeepEP decode under
`auto`, you are measuring SGLang's mode-selection heuristic as much as DeepEP** — see
[Pinning `--deepep-mode`](#pinning---deepep-mode).

**Prefill is where DeepEP wins, but only once the batch is large.** The colocated sweep stops at
concurrency 32, where DeepEP has only just overtaken the baseline. Pushed to concurrency 256 in the
2P2D runs it reaches **161.5k input tok/s — 3× the alternatives** at the same 16 GPUs.

**What all of this does establish:** the DeepEP dispatch/combine kernels are correct and run at
IB-class bandwidth over EFA with no InfiniBand present, and the build plugs into a real
DeepSeek-R1 SGLang server end to end, in both topologies. Before this port, DeepEP could not use
EFA at all — NVSHMEM offered only the IBRC/IBGDA InfiniBand transports.

## Caveats — please read before quoting any number

1. **Single-seed smoke benchmarks.** Each point is one pass of `sglang.bench_serving`. These are
   directional, not statistically tight. Re-run across seeds and compute intervals before treating
   any delta as real. The prefill sweeps in particular are noisy at low concurrency, where a
   4–16-request point is dominated by warmup.
2. **`--deepep-mode` is not uniform across the rows.** The 2026-07-28 colocated rows and the non-DP
   2P2D **DeepEP** rows use `auto`; DeepEP+DP and **every UCCL row** pin `normal` on prefill and
   `low_latency` on decode (UCCL requires the pinned form — see
   [UCCL-EP launch configuration](#uccl-ep-launch-configuration-that-actually-matters)). Under
   `auto` the per-batch kernel choice is a hidden variable, so DeepEP-vs-UCCL is confounded by mode
   selection at the non-DP points and clean at the DP points. **This is not a small confound**: at
   conc 32 the pinned-`low_latency` DeepEP server does 2.0× the throughput of the `auto` one, so a
   `auto`-vs-pinned-UCCL comparison at low concurrency understates DeepEP by about a factor of two.
   Both `recipe/serve.sh` (via `DEEPEP_MODE`) and `recipe/serve-pd.sh` (per role) can pin it — but
   pinning `low_latency` on a *colocated* server takes three extra settings plus a lower
   `--mem-fraction-static`, and pinning either mode changes more than the kernel. Both pinned modes
   were measured on 2026-07-28; see [Pinning `--deepep-mode`](#pinning---deepep-mode).
3. **UCCL rows come from a different image**, everything else from this sample's image, so
   UCCL-vs-others is one step less controlled than DeepEP-vs-baseline-vs-TP (which are same-image).
   The two images were bridged by running the `baseline` and `pure TP` configs on both. At
   saturation they agree — the four highest-concurrency prefill points per input length are within
   **±5%** — but at low concurrency they diverge wildly (`baseline` 1K×conc8: 11.0k vs 17.2k, +57%;
   `pure TP` 4K×conc1: 21.8k vs 11.8k, −46%; `pure TP` 64K×conc2: 52.4k vs 11.1k), which is
   consistent with caveat 1: a 4–24-request point is warmup noise, not a measurement. **Treat the
   saturated points as comparable across images and the low-concurrency points as not comparable at
   all.** The bridge was only run for prefill — there is **no cross-image control on the decode
   sweeps**, so the UCCL decode rows carry an unquantified image delta. Given they are the basis of
   this document's decode recommendation, that control is the first thing to add on a re-run.
4. **`NCCL_NET_PLUGIN=ofi` is required for this image, but the tables here are not biased by its
   absence.** NCCL's built-in plugin search name is `libnccl-net.so`. That file does not exist in
   the image built from the committed Dockerfile — the EFA installer there lays down only
   `libnccl-net-ofi.so` — so without the env var NCCL silently falls back to `NET/Socket` with GPU
   Direct RDMA disabled. Hence the `ENV NCCL_NET_PLUGIN=ofi` in the Dockerfile and the assertion in
   `recipe/verify-image.sh`; keep both.

   It does **not** follow that the runs on this page were on TCP, and specifically not the Part 2
   runs, which predate the Dockerfile fix. EFA installer 1.47/1.48 — what every run here used —
   ships the plugin under *both* names, including the default `libnccl-net.so`, so NCCL auto-loaded
   it with no env var set. Verified two ways: `find /opt/amazon -name 'libnccl-net*'` on a 1.47.0
   host returns both files; and re-running the `pure TP` prefill sweep — the arm where *every*
   collective is NCCL, so the most sensitive one — with the var explicitly set reproduces an
   earlier sweep of the same points within ±4% on three of five, +17% on one and −12% on another.
   Scatter in both directions, no systematic gain; a socket→EFA-RDMA switch would be large, uniform
   and one-directional. So do not read the DeepEP-vs-rest comparisons as handicapping the NCCL arms.
   (An earlier revision of this caveat claimed the opposite. That was an inference from the image,
   not an observation, and it does not hold. No NCCL debug log was retained from the Part 2 runs, so
   this rests on the packaging check plus the re-run, not on a transport line from the runs
   themselves.)
5. **Server flags are not identical between the two topologies.** 2P2D used
   `--mem-fraction-static 0.82 --chunked-prefill-size 16384 --watchdog-timeout 1200`;
   `recipe/serve.sh` sets only `--mem-fraction-static` and takes SGLang's defaults for the other
   two (`chunked_prefill_size=8192`, `watchdog_timeout=300`). Another reason the two topologies are
   not a controlled A/B for each other.
6. **Kernel-level bandwidth is mostly out of scope here.** The smoke numbers under
   [Kernel-level context](#kernel-level-context-why-the-efa-port-is-nonetheless-sound) are
   pre-flight checks. For matched-config dispatch/combine across NCCL/UCCL/NVSHMEM out to 256
   ranks, see
   [`ep-backend-comparison`](../../../../../micro-benchmarks/expert-parallelism/ep-backend-comparison).
7. **Peak input throughput is a saturation metric, not a latency metric.** The high-concurrency
   points have TTFT in the seconds. Read the TTFT column too if you care about interactive prefill.
8. **`recipe/benchmark.sh` gained an HF cache mount partway through these runs.** `--dataset-name
   random` still downloads the 256 MB ShareGPT blob to source its token ids, and the bench container
   is `--rm`, so before the fix every point re-downloaded it. That does not change any measurement
   (the download completes before `bench_serving` sends a request), but it can **hang**: on
   2026-07-28 a client sat 20+ minutes at 0% CPU inside `huggingface_hub`'s `xet_get` with a healthy
   server and idle GPUs, which looks exactly like a stuck benchmark. If a sweep appears frozen with
   no output, check the client's Python stack before suspecting the server.

---

# Part 1 — Colocated 2-node TP16/EP16 (matches `recipe/serve.sh`)

**Cluster:** 4× `p5en.48xlarge` (8×H200 141GB, sm_90), us-east-2, 16 EFA NICs/node, iface
`enp71s0`. SGLang 0.5.13.post1 + DeepEP `567632d` + NVSHMEM v3.7.0 (libfabric/EFA). Serving 2 nodes
/ TP=16 / EP=16, `--mem-fraction-static 0.85`. All DeepEP kernel correctness checks passed first.

`pure TP` = `--tp-size 16` with no `--ep-size` (MoE weights TP-sharded; no dispatch/combine at
all) = `MOE_BACKEND=tp`. `Baseline` = EP with the ordinary fused-MoE all-to-all =
`MOE_BACKEND=baseline`. `DeepEP` = `MOE_BACKEND=deepep`.

**Date: 2026-07-28.** Harness: `recipe/benchmark.sh` with `--random-range-ratio 1.0` (fixed
sequence lengths), `--seed 42`, and `--warmup-requests` scaled to each point's concurrency. All
three matter, because `bench_serving`'s defaults are traps:

- **`--random-range-ratio` defaults to 0.0**, which samples every length uniformly from `[0, len]`
  rather than using `len`. A nominal "256 in / 512 out at conc 64" point then pushes about half the
  tokens, and short requests keep draining the batch so sustained concurrency lands near 46 instead
  of 64. Measured back-to-back on one live server, same nominal point, differing only in that flag:
  **374 tok/s at ratio 0.0 versus 1127 tok/s at ratio 1.0**, with mean E2E latency roughly equal
  (32.2 s vs 29.1 s). The 3× is the workload changing, not the server getting faster.
- **`--warmup-requests` defaults to 1**, which only exercises the batch-1 code path.

Set both explicitly on every arm, or the numbers are not comparable to anything — including to each
other.

## Prefill-only (`--random-output-len 1`, so TTFT ≈ prefill time)

TTFT ms (lower better) / input-token tok/s (higher better). **Bold = best in row.**

| Input × conc | pure TP | Baseline | DeepEP (`auto`) |
|---|---|---|---|
| 1024 × 1  | 88 / **11.5k** | 91 / 11.0k | 270 / 3.7k |
| 4096 × 1  | **155** / **26.1k** | 179 / 22.6k | 347 / 11.5k |
| 8192 × 1  | 210 / 38.7k | 217 / 37.5k | **297** / 27.2k |
| 4096 × 8  | **850** / 35.9k | 1051 / 29.4k | 816 / **37.9k** |
| 4096 × 32 | 3406 / 33.4k | 2059 / 49.9k | **1905** / **55.9k** |

**At low concurrency the non-EP paths win 2–3× on TTFT** — EP's per-layer dispatch/combine overhead
is not amortized when few tokens move. **DeepEP overtakes at concurrency ≥8** and leads clearly at
conc 32 (55.9k tok/s vs 49.9k baseline, 33.4k pure TP). The 2P2D sweep in Part 2 continues that ramp
to concurrency 256, where the margin becomes 3×.

## Decode-only (input 256, output 512, so TPOT dominates)

Output tok/s (higher better) / mean TPOT ms (lower better). **Bold = best in row.**

| Concurrency | pure TP | Baseline | DeepEP (`auto`) |
|---|---|---|---|
| 32  | **1224** / **25.9** | 1170 / 27.1 | 274 / 115.9 |
| 64  | **2161** / **29.3** | 2110 / 29.4 | 732 / 86.0 |
| 256 | **5929** / 41.9 | 5888 / **41.3** | 2874 / 85.6 |
| 512 | 8361 / 58.1 | **8939** / **53.7** | 6685 / 73.0 |

**pure TP ≈ baseline, and both beat DeepEP by a wide margin** — 4.5× at conc 32, narrowing to 1.3×
at conc 512 as DeepEP's fixed per-layer cost finally amortises. Same ordering as the 2P2D decode
sweep in Part 2, on the same hardware in a different topology.

The DeepEP rows here are `--deepep-mode auto`, which leaves the per-batch kernel choice unpinned
(caveat 2). That turns out to cost DeepEP most of its conc-32 deficit: pinning `low_latency` doubles
that row to **554 tok/s / 54.5 ms**, cutting the gap to pure TP from 4.5× to 2.2× and the TPOT gap
from 4.5× to 2.1×. DeepEP still loses decode at this scale, but by half as much as the `auto` column
suggests. See [Pinning `--deepep-mode`](#pinning---deepep-mode) for the full table and the four
settings it takes.

## Pinning `--deepep-mode`

Pinning the mode is the standard advice for making a DeepEP benchmark attributable to one kernel
path. On a **colocated** server it is not free, and for `low_latency` it is not even possible
without changing the workload — worth knowing before you follow that advice.

### `low_latency` takes four settings, not one

The low-latency dispatch caps tokens **per rank per call** at
`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`, whose default is **128**. A decode batch stays
under that. A prefill chunk does not: `chunked_prefill_size` defaults to 8192. On a colocated server
the same ranks serve both stages, so `DEEPEP_MODE=low_latency` starts, answers `/health`, and then
**every rank dies on the first real request**:

```
RuntimeError: Failed: Assertion error /opt/DeepEP/csrc/deep_ep.cpp:1553
  'x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank'
```

Measured on 2× p5en, 2026-07-28: server ready at 05:31:22, first request at 05:33:33, 8/8 ranks
down, container exit 137. Note the failure mode — it is *not* a startup error, so a health check
passes and the crash looks like it belongs to whatever request happened to arrive first.

Raising that cap is not one setting but a chain of four, each of which fails **after** `/health`
goes green, on the first real request. `recipe/serve.sh` applies them all when
`DEEPEP_MODE=low_latency`, using `LL_MAX_TOKENS` (default 512) as the cap:

| # | Setting | Constraint | Symptom if unset |
|---|---|---|---|
| 1 | `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | ≤ 1024 — DeepEP's internode low-latency dispatch uses `FINISHED_SUM_TAG=1024` and needs the per-rank-pair token count below it (`deepep.py:339`) | `deep_ep.cpp:1553` assertion, all ranks |
| 2 | `--chunked-prefill-size` | ≤ the cap | same assertion, from the prefill path |
| 3 | `NVSHMEM_QP_DEPTH` | ≥ (cap + 1) × 2, asserted at `deep_ep/buffer.py:601`. Default is **1024**, which only covers a cap of 511 | bare `AssertionError` at dispatch |
| 4 | `--mem-fraction-static` | lowered — the low-latency RDMA buffers scale with the cap; at 1024 a single `low_latency_dispatch` allocation is **1.75 GiB** | `torch.OutOfMemoryError` during scheduler init |

Constraints 1 and 3 squeeze from opposite sides, and 4 prices the result: the cap must be ≤ 1024,
`NVSHMEM_QP_DEPTH` must be ≥ 2050 to allow that, and the buffers then need ~1.75 GiB per rank of
headroom that `--mem-fraction-static 0.85` does not leave. On 2× p5en (141 GB H200) a cap of 1024
OOMs at 0.85; **512 with `--mem-fraction-static 0.75`** is what runs, and 512 still covers a
concurrency-512 decode batch. That pair is what `recipe/serve.sh` defaults to for this mode — not the
1024 maximum, which is reachable in principle but not on 141 GB H200 at any mem-fraction we tried.

Constraint 2 is a real workload change — prefill in cap-sized chunks instead of 8192 — so **a
`low_latency`-pinned colocated server measures decode and nothing else.** Its prefill numbers are not
comparable to any other row here. `serve-pd.sh` needs none of this, because there the decode role
never prefills.

This is also the answer to "why does `auto` exist" — it is not indecision. It is the only setting
that serves both stages on one server without capping prefill, over-allocating RDMA buffers, or
crashing.

### `normal` is slower than `auto` at every prefill point

`recipe/benchmark.sh prefill` against a `DEEPEP_MODE=normal` server, against the same points under
`auto`. TTFT ms (lower better) / input tok/s (higher better):

| Input × conc | DeepEP `auto` | DeepEP `normal` (pinned) | `normal` vs `auto` |
|---|---|---|---|
| 1024 × 1  | **270** / **3.7k** | 480 / 2.1k | −43% tok/s |
| 4096 × 1  | **347** / **11.5k** | 445 / 9.2k | −20% |
| 8192 × 1  | **297** / **27.2k** | 511 / 16.0k | −41% |
| 4096 × 8  | **816** / **37.9k** | 950 / 31.9k | −16% |
| 4096 × 32 | **1905** / **55.9k** | 2026 / 51.7k | −8% |

This is the opposite of what "pin the high-throughput mode for prefill" predicts, and it is
consistent at all five points, so it is not noise. The most likely reason is that `auto` is not
choosing between two kernels at random — SGLang's per-batch selection is *also* free to keep the
low-latency path warm and to avoid `normal`'s dispatch setup on batches too small to amortise it,
which is most of this sweep. The gap narrows monotonically as concurrency rises (−43% at conc 1 to
−8% at conc 32), exactly as an amortised-fixed-cost story predicts, so at the concurrency 256 the
2P2D sweep reaches, pinned `normal` would plausibly catch up or win. Not measured here.

**Practical reading:** pinning `normal` buys attributability at a measurable throughput cost below
conc ~32. Quote `auto` for "what a deployment gets" and `normal` for "what the high-throughput
kernel does", and do not treat them as interchangeable.

### `low_latency` decode: faster per token below conc ~64, slower overall above it

Measured with the four settings above (cap 512, `--chunked-prefill-size 512`,
`NVSHMEM_QP_DEPTH=1026`, `--mem-fraction-static 0.75`), against the same points under `auto`. Output
tok/s (higher better) / mean TPOT ms (lower better) / mean TTFT ms:

| Concurrency | DeepEP `auto` | DeepEP `low_latency` (pinned) | tok/s delta |
|---|---|---|---|
| 32  | 274 / 115.9 / 505 | **554** / **54.5** / 1682 | **+102%** |
| 64  | 732 / 86.0 / 777 | **1021** / **59.3** / 1728 | **+40%** |
| 256 | **2874** / 85.6 / **1841** | 2634 / **83.0** / 7319 | −8% |
| 512 | **6685** / 73.0 / **1752** | 4371 / 96.2 / 10683 | −35% |

So pinning `low_latency` does exactly what its name says at small batch — it **halves TPOT at
concurrency 32** (115.9 → 54.5 ms) and doubles throughput — and then loses badly at large batch. The
crossover is between 64 and 256.

Two separate effects are stacked in those rows, and the TTFT column separates them:

- **The kernel genuinely helps decode at small batch.** At conc 32–64 the low-latency dispatch is
  the whole story: TPOT drops 47% and 31%, and p99 ITL drops even harder (298 → 54 ms, 316 → 59 ms)
  — the `auto` server's per-batch mode switching is what produces those p99 spikes, and pinning
  removes them. If you care about tail latency, this is the most convincing pair of numbers on this
  page.
- **The 512-token prefill chunk cap dominates at large batch.** TTFT rises from 1841 ms to 7319 ms
  at conc 256 and from 1752 ms to 10683 ms at conc 512 — 4× and 6×. Each request's 256-token prompt
  is fine, but 512 concurrent prompts must now be chunked through a 512-token-per-call dispatch
  instead of 8192, so requests queue in prefill. In a decode sweep TTFT is still counted in
  end-to-end throughput, so the tok/s regression at conc 256/512 is mostly **the constraint-2 side
  effect, not the decode kernel losing**. TPOT at conc 256 is still marginally better than `auto`
  (83.0 vs 85.6); only at 512 does TPOT itself regress (96.2 vs 73.0), where the cap is also
  squeezing the decode batch.

**This is the concrete argument for PD disaggregation.** On a colocated server the two effects are
inseparable: you cannot take the low-latency kernel without taking the prefill chunk cap. Split the
roles and you can — `serve-pd.sh` pins `low_latency` on decode nodes that never prefill, so the
conc-32 gain is available with no TTFT penalty at all. Every row above is an argument for not
running colocated if decode latency matters.

Note the `auto` rows were measured on the P5EN-1/2 pair and the `low_latency` rows on P5EN-3/4 —
identical instance type, image and flags otherwise, but not the same silicon. Per caveat 1, treat
the conc-32 doubling as real (the effect is far larger than any node-to-node spread seen elsewhere
in these runs) and the −8% at conc 256 as within noise.

---

# Part 2 — 4-node 2P2D PD-disaggregated (matches `recipe/serve-pd.sh`)

Prefill/decode-disaggregated serving of DeepSeek-R1 671B FP8, sweeping the four ways to parallelise
the MoE layer, and separating **prefill** from **decode** because the two stages want opposite
things from the all-to-all.

**Cluster:** 4× `p5en.48xlarge` (8×H200 141 GB, `sm_90`), us-east-2, 16 EFA NICs/node, iface
`enp71s0`. Date: 2026-06-24. SGLang 0.5.13.post1. Benched through the router with `--pd-separated`
and `--random-range-ratio 1` (so these are fixed-length, like the 2026-07-28 colocated round).

```
  Prefill role (TP=16 / EP=16)          Decode role (TP=16 / EP=16)
  ┌──────────┐  ┌──────────┐            ┌──────────┐  ┌──────────┐
  │ node 1   │──│ node 2   │  KV cache  │ node 3   │──│ node 4   │
  │ rank 0   │  │ rank 1   │ ─────────▶ │ rank 0   │  │ rank 1   │
  └──────────┘  └──────────┘  Mooncake  └──────────┘  └──────────┘
        ▲                     over EFA        ▲
        └──────── sglang_router --pd-disaggregation ────────┘
```

KV cache moves prefill→decode over **EFA RDMA** via Mooncake, which requires
[`MOONCAKE_PROTOCOL=efa`](#required-setting-mooncake_protocolefa) — omit it and Mooncake silently
falls back to TCP sockets, which invalidates every number in this part.

## The four configurations

| Label | SGLang flags | What the MoE layer does |
|---|---|---|
| **DeepEP** | `--ep-size 16 --moe-a2a-backend deepep` | DeepEP dispatch/combine kernels over NVSHMEM-libfabric/**EFA** — this sample's build |
| **UCCL-EP** | `--ep-size 16 --moe-a2a-backend deepep` on the UCCL image | Same DeepEP-compatible API, but UCCL's own RDMA proxy engine over EFA (`deep_ep_wrapper`) |
| **baseline** | `--ep-size 16 --moe-a2a-backend none` | Expert-parallel, but SGLang's ordinary fused-MoE all-to-all over NCCL |
| **pure TP** | `--tp-size 16`, **no** `--ep-size` | No expert parallelism at all: MoE weights are TP-sharded, the only cross-GPU traffic is the TP all-reduce |

**The UCCL-EP image is not in this repo.** UCCL's kernels are packaged here by
[`uccl-ep-benchmark`](../../../../../micro-benchmarks/expert-parallelism/uccl-ep-benchmark) (a
kernel harness with no serving engine) and combined with *vLLM* by
[`dsv3-uccl-nixl`](../../../vllm/dsv3-uccl-nixl). The rows below came from a third image: an SGLang
base plus UCCL's `ep/deep_ep_wrapper`, which presents UCCL under DeepEP's Python API so
`--moe-a2a-backend deepep` drives it unchanged. Reproducing them means building that image — this
sample does not ship it. Its launch requirements also differ from DeepEP's; see
[UCCL-EP launch configuration](#uccl-ep-launch-configuration-that-actually-matters).

## Prefill — input 1K→64K, adaptive concurrency

`--random-output-len 1`, so TTFT ≈ pure prefill time. Concurrency is swept per input length (high
at 1K, down to 1–2 at 64K, because a 64K request already fills the prefill batch).

### Peak input throughput per input length

tok/s, higher better; the concurrency that achieved it in parentheses. **Bold = best in row.**

| Input | pure TP | baseline | DeepEP | UCCL-EP |
|---|---|---|---|---|
| 1K  | 48.2k (512) | 53.8k (512) | **161.5k** (256) | 51.5k (512) |
| 4K  | 43.6k (128) | 42.7k (128) | **55.5k** (128) | 42.6k (128) |
| 16K | 80.2k (32) | 75.8k (32) | **84.0k** (32) | 69.2k (32) |
| 32K | 46.1k (8) | 42.2k (1) | **52.3k** (8) | 42.1k (8) |
| 64K | 52.7k (1) | 54.4k (2) | **60.0k** (2) | 50.9k (2) |

**DeepEP wins at every input length**, and the margin is enormous where the batch is largest and
the tokens smallest: at 1K input it sustains 161.5k input tok/s against ~48–54k for everything
else. That is the regime where DeepEP's high-throughput dispatch is doing what it was designed
for — one fused dispatch keeps all 16 GPUs fed, while the other three pay a TP all-reduce (or an
ordinary all-to-all) per layer. As input length grows the batch is filled by fewer, longer
sequences, the all-to-all shrinks relative to attention, and the advantage narrows to ~10–30%.

**pure TP ≈ baseline, with UCCL-EP a few percent behind both.** Among the three non-DeepEP
configurations the spread is small: pure TP edges 4K/16K/32K, baseline edges 1K/64K, and UCCL
trails at every input length. Prefill (the high-throughput `normal` dispatch) is UCCL's known soft
spot, matching its kernel-level combine/FP8-dispatch numbers.

### Full sweep at 1K input — where the knee is

TTFT ms / input tok/s.

| conc | pure TP | baseline | DeepEP | UCCL-EP |
|---|---|---|---|---|
| 1   | 344 / 3.0k  | **199** / 5.1k  | 244 / 4.2k  | 480 / 2.1k |
| 8   | 912 / 8.9k  | 735 / 11.0k | **617** / **13.1k** | 749 / 10.8k |
| 64  | 2231 / 27.1k | 2110 / 29.3k | **711** / **87.0k** | 2237 / 28.0k |
| 256 | 4310 / 43.8k | 4189 / 44.3k | **1399** / **161.5k** | 4480 / 43.0k |
| 512 | 6766 / 48.2k | 6233 / **53.8k** | 5242 / 68.5k | 6442 / 51.5k |

The conc-64 row is the interesting one: DeepEP holds TTFT nearly flat from conc 8→64 (617→711 ms)
while the other three triple it (735→2110 ms). DeepEP saturates at **conc 256**; past it (conc
512) throughput falls back to 68.5k as the prefill role overloads, so 256 is the operating point,
not 512.

All four completed every point at full success, including 64K×conc2 (4/4).

### Prefill limits found

- **128K input fails.** Prefill computes fine, but the Mooncake/EFA transfer of a 128K-token KV
  cache times out: `EFA submitSlicesOnPeer: CQ drain wr_depth=256, max=256`. Raising `MC_MAX_WR`
  or chunking the KV transfer would be the thing to try. **The 2P2D prefill ceiling here is ~64K
  input**, and it is a KV-transfer ceiling, not a compute one.
- **64K at concurrency ≥4** exhausts the 2-node prefill KV pool. 64K is single- or
  low-concurrency territory.

## Decode — output length and concurrency sweeps

Short input (256), long output, so TPOT dominates. Two axes.

### A. Decode-compute throughput at fixed concurrency 32

Output tok/s / mean TPOT ms. **Bold = best in row.**

| Output len | baseline | pure TP | UCCL-EP | DeepEP |
|---|---|---|---|---|
| 256  | 827 / 27.3 | **1142** / 26.1 | 721 / 32.6 | 531 / 49.1 |
| 512  | 1148 / 27.1 | **1177** / 25.9 | 943 / 32.7 | 637 / 49.2 |
| 1024 | 1157 / 27.3 | **1210** / 26.1 | 954 / 32.9 | 642 / 49.3 |
| 2048¹ | 911 / 26.1 | **967** / 24.5 | 746 / 31.9 | 494 / 48.3 |

¹ conc 24, not 32, to bound runtime.

### B. Concurrency ramp at output 512

Output tok/s / mean TPOT ms. **Bold = best in row.**

| conc | baseline | pure TP | UCCL-EP | DeepEP |
|---|---|---|---|---|
| 16  | 465 / 33.7 | 481 / 32.5 | **496** / **31.1** | 332 / 47.3 |
| 32  | 1148 / 27.1 | **1177** / **25.9** | 943 / 32.7 | 637 / 49.2 |
| 64  | 2062 / 29.5 | **2114** / **29.4** | 1631 / 37.2 | 1151 / 54.0 |
| 96  | 2773 / 33.0 | **2839** / **32.6** | 2385 / 38.1 | 1654 / 56.4 |
| 128 | **3626** / 33.5 | 3546 / 34.1 | 3138 / 39.2 | 2165 / 57.1 |

**Decode ranking: pure TP ≈ baseline > UCCL-EP > DeepEP.** The two TP-MoE configurations are
essentially tied and fastest — with no all-to-all in the decode path, attention and MoE both run
as plain TP. **UCCL-EP is a clear third but healthy** (~13% below baseline at conc 128, TPOT 39 vs
33 ms). **DeepEP is last** at 2165 tok/s / TPOT 57 ms, about 1.7× the others' TPOT: its per-layer
low-latency dispatch/combine is not amortised when each of 16 ranks already owns 16 of 256
experts and most of the fan-out stays on NVLink.

The same ordering, on the same hardware, appears in the colocated Part 1 decode tables — two
topologies agreeing that 16-GPU decode is not DeepEP's regime.

All four scale cleanly to concurrency 128 with no wedge. (Earlier runs hit a ceiling well below
that, which turned out to be the KV transport falling back to TCP — see
[`MOONCAKE_PROTOCOL=efa`](#required-setting-mooncake_protocolefa).)

## DP-attention: the lever that reverses the decode ranking

SGLang's high-throughput option runs **attention data-parallel** across all 16 ranks
(`--dp-size 16 --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1`). It is
**orthogonal to the MoE backend**, so all four were measured with and without it.

### Decode at concurrency 128, output 512

Output tok/s / TPOT ms.

| Backend | non-DP | **+ DP-attention** | DP effect |
|---|---|---|---|
| **UCCL-EP** | 3138 / 39.2 | **4094 / 28.5** | **+30%** ✅ — fastest decode of every configuration tested |
| **DeepEP** | 2165 / 57.1 | 2691 / 45.4 | **+24%** ✅ (still the lowest absolute) |
| baseline | 3626 / 33.5 | 3317 / 36.4 | **−9%** ❌ |
| pure TP | 3546 / 34.1 | 3153 / 38.3 | **−11%** ❌ |

### Decode ramp, DP variants (output 512)

| conc | UCCL+DP | DeepEP+DP | baseline+DP | pure TP+DP |
|---|---|---|---|---|
| 16  | **652** / 23.5 | 410 / 38.2 | 466 / 33.5 | 470 / 33.1 |
| 32  | **1175** / 25.8 | 763 / 40.7 | 899 / 34.6 | 875 / 35.4 |
| 64  | **2167** / 27.5 | 1427 / 43.4 | 1665 / 35.7 | 1613 / 37.1 |
| 96  | **3084** / 28.2 | 2035 / 44.7 | 2512 / 36.2 | 2406 / 38.0 |
| 128 | **4094** / 28.5 | 2691 / 45.4 | 3317 / 36.4 | 3153 / 38.3 |

**DP-attention helps EP-MoE decode and hurts TP-MoE decode.** With expert parallelism the
per-rank token batch is small; making attention data-parallel (each DP rank owning whole
sequences) removes redundant attention/KV work and lets the all-to-all coordinate over larger
per-rank chunks. With TP-MoE, TP16 attention is already efficient at 16 GPUs, so `dp16` only adds
synchronisation and imbalance. The practical consequence: **turning DP on flips the decode
winner** from baseline to UCCL-EP.

### DP-attention hurts prefill, universally

Peak input throughput at 1K input, non-DP → +DP:

| Backend | non-DP | + DP | Effect |
|---|---|---|---|
| DeepEP | 161.5k | 59.6k | **−63%** |
| baseline | 53.8k | 34.8k | −35% |
| UCCL-EP | 51.5k | 37.0k | −28% |
| pure TP | 48.2k | 34.7k | −28% |

DP adds synchronisation with no upside on batches that are already large, and it is worst for
DeepEP, which is precisely the config whose prefill advantage comes from a big fused dispatch.
**Never enable DP-attention on the prefill role.**

### DP-attention gotchas (both are crashes, not slowdowns)

- **Pre-compile DeepGEMM first, on 2 nodes**, for any DeepEP-backed DP mode
  (`recipe/serve-pd.sh precompile`). DP+EP uses grouped-GEMM shapes (`num_groups=16`) that are
  absent from the normal cache, and their first-time JIT compile is slow enough to trip DeepEP's
  dispatch warmup timeout — `DeepEP error: timeout (dispatch CPU)` — killing the prefill server.
  Pre-warming the mounted `deep_gemm` cache fixes it, once per host. A single-node precompile
  cannot initialise 16-rank EP, so this must run on both prefill nodes. (UCCL does not hit this;
  its wrapper tolerates the cold compile. baseline/pure TP do not use DeepGEMM here at all.)
- **The decode role must pin `--cuda-graph-bs 128`** (with `--mem-fraction-static 0.78`).
  Otherwise each of the 16 DP ranks captures the full batch-size list `[1..512]` and OOMs during
  graph capture — the failure surfaces as `scheduler died`.

`recipe/serve-pd.sh` applies both automatically when `DP_ATTENTION=1`.

## Recommended configuration at this scale

| Role | Configuration | Why |
|---|---|---|
| **Prefill** | **DeepEP**, no DP-attention, operate at concurrency ~256 for 1K inputs | 161.5k input tok/s, 3× the alternatives; DP would cost 63% |
| **Decode** | **UCCL-EP + DP-attention**, `--cuda-graph-bs 128`, `--deepep-mode low_latency` | 4094 tok/s / TPOT 28 ms, the fastest decode measured (but see caveat 3) |
| **Decode**, same-image fallback | **baseline**, no DP-attention | 3626 tok/s / TPOT 34 ms — 13% behind, and needs no second image |

Running different backends per role means running different *images* per role, which a PD
deployment makes practical — the roles share nothing but the KV stream. If a single image is a
hard requirement, **DeepEP prefill + baseline (non-DP) decode** is the best same-image split
(161.5k / 3626), and **pure TP everywhere** is the best single-config answer if you want one
knob and no EP at all.

## UCCL-EP is instance-type sensitive

The same UCCL-EP build behaves very differently on `p5` (H100, **32** EFA NICs/node) than on `p5en`
(H200, 16 NICs/node). 2P2D decode, input 256 / output 128, same sweep on both:

| conc | UCCL on `p5` (H100) | UCCL on `p5en` (H200) | ratio |
|---|---|---|---|
| 8  | 42.9 tok/s, TPOT 163 ms | **186.5** tok/s, TPOT **25.4** ms | 4.3× |
| 16 | 95.9, 160 ms | **323.4**, **27.3** ms | 3.4× |
| 32 | 188.0, 163 ms | **451.8**, **26.6** ms | 2.4× |

**UCCL-EP is 2.4–4.3× faster on p5en/H200 than on p5/H100, and the TPOT difference is a factor of
six** (26 vs 160 ms). On p5 it looks like a broken backend, a distant third behind even DeepEP; on
p5en it is comfortably ahead of DeepEP and, with DP-attention, the fastest decode in this whole
document. The gap is consistent with UCCL's EFA path not being tuned for p5's NIC topology (32 EFA
NICs for 8 GPUs, a different NIC/PCIe-switch layout than p5en's 16) rather than with any defect in
its kernels — its correctness checks pass identically on both, and the same
[`ep-backend-comparison`](../../../../../micro-benchmarks/expert-parallelism/ep-backend-comparison)
data shows the UCCL-vs-NVSHMEM ordering flipping by GPU generation as well.

**So: choose the EP backend per instance type, not once and globally.** On p5en, DeepEP for prefill
and UCCL-EP for decode; on p5, do not assume the same answer.

This is the only H100 result kept on this page, and it is kept because it is a *warning* rather
than a measurement to quote: the rest of the p5 2P2D sweep predated `MOONCAKE_PROTOCOL=efa`, so its
DeepEP and baseline rows ran with KV over TCP while the UCCL launcher always set it. That made
UCCL's p5 *prefill* rows look 3–4× dominant, an artifact of the transport rather than the backend —
and the ordering reverses on p5en once all four run on EFA (DeepEP 161.5k vs UCCL 51.5k at 1K). The
decode ratios above survive that confound only because they compare UCCL to UCCL, both with
`MOONCAKE_PROTOCOL=efa` set. **Do not carry any other p5 conclusion forward.**

## Required setting: `MOONCAKE_PROTOCOL=efa`

**Every 2P2D number on this page was measured with the KV cache on EFA RDMA, which requires
`MOONCAKE_PROTOCOL=efa`.** On EFA hardware there is no reason to run the KV path any other way, so
treat this as part of the deployment, not as a tuning knob.

It matters because **omitting it is a silent fallback to TCP sockets, not an error.** The server
comes up, returns correct tokens, and passes a smoke test; under a sustained high-concurrency
prefill burst it then deadlocks with `KVTransferError ... session is not alive`. The only evidence
of which transport you are on is one line in the server log:

```
transfer_engine_py.cpp:241] Installing TCP transport (auto_discover disabled in EFA build)   # WRONG
tcp_transport.cpp:678]      TcpTransport: listen on port 16705

transfer_engine_py.cpp:100] Using default malloc/free for protocol: efa                      # right
efa_transport.cpp:1025]     EfaTransport: Initialized EFA device rdmap160s0 ...
```

So **check the log rather than assuming**, on every role, before benchmarking:

```bash
docker logs r1-pd-prefill 2>&1 | grep -E 'EfaTransport|Installing TCP transport'
```

`recipe/serve-pd.sh` sets the variable and prints this grep. One plausible-looking explanation for
the deadlock is worth ruling out up front, because it cost debugging time here: it is **not**
NVSHMEM contention — the wedge reproduces with `MOE_BACKEND=tp`, which never loads DeepEP. The KV
path is Mooncake's alone, so the fix is Mooncake's alone.

## UCCL-EP launch configuration that actually matters

Reproducing the UCCL rows needs four things that DeepEP does not. Getting them wrong produces a
SIGSEGV during prefill startup, which reads like a UCCL bug and is not:

1. **`--deepep-mode normal` on prefill and `low_latency` on decode — not `auto`.** This is the one
   that matters most: `auto` drives UCCL's prefill down a path that segfaults.
2. **`MOONCAKE_PROTOCOL=efa`** — same as above.
3. **`--privileged` on the container.** UCCL registers GPU memory for RDMA through the
   dma-buf/ibverbs path, which needs the privilege. DeepEP/NVSHMEM use GDRCopy and do not.
4. **`--mem-fraction-static 0.70` on decode.** UCCL's `low_latency` RDMA buffers are large; 0.82
   OOMs during CUDA-graph capture at `uccl_ep.cc:289`.

Plain TP16/EP16 works — no `--dp-size`/`--enable-dp-attention` is required (though DP is what
makes UCCL the decode winner, per above). The compute kernels were never implicated:
`compute-sanitizer` reports 0 errors from UCCL's `ep.abi3.so`, and its intranode and internode
kernel tests pass.

---

# Kernel-level context (why the EFA port is nonetheless sound)

The serving numbers above are a *scale* story. The kernel numbers are the *enablement* story — on
H200, 2 nodes, 4096 tokens × hidden 7168:

| Metric | DeepEP official (H800 + CX7 IB 400Gb/s) | This port (H200 + EFA) |
|---|---|---|
| 16-EP dispatch (RDMA) | 43 GB/s | **62.7 (FP8) / 72.3 (BF16)** |
| 16-EP combine (RDMA) | 43 GB/s | **59.4** |
| 32-EP dispatch (RDMA) | 58 GB/s | 47.1 (FP8) / 51.5 (BF16) |
| 32-EP combine (RDMA) | 57 GB/s | 53.2 |

Different hardware on each side (H200 + EFA vs H800 + CX7), so read this as a sanity band rather
than a controlled A/B. The point is that **EFA delivers IB-class dispatch/combine bandwidth** for
these kernels. Full kernel tables, the `NVSHMEM_NETDEVS_POLICY` ablation, and the cross-generation
deltas live in the `deepep-benchmark` and `ep-backend-comparison` directories.

Smoke results from `recipe/run-kernel-test.sh` against the image built from the committed
Dockerfile, on `p5en.48xlarge` (8×H200) on 2026-07-28 — these are the pre-flight checks the README
tells you to run, not a benchmark:

| Test | Correctness | Bandwidth |
|---|---|---|
| `intranode` (NVLink only, no NVSHMEM, 8 ranks) | **24/24 passed** | dispatch 316.6 GB/s (FP8) / 330.4 (BF16), combine 323.8 — all NVLink |
| `low_latency` (NVSHMEM, 8 ranks) | passed | dispatch 180–189 GB/s (~40 µs), combine 218–226 GB/s (~65 µs) |
| `internode` (NVSHMEM/EFA, 2 nodes, 16 EP ranks) | **32/32 passed on both ranks** | dispatch FP8 **62.1** GB/s RDMA / 202.8 NVL; BF16 **71.1** / 232.2; combine **59.2** / 193.3 |

The internode figures land within ±2% of the 16-EP column in the table above (62.7 / 72.3 / 59.4),
which was measured on different instances on a different date — so the EFA transport reproduces,
and the `internode` row of that table is no longer inherited from an earlier image.

Note the shape of the three rows: `intranode` is pure NVLink at ~320 GB/s, `low_latency` on one node
never leaves NVLink either, and only `internode` puts bytes on EFA — which is why it is the only row
that speaks to the port at all. Run all three anyway; the first two failing tells you the image is
broken before you spend two nodes finding out.

# Blackwell: expect DeepEP to lose at 2 nodes

No serving run on Blackwell is in this repo. But the question comes up — *"we benchmarked DeepEP
vs the NCCL all-to-all on 2× B300 and DeepEP was slower in every configuration; is that
expected?"* — and the answer from the data that **is** here is **yes, at 2 nodes it is expected,
and it is not an EFA problem.** Reported shape of such a result, for reference: output throughput
−7% to −26%, median TTFT +17% to +82%, P99 ITL 1.2–1.9 s vs 0.8–0.9 s, with the `normal` (HT) mode
the slowest where it ran at all — at TP16/EP16 across two nodes, 8K input / 1K output, concurrency
128.

Three reasons this is the expected result rather than a misconfiguration:

1. **16 ranks is DeepEP's worst case, on any GPU.** Every table on this page says so. With 256
   experts over 16 ranks each GPU owns 16 experts, so most of the fan-out never leaves NVLink and
   the per-layer dispatch/combine launches cost more than they save. The colocated decode sweep has
   DeepEP at 0.23–0.75× the baseline's throughput and 1.4–4.3× its TPOT; the 2P2D sweep has
   0.55–0.71× throughput at 1.4–1.8× TPOT. A −7% to −26% *aggregate* regression on a mixed 8K/1K
   workload is **milder than what we measure on Hopper**, not worse.
2. **The published B300 kernel numbers are healthy, which localises the gap above the transport.**
   At 2 nodes / 16 ranks on `p6-b300`, DeepEP-over-EFA dispatch/combine is **126.6 / 106.4 GB/s**
   — the best of the three backends there, and *above* the NCCL all-to-all's 104.9 GB/s at matched
   payload
   ([`ep-backend-comparison`](../../../../../micro-benchmarks/expert-parallelism/ep-backend-comparison/RESULTS.md)).
   So the fabric and the kernels are fine at that scale; the serving regression is per-layer
   launch/scheduling overhead and MoE-runner choice, not bytes on the wire.
3. **`normal`/HT being the slowest at decode-heavy concurrency is by design.** HT is the
   high-throughput dispatch for large prefill batches; on a 1K-output workload the run is
   TPOT-dominated, where `low_latency` is the intended mode. `--deepep-mode auto` landing between
   the two is consistent with that, and matches the ordering here.

## Confounds worth eliminating before concluding anything from such a run

- **Confirm NCCL is actually on EFA, on both arms** — check for `NET/OFI` rather than `NET/Socket`
  in `NCCL_DEBUG=INFO` output instead of assuming. Whether it needs `NCCL_NET_PLUGIN=ofi` depends on
  the EFA installer: 1.47/1.48 ship the plugin under NCCL's default `libnccl-net.so` name and it
  auto-loads, while the image here ships only `libnccl-net-ofi.so` and needs the var. If NCCL *did*
  land on sockets it penalises the NCCL arm, making a measured DeepEP loss a lower bound — but do
  not assume that direction without the log line. See caveat 4.
- **The MoE runner is not held constant** in the usual formulation of this comparison: DeepEP rows
  run `--moe-runner-backend deep_gemm` while the no-DeepEP rows resolve `auto` to
  `flashinfer_trtllm` on Blackwell. That is two variables — the all-to-all *and* the grouped-GEMM
  implementation — and TRT-LLM's Blackwell MoE kernels are heavily tuned. Re-run DeepEP against
  `--moe-runner-backend flashinfer_trtllm` (or the baseline against `deep_gemm`) before attributing
  the delta to the all-to-all.
- **DeepGEMM JIT warmup.** `deep_gemm` compiles shapes on first use; 32 requests at concurrency 16
  is not necessarily enough to cover the shapes a concurrency-128 run then hits. It inflates early
  TTFT and P99 ITL specifically — the two metrics that move most in reports like the one above.
  Pre-warm with `python3 -m sglang.compile_deep_gemm` on **both** nodes (`recipe/serve-pd.sh
  precompile <rank>` does this). On this sample's image the cache is fully populated during server
  startup, before the health endpoint answers, so it is not a factor in the tables here — verify
  the same on yours rather than assuming it.
- **`--deepep-mode auto` on a mixed workload.** Pin `normal` and `low_latency` separately
  (`DEEPEP_MODE=` for `recipe/serve.sh`, automatic per role in `recipe/serve-pd.sh`), or better,
  disaggregate: this page's own answer to "which backend wins" is *neither, per stage*.
- **Benchmark harness defaults.** `bench_serving --random-range-ratio` defaults to **0.0**, which
  silently halves the token volume and lets short requests drain the batch; `--warmup-requests`
  defaults to **1**, which only exercises the batch-1 path. On one live server the ratio flag alone
  moved a nominal 256/512/conc-64 point from 374 to 1127 tok/s (see Part 1). Set both explicitly on
  every arm.
- **An HT-path hang that does not reproduce** is worth keeping an eye on rather than dismissing.
  At ≥128 ranks the NVSHMEM-libfabric host proxy exhausts libfabric retries
  (`EAGAIN` in `nvshmemi_process_multisend_rma`) and kills a different pair of ranks each run;
  that is a documented statistical fan-out limit, not a bad node. At 16 ranks it should not fire,
  but a non-reproducing hang on the HT path has the same signature.

**The load-bearing point for a large fleet: 2 nodes measures the wrong thing.** DeepEP is built
for EP domains where experts are spread thin enough that every token crosses the fabric. The
kernel-level scaling out to 256 ranks on `p6-b300` is already characterised, and it says the
useful envelope is ~64–160 ranks flat, with hard implementation caps past that (HT: 160 PEs at
`deep_ep.cpp:158`; low-latency: between 64 and 128 PEs). Training deployments stay inside it by
running EP32/EP64 groups within a larger world. A production-EP-width run — EP32 or EP64, not
EP16 — is the measurement that decides this, and it is a different experiment from the one above.

# Reproduce

## Colocated, 2 nodes

```bash
source setup/env_vars

# Pin the DeepEP kernel mode to the stage you are about to measure.
DEEPEP_MODE=normal      MOE_BACKEND=deepep recipe/serve.sh 0   # node 1, then benchmark prefill
DEEPEP_MODE=normal      MOE_BACKEND=deepep recipe/serve.sh 1   # node 2
# ...or, for decode:
DEEPEP_MODE=low_latency MOE_BACKEND=deepep recipe/serve.sh 0
DEEPEP_MODE=low_latency MOE_BACKEND=deepep recipe/serve.sh 1

curl -s localhost:30000/health && echo OK
recipe/benchmark.sh prefill      # or: recipe/benchmark.sh decode
```

Then repeat with `MOE_BACKEND=baseline` and `MOE_BACKEND=tp`. `recipe/serve.sh` derives
`NVSHMEM_DISABLE_CUDA_VMM` from `DEEPEP_MODE` — `normal` needs VMM off or NVSHMEM's
topology/transport-map init fails, `low_latency`/`auto` need it on or the RDMA-buffer `cudaMemset`
fails with `invalid argument` at `deep_ep.cpp:371`. Getting it wrong is a startup failure, not a
slow server.

## 2P2D, 4 nodes

Fill in the PD block of `setup/env_vars` first. The UCCL rows need a separate image that this
sample does not ship (see [the four configurations](#the-four-configurations)); once built, point
`IMAGE_URI` at it and add `UCCL=1`.

```bash
source setup/env_vars

# Optional, and REQUIRED for DeepEP + DP-attention: warm the DeepGEMM cache (both prefill nodes).
DP_ATTENTION=1 recipe/serve-pd.sh precompile 0     # on prefill node 0
DP_ATTENTION=1 recipe/serve-pd.sh precompile 1     # on prefill node 1

# Servers: one invocation per node, role and rank varying.
MOE_BACKEND=deepep recipe/serve-pd.sh serve prefill 0    # node 1
MOE_BACKEND=deepep recipe/serve-pd.sh serve prefill 1    # node 2
MOE_BACKEND=deepep recipe/serve-pd.sh serve decode  0    # node 3
MOE_BACKEND=deepep recipe/serve-pd.sh serve decode  1    # node 4

recipe/serve-pd.sh router                                # on the router host
curl -s localhost:8000/health && echo OK

# Sweeps, through the router. Decode first: the prefill sweep's large batches can
# destabilise the 2-node prefill role, and you do not want that to cost the decode data.
recipe/benchmark-pd.sh decode
recipe/benchmark-pd.sh prefill

recipe/serve-pd.sh stop                                  # on every node
```

Then repeat with `MOE_BACKEND=baseline`, `MOE_BACKEND=tp`, and `DP_ATTENTION=1` for the DP rows.

# Open items

- **Re-run the colocated DeepEP rows with `--deepep-mode` pinned.** Both rounds in Part 1 used
  `auto`, which is why the low-concurrency DeepEP numbers do not reproduce between them. This is
  the single largest loose end on this page.
- **Re-run the 2P2D sweep with the 2026-07-28 harness** (`--random-range-ratio 1` was already set,
  but `--warmup-requests` was not scaled) so Parts 1 and 2 share a harness exactly.
- **Add a cross-image decode control** — run `baseline` decode on both the DeepEP and UCCL images —
  so caveat 3 stops applying to this page's decode recommendation.
- **The per-role split recommendation is inferred, not measured.** DeepEP prefill and UCCL+DP
  decode were each measured in a same-backend-both-roles deployment; the mixed deployment was
  never run end to end. It should work — the roles only share the KV stream — but it is untested.
- **128K input needs a larger `MC_MAX_WR` or chunked KV transfer** to get past the EFA CQ-drain
  timeout.
- **Larger EP scale is the real question.** Everything here is 16 GPUs per role, where DeepEP's
  decode overhead cannot be amortised. The kernel scaling out to 32 nodes / 256 ranks is
  characterised in `ep-backend-comparison`; the *serving* crossover is not. For decode
  specifically, re-run on ≥8–16 nodes with a larger `--ep-size` and `--deepep-mode low_latency`
  pinned. On 2 nodes the decode crossover is not reachable — though DP-attention recovers ~25% of
  the gap for DeepEP, and ~30% for UCCL-EP.
