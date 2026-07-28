<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# DeepSeek-R1 on SGLang + DeepEP-over-EFA — measured results

End-to-end serving results for the image in this directory, all on **H200** (`p5en.48xlarge`), in
two topologies:

- **Colocated 2-node TP16/EP16** — [`recipe/serve.sh`](../recipe/serve.sh).
- **4-node 2P2D prefill/decode-disaggregated** — [`recipe/serve-pd.sh`](../recipe/serve-pd.sh), KV
  cache over EFA RDMA via Mooncake.

The two topologies reach different conclusions, so each is labelled and they are not read across as
one experiment. For **kernel-level** dispatch/combine bandwidth of the same build — including
8-to-32-node scaling and the NCCL/UCCL comparison — see
[`micro-benchmarks/expert-parallelism/deepep-benchmark`](../../../../../micro-benchmarks/expert-parallelism/deepep-benchmark)
and [`ep-backend-comparison`](../../../../../micro-benchmarks/expert-parallelism/ep-backend-comparison).

## Summary

**No single MoE backend wins both stages.**

| Stage | Winner | Margin |
|---|---|---|
| **Prefill** (large batches) | **DeepEP** | 30.8k input tok/s at 4K×conc32 colocated — **+13%** over the best non-EP path, **+24%** over UCCL-EP, and the gap widens with concurrency |
| **Decode** (small batches) | **UCCL-EP + DP-attention** | 4094 output tok/s at conc128, TPOT 28 ms in 2P2D — **1.5×** DeepEP+DP, **1.13×** the best non-DP config |

Prefill and decode run on separate node groups in a PD deployment, so you can act on that: **DeepEP
on the prefill role, UCCL-EP + DP-attention on the decode role.** That split is the most useful
result here and it exists only because the roles are disaggregated.

**On decode at 16 GPUs, DeepEP loses to everything** — the ordinary all-to-all, pure TP, and
UCCL-EP (the last of those confirmed in both topologies, under matched `--deepep-mode` and with a
[cross-image control](#cross-image-control-both-stages-transfer)). Expected at this scale, not a defect in the EFA port: with 16 ranks and 256 experts each
GPU owns 16 experts, so the fan-out is small and mostly intra-node NVLink, and DeepEP's per-layer
dispatch/combine cost is not amortised. DeepEP targets EP domains wide enough that every token must
cross the fabric; that crossover is not reachable on 2 nodes for decode.

**Every number below pins `--deepep-mode` to the stage being measured** — `normal` for prefill,
`low_latency` for decode — and runs with the radix prefix cache off. Both are measurement
requirements, not tuning: see [section 2](#2-pin---deepep-mode-to-the-stage-you-are-measuring) and
[section 5](#5-disable-the-prefix-cache-or-the-sweep-measures-the-cache).

**What the port establishes:** the DeepEP dispatch/combine kernels are correct and run at IB-class
bandwidth over EFA with no InfiniBand present, in a real DeepSeek-R1 server, in both topologies.
Before this port DeepEP could not use EFA at all — NVSHMEM offered only IBRC/IBGDA.

---

# How to measure this correctly

Four things must be set explicitly. Defaults elsewhere in the stack will silently give you numbers
that are not comparable to anything.

## 1. Fix the sequence lengths and scale the warmup

`bench_serving --random-range-ratio` **defaults to 0.0**, which samples every length uniformly from
`[0, len]` instead of using `len`. A nominal "256 in / 512 out at conc 64" point then pushes about
half the tokens, and short requests keep draining the batch so sustained concurrency lands near 46
instead of 64. Measured back-to-back on one live server, same nominal point, only that flag
differing: **374 tok/s at ratio 0.0 versus 1127 tok/s at ratio 1.0**, with mean E2E latency roughly
equal (32.2 s vs 29.1 s) — the 3× is the workload changing, not the server getting faster.

`--warmup-requests` **defaults to 1**, which only exercises the batch-1 path.

`recipe/benchmark.sh` sets `--random-range-ratio 1.0`, `--seed 42`, and scales `--warmup-requests`
to each point's concurrency. Use it rather than calling `bench_serving` directly.

## 2. Pin `--deepep-mode` to the stage you are measuring

`auto` lets SGLang pick per batch, which is what a deployment wants but makes a benchmark
unattributable — and the effect is large (2.0× at conc 32, see below). Pin it:

```bash
DEEPEP_MODE=normal      MOE_BACKEND=deepep recipe/serve.sh <rank>   # then benchmark prefill
DEEPEP_MODE=low_latency MOE_BACKEND=deepep recipe/serve.sh <rank>   # then benchmark decode
```

`recipe/serve-pd.sh` pins it per role automatically. Quote `auto` for "what a deployment gets" and
a pinned mode for "what that kernel does"; they are not interchangeable.

## 3. `low_latency` on a colocated server needs four coupled settings

The low-latency dispatch caps tokens **per rank per call**, and on a colocated server the same ranks
also serve prefill, where a chunk is `chunked_prefill_size` tokens. `recipe/serve.sh` applies all
four from `LL_MAX_TOKENS` (default **512**) when `DEEPEP_MODE=low_latency`:

| Setting | Value | Constraint |
|---|---|---|
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | `LL_MAX_TOKENS` | ≤ 1024 — DeepEP's internode low-latency dispatch uses `FINISHED_SUM_TAG=1024` and needs the per-rank-pair count below it |
| `--chunked-prefill-size` | `LL_MAX_TOKENS` | ≤ the cap, or prefill chunks overflow the dispatch |
| `NVSHMEM_QP_DEPTH` | `(cap + 1) × 2` | default 1024 only covers a cap of 511 |
| `--mem-fraction-static` | 0.75 | the low-latency RDMA buffers scale with the cap — 1.75 GiB per allocation at cap 1024 |

512 / 0.75 is what runs on 141 GB H200; the 1024 maximum OOMs at any mem-fraction tried. 512 still
covers a concurrency-512 decode batch. **All four fail after `/health` already returns 200**, on the
first real request, so a green health check proves nothing here.

The `--chunked-prefill-size` constraint is a real workload change (cap-sized prefill chunks instead
of 8192), so **a `low_latency`-pinned colocated server measures decode only** — its prefill numbers
are not comparable to anything. `serve-pd.sh` needs none of this: there the decode role never
prefills. That is also why `auto` exists — it is the only setting that serves both stages on one
server without capping prefill.

## 4. Confirm the transports rather than assuming

- **NCCL on EFA.** Check for `NET/OFI` (not `NET/Socket`) in `NCCL_DEBUG=INFO`. This image ships only
  `libnccl-net-ofi.so`, so it needs `NCCL_NET_PLUGIN=ofi` — set in the Dockerfile and asserted by
  `recipe/verify-image.sh`. EFA installers 1.47/1.48 also ship the plugin under NCCL's default
  `libnccl-net.so` name, where it auto-loads; do not infer the transport from the installer version.
- **KV cache on EFA**, for 2P2D only. Requires
  [`MOONCAKE_PROTOCOL=efa`](#required-setting-mooncake_protocolefa) — omitting it silently falls back
  to TCP — plus `--privileged` and `FI_HMEM=system,cuda`; see
  [The 2P2D KV path has three container-level requirements](#the-2p2d-kv-path-has-three-container-level-requirements).
- **Run the kernel tests first.** `recipe/run-kernel-test.sh intranode|low_latency|internode`. The
  first two are single-node and cheap; only `internode` puts bytes on EFA.

## 5. Disable the prefix cache, or the sweep measures the cache

`recipe/benchmark.sh` and `recipe/benchmark-pd.sh` pin `--seed 42`, so **every point that shares an
input length replays the same prompts**. With SGLang's radix prefix cache on, the conc-1 point
populates it and the conc-8/32 points that follow read their prefixes back instead of prefilling.
Measured on one live server, 4096-in conc-8: **27.2k input tok/s with the cache flushed vs 334k on an
immediate re-run of the identical point** — a 12× artefact. Nothing in the sweep output flags it, and
the hit rate is not stable run to run, so it does not cancel out across a comparison either.

Both serve scripts therefore pass `--disable-radix-cache` unconditionally. Confirm it took:

```bash
docker logs r1-deepep 2>&1 | grep -o "disable_radix_cache=[A-Za-z]*"   # want True
```

What it costs you, measured same-config back to back on the colocated decode sweep (input 256,
output 512):

| Concurrency | TPOT off/on (ms) | output tok/s off/on | TTFT off/on (ms) |
|---|---|---|---|
| 32  | 27.01 / 27.07 | 1156 / 1152 | 352 / 377 |
| 64  | 29.72 / 29.64 | 2076 / 2109 | 579 / 379 |
| 256 | 41.76 / 41.32 | 5736 / 5894 | 1465 / 1081 |
| 512 | 55.37 / 53.37 | 8466 / 9101 | 2540 / 1402 |

**TPOT is essentially immune (≤4%)**, `output tok/s` is inflated up to **7%** at conc 512 because it
includes TTFT, and **TTFT itself moves ~1.8×**. So a decode ranking survives the cache; a prefill
table does not. If you want to measure cache-hit behaviour, that is a separate experiment with varied
prompts, not a side effect of a fixed seed.

## Comparability limits of the tables below

1. **Single-seed smoke benchmarks.** One `bench_serving` pass per point — directional, not
   statistically tight. Low-concurrency prefill points (4–16 requests) are dominated by warmup.
2. **`--deepep-mode` is pinned everywhere**, `normal` for prefill and `low_latency` for decode. 2P2D
   pins it per role by construction. Older `auto` measurements have been dropped rather than mixed
   in: an `auto`-vs-pinned comparison understates DeepEP by about 2× at low concurrency.
3. **UCCL rows come from a different image**, bridged by running `baseline` on both — and the bridge
   holds on both stages. Decode agrees within ±1.8% tok/s across the whole ramp and prefill within
   ±1% at every point; see
   [Cross-image control](#cross-image-control-both-stages-transfer). So the DeepEP-vs-UCCL
   margins below are the backend, not the image.
4. **Server flags differ between topologies.** 2P2D used `--chunked-prefill-size 16384
   --watchdog-timeout 1200 --mem-fraction-static 0.82`; `recipe/serve.sh` takes SGLang's defaults
   (8192 / 300). The two topologies are not a controlled A/B for each other.
5. **Peak throughput is a saturation metric.** High-concurrency points have TTFT in the seconds;
   read the TTFT column if interactive latency matters.

---

# Part 1 — Colocated 2-node TP16/EP16 (`recipe/serve.sh`)

**Cluster:** 4× `p5en.48xlarge` (8×H200 141GB, sm_90), us-east-2, 16 EFA NICs/node, iface
`enp71s0`. SGLang 0.5.13.post1 + DeepEP `567632d` + NVSHMEM v3.7.0 (libfabric/EFA). 2 nodes, TP=16,
EP=16, `--mem-fraction-static 0.85` except where a `low_latency` section says 0.75. Date:
**2026-07-28**. All kernel correctness checks passed first.

The UCCL columns come from a second image on the same two nodes,
[`Dockerfile.uccl`](../Dockerfile.uccl): SGLang `v0.5.13.post1-cu130` + UCCL `f071f2e3` + Mooncake
`622104e`.

`pure TP` = `--tp-size 16` with no `--ep-size` (MoE weights TP-sharded, no dispatch/combine) =
`MOE_BACKEND=tp`. `Baseline` = EP with the ordinary fused-MoE all-to-all = `MOE_BACKEND=baseline`.
`DeepEP` = `MOE_BACKEND=deepep`.

## Prefill-only (`--random-output-len 1`, so TTFT ≈ prefill time)

TTFT ms (lower better) / input tok/s (higher better). **Bold = best in row.** DeepEP and UCCL-EP both
pin `--deepep-mode normal`, the prefill mode; all four columns have the radix prefix cache off (see
[Disable the prefix cache](#5-disable-the-prefix-cache-or-the-sweep-measures-the-cache)).

| Input × conc | pure TP | Baseline | DeepEP `normal` | UCCL-EP `normal` |
|---|---|---|---|---|
| 1024 × 1  | 100 / 10.1k | **98** / **10.2k** | 459 / 2.2k | 574 / 1.8k |
| 4096 × 1  | **177** / **23.0k** | 189 / 21.5k | 474 / 8.6k | 577 / 7.1k |
| 8192 × 1  | 377 / 21.6k | **349** / **23.3k** | 655 / 12.5k | 838 / 9.8k |
| 4096 × 8  | 1145 / 25.9k | 1158 / 26.8k | **1010** / **30.2k** | 1328 / 23.0k |
| 4096 × 32 | **4130** / 28.0k | 4304 / 26.8k | 3723 / **30.8k** | 4891 / 23.4k |

**At low concurrency the non-EP paths win 4–5×** — EP's per-layer dispatch/combine is not amortised
when few tokens move, and at conc 1 that fixed cost is the whole measurement. **DeepEP overtakes at
concurrency ≥8** (30.2k vs 26.8k, TTFT 1010 vs 1158 ms) and holds a ~13% lead at conc 32, with the
ramp still rising while both non-EP columns have flattened. Part 2 continues to concurrency 256.

**pure TP ≈ Baseline at every point** (within ±7%, no consistent winner) — at 16 GPUs the TP
all-reduce and the ordinary fused-MoE all-to-all cost about the same.

## Decode-only (input 256, output 512, so TPOT dominates)

Output tok/s (higher better) / mean TPOT ms (lower better). **Bold = best in row.** DeepEP and
UCCL-EP both pin `--deepep-mode low_latency`, the decode mode, with the cap chain from
[section 3](#3-low_latency-on-a-colocated-server-needs-four-coupled-settings).

| Concurrency | pure TP | Baseline | DeepEP `low_latency` | UCCL-EP `low_latency` |
|---|---|---|---|---|
| 32  | **1224** / **25.9** | 1170 / 27.1 | 554 / 54.5 | 761 / 38.1 |
| 64  | **2161** / **29.3** | 2110 / 29.4 | 1021 / 59.3 | 1360 / 43.1 |
| 256 | **5929** / 41.9 | 5888 / **41.3** | 2634 / 83.0 | 2961 / 69.1 |
| 512 | 8361 / 58.1 | **8939** / **53.7** | 4371 / 96.2 | 4625 / 83.5 |

**pure TP ≈ Baseline, both far ahead of either EP backend** — 2.2× DeepEP at conc 32, narrowing to
1.9× at conc 512. **UCCL-EP beats DeepEP at every point** (see
[the direct comparison](#deepep-vs-uccl-ep-colocated-same-modes-and-same-harness)), but neither
closes on the non-EP paths at 16 GPUs.

The two non-EP columns above were measured with the radix prefix cache **on**; everything else in
this document has it off. The affected metric is `output tok/s`, which includes TTFT — TPOT itself
moves ≤4%. Measured same-config on/off: conc 512 reads 9101 vs 8466 tok/s (−7.0%) and 53.4 vs 55.4 ms
TPOT (+3.7%), conc 32 reads 1152 vs 1156 (+0.4%). So the pure-TP/Baseline throughput above is up to
~7% optimistic at high concurrency and the ordering is unaffected. See
[section 5](#5-disable-the-prefix-cache-or-the-sweep-measures-the-cache).

## `low_latency` on a colocated server: the chunk-cap trade

The cap chain that makes `low_latency` runnable colocated (cap 512, `--chunked-prefill-size 512`,
`NVSHMEM_QP_DEPTH=1026`, `--mem-fraction-static 0.75`) also changes the workload, and the TTFT column
is where that shows up. DeepEP `low_latency`, mean TTFT ms by concurrency: **1682 / 1728 / 7319 /
10683** at conc 32 / 64 / 256 / 512.

TTFT rises ~6× from conc 64 to conc 512, because 512 concurrent prompts must be chunked through a
512-token dispatch instead of 8192. A decode sweep counts TTFT in end-to-end throughput, so at high
concurrency a colocated `low_latency` number is partly measuring the chunk cap, not the decode
kernel — TPOT keeps improving while `output tok/s` does not.

**This is the concrete argument for PD disaggregation.** Colocated, the two effects are inseparable —
you cannot take the low-latency kernel without the prefill chunk cap. Split the roles and you can:
`serve-pd.sh` pins `low_latency` on decode nodes that never prefill, so the small-batch TPOT gain
comes with no TTFT penalty.

## DeepEP vs UCCL-EP, colocated, same modes and same harness

UCCL-EP presents itself under DeepEP's Python API (`ep/deep_ep_wrapper`), so `--moe-a2a-backend
deepep` drives either one and only the image changes. Both columns pin the same `--deepep-mode`, the
same `LL_MAX_TOKENS=512` / `--chunked-prefill-size 512` cap chain and the same
`--mem-fraction-static`, on the same node pair, back to back.

### Prefill, both pinned `normal`

TTFT ms / input tok/s. **Bold = best in row.**

| Input × conc | DeepEP `normal` | UCCL-EP `normal` | tok/s delta |
|---|---|---|---|
| 1024 × 1  | **459** / **2.2k** | 574 / 1.8k | −20% |
| 4096 × 1  | **474** / **8.6k** | 577 / 7.1k | −18% |
| 8192 × 1  | **655** / **12.5k** | 838 / 9.8k | −22% |
| 4096 × 8  | **1010** / **30.2k** | 1328 / 23.0k | −24% |
| 4096 × 32 | **3723** / **30.8k** | 4891 / 23.4k | −24% |

**DeepEP wins prefill at every point by 18–24%.** Same direction as the 2P2D prefill table and as
UCCL's own kernel-level FP8 dispatch numbers: the `normal` high-throughput dispatch is UCCL's soft
spot, and the margin widens slightly with concurrency rather than closing.

### Decode, both pinned `low_latency`

Output tok/s / mean TPOT ms / p99 ITL ms. **Bold = best in row.**

| Concurrency | DeepEP `low_latency` | UCCL-EP `low_latency` | tok/s delta |
|---|---|---|---|
| 32  | 554 / 54.5 / 54 | **761** / **38.1** / **37** | **+37%** |
| 64  | 1021 / 59.3 / 59 | **1360** / **43.1** / **41** | **+33%** |
| 256 | 2634 / 83.0 / 70 | **2961** / **69.1** / **55** | **+12%** |
| 512 | 4371 / 96.2 / 465 | **4625** / **83.5** / **435** | **+6%** |

**UCCL-EP wins decode at every point**, by 37% at conc 32 narrowing to 6% at conc 512, with TPOT
30% lower where it matters. This is the ordering to trust: matched modes, one topology, one harness.

Both backends' p99 ITL jumps ~8× at conc 512 (465 and 435 ms) while their mean TPOT keeps improving:
that is the saturation tail, common to both, not a backend property.

**Both stages, one sentence: prefill DeepEP, decode UCCL-EP** — the same split Part 2 recommends,
now measured colocated with nothing else varying.

## Cross-image control: both stages transfer

The UCCL image is a different SGLang build, so any UCCL-vs-DeepEP number could be measuring the image
rather than the backend. `MOE_BACKEND=baseline` touches neither DeepEP nor UCCL
(`--moe-a2a-backend none`), so running it on both images isolates that. Same node pair, same sweeps,
back to back.

**Decode** (input 256, output 512) — output tok/s / mean TPOT ms:

| Concurrency | baseline on DeepEP image | baseline on UCCL image | delta |
|---|---|---|---|
| 32  | 1170 / 27.1 | 1152 / 27.1 | −1.6% |
| 64  | 2110 / 29.4 | 2108 / 29.6 | −0.1% |
| 256 | 5888 / 41.3 | 5894 / 41.3 | +0.1% |
| 512 | 8939 / 53.7 | 9101 / 53.4 | +1.8% |

**Prefill** (`--random-output-len 1`, cache off on both sides) — TTFT ms / input tok/s:

| Input × conc | baseline on DeepEP image | baseline on UCCL image | tok/s delta |
|---|---|---|---|
| 1024 × 1  | 98 / 10.2k  | 99 / 10.1k  | −0.9% |
| 4096 × 1  | 189 / 21.5k | 190 / 21.4k | −0.7% |
| 8192 × 1  | 349 / 23.3k | 346 / 23.6k | +1.0% |
| 4096 × 8  | 1158 / 26.8k | 1154 / 26.9k | +0.4% |
| 4096 × 32 | 4317 / 26.8k | 4304 / 26.9k | +0.3% |

**Within ±1.8% on decode and ±1% on prefill at every point.** Both images are the same machine as far
as these workloads are concerned, so the DeepEP-vs-UCCL margins above are the backend. (An earlier
version of this control failed on prefill at low concurrency; that was the prefix cache, not the
image — see [section 5](#5-disable-the-prefix-cache-or-the-sweep-measures-the-cache).)

---

# Part 2 — 4-node 2P2D PD-disaggregated (`recipe/serve-pd.sh`)

**Cluster:** 4× `p5en.48xlarge` (8×H200 141 GB, `sm_90`), us-east-2, 16 EFA NICs/node, iface
`enp71s0`. SGLang 0.5.13.post1. Benched through the router with `--pd-separated` and
`--random-range-ratio 1`.

> **Read Part 2 as directional.** Unlike Part 1, these sweeps predate two of the measurement rules
> above. The DeepEP rows ran `--deepep-mode auto` rather than pinned per role, and the radix prefix
> cache was on for the non-DP rows and off for the DP rows — so a "DP effect" column mixes the DP
> lever with a cache difference. Both effects push in DeepEP's favour on prefill and neither is
> quantified here. The *shape* of Part 2 (DeepEP wins prefill, loses decode; DP-attention reverses the
> decode order and hurts prefill) is confirmed independently by the clean Part 1 tables; the absolute
> figures below, especially the 161.5k 1K-prefill peak, are not directly comparable to Part 1's.

```
  Prefill role (TP=16 / EP=16)          Decode role (TP=16 / EP=16)
  ┌──────────┐  ┌──────────┐            ┌──────────┐  ┌──────────┐
  │ node 1   │──│ node 2   │  KV cache  │ node 3   │──│ node 4   │
  │ rank 0   │  │ rank 1   │ ─────────▶ │ rank 0   │  │ rank 1   │
  └──────────┘  └──────────┘  Mooncake  └──────────┘  └──────────┘
        ▲                     over EFA        ▲
        └──────── sglang_router --pd-disaggregation ────────┘
```

## The four configurations

| Label | SGLang flags | What the MoE layer does |
|---|---|---|
| **DeepEP** | `--ep-size 16 --moe-a2a-backend deepep` | DeepEP dispatch/combine over NVSHMEM-libfabric/**EFA** — this sample's build |
| **UCCL-EP** | `--ep-size 16 --moe-a2a-backend deepep` on the UCCL image | Same API, UCCL's own RDMA proxy engine over EFA (`deep_ep_wrapper`) |
| **baseline** | `--ep-size 16 --moe-a2a-backend none` | Expert-parallel, but SGLang's ordinary fused-MoE all-to-all over NCCL |
| **pure TP** | `--tp-size 16`, **no** `--ep-size` | No expert parallelism; MoE weights TP-sharded, only the TP all-reduce crosses GPUs |

**The UCCL-EP image is [`../Dockerfile.uccl`](../Dockerfile.uccl)** — same base, EFA and Mooncake
stack as this sample's `Dockerfile`, with UCCL's `ep/deep_ep_wrapper` in place of DeepEP/NVSHMEM. The
wrapper presents UCCL under DeepEP's Python API, so `--moe-a2a-backend deepep` drives it unchanged
and the image is the only variable. Its launch requirements differ — see
[UCCL-EP launch configuration](#uccl-ep-launch-configuration-that-actually-matters). It is Hopper-only
(`sm_90`); for Blackwell, or for kernel-level UCCL numbers with no serving engine, see
[`uccl-ep-benchmark`](../../../../../micro-benchmarks/expert-parallelism/uccl-ep-benchmark), and for
UCCL under *vLLM* see [`dsv3-uccl-nixl`](../../../vllm/dsv3-uccl-nixl).

## Prefill — input 1K→64K, adaptive concurrency

`--random-output-len 1`. Concurrency is swept per input length (high at 1K, down to 1–2 at 64K,
because a 64K request already fills the prefill batch).

### Peak input throughput per input length

tok/s, higher better; achieving concurrency in parentheses. **Bold = best in row.**

| Input | pure TP | baseline | DeepEP | UCCL-EP |
|---|---|---|---|---|
| 1K  | 48.2k (512) | 53.8k (512) | **161.5k** (256) | 51.5k (512) |
| 4K  | 43.6k (128) | 42.7k (128) | **55.5k** (128) | 42.6k (128) |
| 16K | 80.2k (32) | 75.8k (32) | **84.0k** (32) | 69.2k (32) |
| 32K | 46.1k (8) | 42.2k (1) | **52.3k** (8) | 42.1k (8) |
| 64K | 52.7k (1) | 54.4k (2) | **60.0k** (2) | 50.9k (2) |

**DeepEP wins at every input length**, hugely so where the batch is largest and the tokens shortest:
161.5k input tok/s at 1K against ~48–54k for everything else. That is the regime its high-throughput
dispatch was designed for — one fused dispatch keeps all 16 GPUs fed while the others pay a TP
all-reduce or an ordinary all-to-all per layer. As inputs lengthen, the batch is filled by fewer
longer sequences, the all-to-all shrinks relative to attention, and the margin narrows to ~10–30%.

**pure TP ≈ baseline, UCCL-EP a few percent behind both.** Prefill (the `normal` dispatch) is UCCL's
known soft spot, matching its kernel-level FP8-dispatch/combine numbers.

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
while the other three triple it. DeepEP saturates at **conc 256**; past it throughput falls back to
68.5k as the prefill role overloads, so **256 is the operating point, not 512.** All four completed
every point at full success, including 64K×conc2.

### Prefill ceilings

- **128K input fails.** Prefill computes fine, but the Mooncake/EFA transfer of a 128K-token KV
  cache times out: `EFA submitSlicesOnPeer: CQ drain wr_depth=256, max=256`. Raising `MC_MAX_WR` or
  chunking the KV transfer is the thing to try. **The ceiling here is ~64K input, and it is a
  KV-transfer limit, not a compute one.**
- **64K at concurrency ≥4** exhausts the 2-node prefill KV pool. 64K is low-concurrency territory.

## Decode — output length and concurrency sweeps

Input 256, long output, so TPOT dominates.

### A. Fixed concurrency 32, varying output length

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
essentially tied and fastest — no all-to-all in the decode path at all. **UCCL-EP is a clear third
but healthy** (~13% below baseline at conc 128, TPOT 39 vs 33 ms). **DeepEP is last** at 2165 tok/s
/ TPOT 57 ms, ~1.7× the others' TPOT. All four scale cleanly to concurrency 128 with no wedge.

Same ordering as the colocated Part 1 tables — two topologies agreeing that 16-GPU decode is not
DeepEP's regime.

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

**DP-attention helps EP-MoE decode and hurts TP-MoE decode.** With expert parallelism the per-rank
token batch is small; making attention data-parallel removes redundant attention/KV work and lets
the all-to-all coordinate over larger per-rank chunks. With TP-MoE, TP16 attention is already
efficient at 16 GPUs, so `dp16` only adds synchronisation and imbalance. Consequence: **turning DP
on flips the decode winner** from baseline to UCCL-EP.

### DP-attention hurts prefill, universally

Peak input throughput at 1K input:

| Backend | non-DP | + DP | Effect |
|---|---|---|---|
| DeepEP | 161.5k | 59.6k | **−63%** |
| baseline | 53.8k | 34.8k | −35% |
| UCCL-EP | 51.5k | 37.0k | −28% |
| pure TP | 48.2k | 34.7k | −28% |

DP adds synchronisation with no upside on already-large batches, and it is worst for DeepEP, whose
prefill advantage comes from a big fused dispatch. **Never enable DP-attention on the prefill role.**

### Two settings DP-attention requires

`recipe/serve-pd.sh` applies both when `DP_ATTENTION=1`; both are crashes if missed, not slowdowns.

- **Pre-compile DeepGEMM on both prefill nodes** (`recipe/serve-pd.sh precompile <rank>`) for any
  DeepEP-backed DP mode. DP+EP uses grouped-GEMM shapes (`num_groups=16`) absent from the normal
  cache, and the first-time JIT compile trips DeepEP's dispatch warmup timeout. A single-node
  precompile cannot initialise 16-rank EP, so it must run on both. (UCCL tolerates the cold compile;
  baseline/pure TP do not use DeepGEMM here.)
- **The decode role must pin `--cuda-graph-bs 128`** with `--mem-fraction-static 0.78`, or each of
  the 16 DP ranks captures the full batch-size list `[1..512]` and OOMs during graph capture.

## Recommended configuration at this scale

| Role | Configuration | Why |
|---|---|---|
| **Prefill** | **DeepEP**, no DP-attention, operate at concurrency ~256 for 1K inputs | 161.5k input tok/s, 3× the alternatives; DP would cost 63% |
| **Decode** | **UCCL-EP + DP-attention**, `--cuda-graph-bs 128`, `--deepep-mode low_latency` | 4094 tok/s / TPOT 28 ms, the fastest decode measured |
| **Decode**, same-image fallback | **baseline**, no DP-attention | 3626 tok/s / TPOT 34 ms — 13% behind, needs no second image |

Different backends per role means different *images* per role, which PD makes practical — the roles
share nothing but the KV stream. If one image is a hard requirement, **DeepEP prefill + baseline
(non-DP) decode** is the best same-image split, and **pure TP everywhere** the best single-config
answer.

## UCCL-EP is instance-type sensitive

The same UCCL-EP build behaves very differently on `p5` (H100, **32** EFA NICs/node) than on `p5en`
(H200, 16 NICs/node). 2P2D decode, input 256 / output 128, same sweep on both:

| conc | UCCL on `p5` (H100) | UCCL on `p5en` (H200) | ratio |
|---|---|---|---|
| 8  | 42.9 tok/s, TPOT 163 ms | **186.5** tok/s, TPOT **25.4** ms | 4.3× |
| 16 | 95.9, 160 ms | **323.4**, **27.3** ms | 3.4× |
| 32 | 188.0, 163 ms | **451.8**, **26.6** ms | 2.4× |

**2.4–4.3× faster on p5en/H200 than on p5/H100, with a 6× TPOT difference** (26 vs 160 ms). On p5 it
looks like a broken backend; on p5en it is comfortably ahead of DeepEP and, with DP-attention, the
fastest decode in this document. Consistent with UCCL's EFA path not being tuned for p5's NIC
topology (32 NICs for 8 GPUs, a different NIC/PCIe-switch layout) rather than a kernel defect — its
correctness checks pass identically on both, and
[`ep-backend-comparison`](../../../../../micro-benchmarks/expert-parallelism/ep-backend-comparison)
shows the UCCL-vs-NVSHMEM ordering flipping by GPU generation too.

**Choose the EP backend per instance type, not once and globally.** These decode ratios compare UCCL
to UCCL with `MOONCAKE_PROTOCOL=efa` set on both sides. The rest of that p5 sweep ran before the
Mooncake fix, so no other p5 conclusion should be carried forward.

## Required setting: `MOONCAKE_PROTOCOL=efa`

Every 2P2D number here was measured with the KV cache on EFA RDMA, which requires
`MOONCAKE_PROTOCOL=efa`. On EFA hardware there is no reason to run the KV path any other way, so
treat it as part of the deployment, not a tuning knob.

**Omitting it is a silent fallback to TCP, not an error.** The server comes up, returns correct
tokens, and passes a smoke test; under a sustained high-concurrency prefill burst it then deadlocks
with `KVTransferError ... session is not alive`. The only evidence of the transport is the server log:

```
transfer_engine_py.cpp:241] Installing TCP transport (auto_discover disabled in EFA build)   # WRONG
tcp_transport.cpp:678]      TcpTransport: listen on port 16705

transfer_engine_py.cpp:100] Using default malloc/free for protocol: efa                      # right
efa_transport.cpp:1025]     EfaTransport: Initialized EFA device rdmap160s0 ...
```

So check it on every role before benchmarking:

```bash
docker logs r1-pd-prefill 2>&1 | grep -E 'EfaTransport|Installing TCP transport'
```

`recipe/serve-pd.sh` sets the variable and prints this grep. Note the deadlock is **not** NVSHMEM
contention — it reproduces with `MOE_BACKEND=tp`, which never loads DeepEP. The KV path is
Mooncake's alone.

## The 2P2D KV path has three container-level requirements

`MOONCAKE_PROTOCOL=efa` is one of three. All three share a failure signature that makes them hard to
tell apart: **each one logs a warning, lets startup complete, passes the built-in PD warmup, and then
kills the first real KV transfer.** The visible error is always on the wrong side — prefill reports
`Decode instance could be dead, remote mooncake session <ip:port> is not alive`, decode reports
`Failed to get kvcache from prefill instance, it might be dead`. Nothing is dead; the buffers were
never registered. `recipe/serve-pd.sh` sets all three for both roles and every backend.

| Requirement | Warning it emits | Why |
|---|---|---|
| `--privileged` | `memory_location.cpp: Failed to get NUMA node, addr: … Operation not permitted` | Mooncake asks the kernel which NUMA node each registered buffer sits on; the query is not permitted under Docker's default capability set. |
| `FI_HMEM=system,cuda` | `efa_context.cpp: fi_mr_regattr failed for GPU memory … Operation not supported` then `Failed to register memory region chunk 0 with EFA context 0` | The KV cache is GPU memory, and the EFA provider's `fi_mr_reg()` hardcodes `FI_HMEM_SYSTEM`, so Mooncake registers device buffers via `fi_mr_regattr(iface=FI_HMEM_CUDA)`. Mooncake's own default is `setenv("FI_HMEM", "system", 0)` — system **only** — so leaving it unset disables the very interface it then asks for. |
| `MOONCAKE_PROTOCOL=efa` | `Installing TCP transport` | Silent fallback to sockets; see above. |

Verify all three on one role before benchmarking — this must print nothing:

```bash
docker logs r1-pd-decode 2>&1 | grep -E "Operation not permitted|fi_mr_regattr failed|Installing TCP transport"
```

**The PD warmup does not cover this.** SGLang's own disaggregation warmup request succeeds even with
GPU-memory registration broken (its 4-token prompt takes a path that does not need it), so
`The server is fired up and ready to roll!` plus a 200 from the warmup is not evidence the KV path
works. Send one real request through the router instead.

## UCCL-EP launch configuration that actually matters

Reproducing the UCCL rows needs four things DeepEP does not. Getting them wrong produces a SIGSEGV
during prefill startup that reads like a UCCL bug and is not:

1. **`--deepep-mode normal` on prefill and `low_latency` on decode — not `auto`.** The one that
   matters most: `auto` drives UCCL's prefill down a path that segfaults.
2. **`MOONCAKE_PROTOCOL=efa`.**
3. **`--privileged` on the container.** UCCL registers GPU memory for RDMA through the
   dma-buf/ibverbs path, which needs the privilege; DeepEP/NVSHMEM use GDRCopy and do not.
4. **`--mem-fraction-static 0.70` on decode.** UCCL's `low_latency` RDMA buffers are large; 0.82
   OOMs during CUDA-graph capture at `uccl_ep.cc:289`.

Plain TP16/EP16 works — no `--dp-size`/`--enable-dp-attention` required (though DP is what makes
UCCL the decode winner). The compute kernels were never implicated: `compute-sanitizer` reports 0
errors from UCCL's `ep.abi3.so`, and its kernel tests pass.

---

# Kernel-level context (why the EFA port is nonetheless sound)

The serving numbers are a *scale* story; the kernel numbers are the *enablement* story — H200, 2
nodes, 4096 tokens × hidden 7168:

| Metric | DeepEP official (H800 + CX7 IB 400Gb/s) | This port (H200 + EFA) |
|---|---|---|
| 16-EP dispatch (RDMA) | 43 GB/s | **62.7 (FP8) / 72.3 (BF16)** |
| 16-EP combine (RDMA) | 43 GB/s | **59.4** |
| 32-EP dispatch (RDMA) | 58 GB/s | 47.1 (FP8) / 51.5 (BF16) |
| 32-EP combine (RDMA) | 57 GB/s | 53.2 |

Different hardware on each side, so read this as a sanity band rather than a controlled A/B. The
point is that **EFA delivers IB-class dispatch/combine bandwidth**. Full kernel tables, the
`NVSHMEM_NETDEVS_POLICY` ablation and cross-generation deltas live in `deepep-benchmark` and
`ep-backend-comparison`.

`recipe/run-kernel-test.sh` against the image built from the committed Dockerfile, `p5en.48xlarge`,
2026-07-28 — the pre-flight checks, not a benchmark:

| Test | Correctness | Bandwidth |
|---|---|---|
| `intranode` (NVLink only, no NVSHMEM, 8 ranks) | **24/24 passed** | dispatch 316.6 GB/s (FP8) / 330.4 (BF16), combine 323.8 — all NVLink |
| `low_latency` (NVSHMEM, 8 ranks) | passed | dispatch 180–189 GB/s (~40 µs), combine 218–226 GB/s (~65 µs) |
| `internode` (NVSHMEM/EFA, 2 nodes, 16 EP ranks) | **32/32 passed on both ranks** | dispatch FP8 **62.1** GB/s RDMA / 202.8 NVL; BF16 **71.1** / 232.2; combine **59.2** / 193.3 |

The internode figures land within ±2% of the 16-EP column above, measured on different instances on
a different date — so the EFA transport reproduces. Only `internode` puts bytes on EFA; run the
other two anyway, because them failing tells you the image is broken before you spend two nodes
finding out.

# Blackwell: expect DeepEP to lose at 2 nodes

No Blackwell serving run is in this repo. But the question comes up — *"we benchmarked DeepEP vs the
NCCL all-to-all on 2× B300 and DeepEP was slower in every configuration; is that expected?"* — and
the answer from the data that **is** here is **yes at 2 nodes, and it is not an EFA problem.**
Reported shape of such a result: output throughput −7% to −26%, median TTFT +17% to +82%, P99 ITL
1.2–1.9 s vs 0.8–0.9 s, `normal` (HT) slowest where it ran, at TP16/EP16 across two nodes, 8K input
/ 1K output, concurrency 128.

1. **16 ranks is DeepEP's worst case, on any GPU.** Every table here says so. The colocated decode
   sweep has DeepEP at 0.23–0.75× the baseline's throughput and 1.4–4.3× its TPOT; the 2P2D sweep
   0.55–0.71× at 1.4–1.8× TPOT. A −7% to −26% *aggregate* regression on a mixed 8K/1K workload is
   **milder than what we measure on Hopper**, not worse.
2. **The published B300 kernel numbers are healthy, which localises the gap above the transport.**
   At 2 nodes / 16 ranks on `p6-b300`, DeepEP-over-EFA dispatch/combine is **126.6 / 106.4 GB/s** —
   best of the three backends there, *above* the NCCL all-to-all's 104.9 GB/s at matched payload
   ([`ep-backend-comparison`](../../../../../micro-benchmarks/expert-parallelism/ep-backend-comparison/RESULTS.md)).
   So fabric and kernels are fine at that scale; the serving regression is per-layer
   launch/scheduling overhead and MoE-runner choice, not bytes on the wire.
3. **`normal`/HT being slowest at decode-heavy concurrency is by design.** On a 1K-output workload
   the run is TPOT-dominated, where `low_latency` is the intended mode.

Before concluding anything from such a run, eliminate these:

- **The four settings in [How to measure this correctly](#how-to-measure-this-correctly)** — the
  harness defaults and the mode pinning in particular. `--random-range-ratio` alone moved a nominal
  256/512/conc-64 point from 374 to 1127 tok/s.
- **The MoE runner is not held constant** in the usual formulation: DeepEP rows run
  `--moe-runner-backend deep_gemm` while the no-DeepEP rows resolve `auto` to `flashinfer_trtllm` on
  Blackwell. That is two variables, and TRT-LLM's Blackwell MoE kernels are heavily tuned. Re-run
  DeepEP against `flashinfer_trtllm` (or the baseline against `deep_gemm`) before attributing the
  delta to the all-to-all.
- **DeepGEMM JIT warmup.** It inflates early TTFT and P99 ITL specifically — the two metrics that
  move most in reports like this. Pre-warm on **both** nodes (`recipe/serve-pd.sh precompile`).
- **An HT-path hang that does not reproduce.** At ≥128 ranks the NVSHMEM-libfabric host proxy
  exhausts libfabric retries (`EAGAIN` in `nvshmemi_process_multisend_rma`) and kills a different
  pair of ranks each run — a documented statistical fan-out limit, not a bad node. At 16 ranks it
  should not fire, but a non-reproducing hang on the HT path has the same signature.

**The load-bearing point for a large fleet: 2 nodes measures the wrong thing.** DeepEP is built for
EP domains where experts are spread thin enough that every token crosses the fabric. Kernel scaling
to 256 ranks on `p6-b300` is already characterised: the useful envelope is ~64–160 ranks, with hard
implementation caps past that (HT: 160 PEs at `deep_ep.cpp:158`; low-latency: between 64 and 128
PEs). **A production-EP-width run — EP32 or EP64, not EP16 — is the measurement that decides this.**

# Reproduce

## Colocated, 2 nodes

```bash
source setup/env_vars

# Pin the kernel mode to the stage you are about to measure.
DEEPEP_MODE=normal      MOE_BACKEND=deepep recipe/serve.sh 0   # node 1, then benchmark prefill
DEEPEP_MODE=normal      MOE_BACKEND=deepep recipe/serve.sh 1   # node 2
# ...or, for decode (serve.sh applies the four low_latency settings for you):
DEEPEP_MODE=low_latency MOE_BACKEND=deepep recipe/serve.sh 0
DEEPEP_MODE=low_latency MOE_BACKEND=deepep recipe/serve.sh 1

curl -s localhost:30000/health && echo OK
recipe/benchmark.sh prefill      # or: recipe/benchmark.sh decode
```

Then repeat with `MOE_BACKEND=baseline` and `MOE_BACKEND=tp`. `recipe/serve.sh` also derives
`NVSHMEM_DISABLE_CUDA_VMM` from `DEEPEP_MODE` — `normal` needs VMM off or NVSHMEM's
topology/transport-map init fails, `low_latency`/`auto` need it on or the RDMA-buffer `cudaMemset`
fails at `deep_ep.cpp:371`. Getting it wrong is a startup failure, not a slow server.

For the UCCL columns, build [`Dockerfile.uccl`](../Dockerfile.uccl), point `IMAGE_URI` at it and add
`UCCL=1`, which switches on
`--privileged`, drops the NVSHMEM environment UCCL has no use for, and makes `DEEPEP_MODE`
mandatory. Everything else — the cap chain, `--mem-fraction-static`, the sweep — is identical, which
is what keeps the two backends' tables comparable:

```bash
IMAGE_URI=<uccl-image> UCCL=1 DEEPEP_MODE=low_latency MOE_BACKEND=deepep recipe/serve.sh 0
```

For the cross-image control, run `MOE_BACKEND=baseline` on both images: it selects
`--moe-a2a-backend none`, so it exercises neither DeepEP nor UCCL and any difference is the image.

## 2P2D, 4 nodes

Fill in the PD block of `setup/env_vars` first. The UCCL rows need the second image
([`Dockerfile.uccl`](../Dockerfile.uccl)); point `IMAGE_URI` at it and add `UCCL=1`.

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

- **Re-measure Part 2 under the Part 1 rules** — prefix cache off on every row and `--deepep-mode`
  pinned per role — so the DP-effect columns isolate the DP lever alone and the two parts share a
  harness exactly. `recipe/serve-pd.sh` now does both by construction; only the tables are stale.
- **The per-role split recommendation is inferred, not measured.** DeepEP prefill and UCCL+DP decode
  were each measured in a same-backend-both-roles deployment; the mixed deployment was never run end
  to end. It should work — the roles share only the KV stream — but it is untested.
- **128K input needs a larger `MC_MAX_WR` or chunked KV transfer** to clear the EFA CQ-drain timeout.
- **Larger EP scale is the real question.** Everything here is 16 GPUs per role, where DeepEP's
  decode overhead cannot amortise. Kernel scaling to 32 nodes / 256 ranks is characterised in
  `ep-backend-comparison`; the *serving* crossover is not. Re-run decode on ≥8–16 nodes with a
  larger `--ep-size` and `low_latency` pinned.
