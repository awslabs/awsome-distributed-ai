<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# DeepSeek-R1 2P2D on SGLang — MoE all-to-all backends compared

Prefill/decode-disaggregated (**2P2D**) serving of DeepSeek-R1 671B FP8 on four `p5en.48xlarge`
nodes, sweeping the four ways to parallelise the MoE layer, and separating **prefill** from
**decode** because the two stages want opposite things from the all-to-all.

This is the **PD-disaggregated** companion to [`RESULTS.md`](./RESULTS.md), which measures the
*colocated* 2-node topology that [`recipe/serve.sh`](../recipe/serve.sh) launches. Different
topology, different scripts ([`recipe/serve-pd.sh`](../recipe/serve-pd.sh) +
[`recipe/benchmark-pd.sh`](../recipe/benchmark-pd.sh)), different conclusions — do not read across
the two documents as one experiment.

## Headline

**The best MoE backend depends on the stage, and no single backend wins both.**

| Stage | Winner | Margin |
|---|---|---|
| **Prefill** (large batches, throughput) | **DeepEP** | 161.5k input tok/s at 1K×conc256 — **3.0×** the best any other backend reaches at 1K, **3.6×** at matched concurrency |
| **Decode** (small batches, latency) | **UCCL-EP + DP-attention** | 4094 output tok/s at conc128, TPOT 28 ms — **1.9×** DeepEP, **1.13×** the best non-DP config |

Because prefill and decode run on *separate* node groups in a PD deployment, you can act on that:
**run DeepEP on the prefill role and UCCL-EP + DP-attention on the decode role.** That split is
the most useful result in this document and it is only available because the roles are
disaggregated.

Two things temper the decode half of it: the UCCL rows come from a different image with no
cross-image decode control (caveat 3), and the UCCL+DP margin over `baseline+DP` is 23% while over
plain `baseline` it is only 13%. The prefill half is same-image and the margin is 3×, so it is the
firmer of the two claims.

## Topology

**Cluster:** 4× `p5en.48xlarge` (8×H200 141 GB, `sm_90`), us-east-2, 16 EFA NICs/node, iface
`enp71s0`. Date: 2026-06-24. SGLang 0.5.13.post1.

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
falls back to TCP sockets, which invalidates every number on this page.

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

## Caveats — read before quoting any number

1. **Single-seed smoke benchmarks.** One pass of `sglang.bench_serving` per point. Directional,
   not statistically tight. The prefill sweep in particular is noisy at low concurrency, where a
   4-request point is dominated by warmup.
2. **`--deepep-mode` is not uniform across the rows.** The non-DP **DeepEP** rows use `auto`;
   DeepEP+DP and **every UCCL row** pin `normal` on prefill and `low_latency` on decode (UCCL
   requires the pinned form — see below). So DeepEP-vs-UCCL is confounded by mode selection at the
   non-DP points, and clean at the DP points. `recipe/serve-pd.sh` pins per role for all backends,
   which is the better default and means a re-run would not reproduce this inconsistency.
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
4. **Every run below predates `NCCL_NET_PLUGIN=ofi`, so NCCL was on TCP sockets, not EFA.** The
   image did not set it, and the EFA installer ships the plugin under a name NCCL does not
   auto-load, so NCCL logged *"Could not find: libnccl-net.so"* and used `NET/Socket` with GPU
   Direct RDMA disabled. This is now fixed in the Dockerfile and asserted by
   `recipe/verify-image.sh`. It **understates the two NCCL-dependent backends most** — `baseline`
   routes its fused-MoE all-to-all over NCCL, and `pure TP` routes every layer's tensor-parallel
   collectives over it — while DeepEP and UCCL move their expert traffic over NVSHMEM/libfabric and
   their own transport respectively, touching NCCL only for the non-MoE collectives. So the DeepEP
   prefill win below is, if anything, **overstated against baseline and TP**, and the decode gap
   where they lead is understated. A re-run with the fix is the highest-value next measurement.
5. **Kernel-level bandwidth is out of scope here.** For matched-config dispatch/combine numbers
   across NCCL/UCCL/NVSHMEM out to 256 ranks, see
   [`ep-backend-comparison`](../../../../../micro-benchmarks/expert-parallelism/ep-backend-comparison).
   This page is only about what a served model does.
6. **Peak input throughput is a saturation metric, not a latency metric.** The high-concurrency
   points have TTFT in the seconds. Read the TTFT column too if you care about interactive
   prefill.

---

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

---

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

All four scale cleanly to concurrency 128 with no wedge. (Earlier runs hit a ceiling well below
that, which turned out to be the KV transport falling back to TCP — see
[`MOONCAKE_PROTOCOL=efa`](#required-setting-mooncake_protocolefa).)

---

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

---

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

---

## Platform matters more than the backend: UCCL-EP on p5 vs p5en

The same UCCL-EP build behaves very differently on the two instance types, and reading only the
p5/H100 data would badly misjudge it.

2P2D decode, input 256 / output 128, same sweep on both:

| conc | UCCL on `p5` (H100) | UCCL on `p5en` (H200) | ratio |
|---|---|---|---|
| 8  | 42.9 tok/s, TPOT 163 ms | **186.5** tok/s, TPOT **25.4** ms | 4.3× |
| 16 | 95.9, 160 ms | **323.4**, **27.3** ms | 3.4× |
| 32 | 188.0, 163 ms | **451.8**, **26.6** ms | 2.4× |

For context, the whole p5 sweep — 4× `p5.48xlarge` (8×H100 80 GB HBM3, **32** EFA NICs/node),
2026-06-23, same 2P2D topology, `--enforce-eager` nowhere, input 256 / output 128:

| conc | baseline | DeepEP | UCCL-EP |
|---|---|---|---|
| 8  | **268.7** / **24.9** | 96.9 / 60.2 | 42.9 / 163.0 |
| 16 | **436.7** / **27.1** | 227.3 / 62.8 | 95.9 / 160.2 |
| 32 | **504.9** / **28.7** | 306.9 / 65.5 | 188.0 / 163.1 |

**UCCL-EP is 2.4–4.3× faster on p5en/H200 than on p5/H100, and the TPOT difference is a factor of
six** (26 vs 160 ms). On p5 it looks like a broken backend, a distant third behind even DeepEP; on
p5en it is comfortably ahead of DeepEP and, with DP-attention, the fastest decode in this whole
document. The gap is consistent with UCCL's EFA path not being tuned for p5's NIC topology (32 EFA
NICs for 8 GPUs, a different NIC/PCIe-switch layout than p5en's 16) rather than with any defect in
its kernels — its correctness checks pass identically on both, and the same
[`ep-backend-comparison`](../../../../../micro-benchmarks/expert-parallelism/ep-backend-comparison)
data shows the UCCL-vs-NVSHMEM ordering flipping by GPU generation as well.

**So: choose the EP backend per instance type, not once and globally.** On p5, DeepEP; on p5en,
DeepEP for prefill and UCCL-EP for decode.

### The p5/H100 2P2D numbers carry a KV-transport confound

The H100 2P2D runs in [`RESULTS.md`](./RESULTS.md#h100-2p2d-pd-disaggregated-different-topology-see-caveat-2)
predate the `MOONCAKE_PROTOCOL=efa` fix described below, and the DeepEP and baseline launchers did
not set it while the UCCL launcher always did. So on p5:

- **DeepEP vs baseline is still a fair comparison** — both ran with KV over TCP, both handicapped
  identically, and both swept only to concurrency 32, low enough that neither hit the
  `session is not alive` wedge.
- **UCCL vs the other two is not.** The p5 prefill sweep (output 1, input tok/s / TTFT ms) makes
  UCCL look dominant:

  | Input × conc | baseline | DeepEP | UCCL-EP |
  |---|---|---|---|
  | 1024 × 1 | 2834 / 360 | 2538 / 402 | **3094** / **330** |
  | 4096 × 1 | 2794 / 1462 | 3230 / 1267 | **11498** / **355** |
  | 8192 × 1 | 3684 / 2223 | 3596 / 2277 | **15619** / **523** |
  | 4096 × 4 | 6256 / 2451 | 5638 / 2744 | **17430** / **936** |

  A 3–4× prefill lead that appears only where the other two are moving KV over TCP is very likely
  measuring the transport, not the backend — and the ordering **reverses** on p5en once all four
  run on EFA (DeepEP 161.5k vs UCCL 51.5k at 1K). **Do not carry the p5 prefill conclusion
  forward.**

This is an inference from the launcher history rather than from a controlled re-run — the p5
cluster was released before it could be repeated. It is stated here because the alternative is
leaving a misleading comparison on the record.

---

## Required setting: `MOONCAKE_PROTOCOL=efa`

**Every number on this page was measured with the KV cache on EFA RDMA, which requires
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

## Reproduce

Four nodes, this sample's image on all of them. Fill in the PD block of `setup/env_vars` first. The
UCCL rows need a separate image that this sample does not ship (see
[the four configurations](#the-four-configurations)); once built, point `IMAGE_URI` at it and add
`UCCL=1`.

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

## Open items

- **The p5/H100 2P2D tables should be re-run with `MOONCAKE_PROTOCOL=efa`** so the UCCL comparison
  there is fair (see the confound note above).
- **The per-role split recommendation is inferred, not measured.** DeepEP prefill and UCCL+DP
  decode were each measured in a same-backend-both-roles deployment; the mixed deployment was
  never run end to end. It should work — the roles only share the KV stream — but it is untested.
- **Add a cross-image decode control** (run `baseline` decode on both images), and re-run the
  non-DP DeepEP rows with `--deepep-mode` pinned per role instead of `auto`, so caveats 2 and 3
  stop applying.
- **128K input needs a larger `MC_MAX_WR` or chunked KV transfer** to get past the EFA CQ-drain
  timeout.
- **Larger EP scale is the real question.** Everything here is 16 GPUs per role, where DeepEP's
  decode overhead cannot be amortised. The kernel scaling out to 32 nodes / 256 ranks is
  characterised in `ep-backend-comparison`; the *serving* crossover is not.
