<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# DeepSeek-R1 on SGLang + DeepEP-over-EFA — measured results

End-to-end serving results for the image in this directory, on both H200 (`p5en.48xlarge`) and
H100 (`p5.48xlarge`). For the **kernel-level** dispatch/combine bandwidth of the same DeepEP-EFA
build — including the 8-to-32-node scaling runs and the NCCL/UCCL cross-backend comparison — see
[`micro-benchmarks/expert-parallelism/deepep-benchmark`](../../../../../micro-benchmarks/expert-parallelism/deepep-benchmark)
and [`ep-backend-comparison`](../../../../../micro-benchmarks/expert-parallelism/ep-backend-comparison).

## Read this first

**On decode at 16 GPUs, DeepEP does not beat SGLang's ordinary all-to-all, and neither beats pure
TP.** That is the expected result at this scale, not a defect in the EFA port:

- With 16 ranks and 256 experts, each GPU already owns 16 experts. The MoE fan-out is small and
  largely intra-node NVLink, which the fused-MoE path handles at near-zero overhead. DeepEP's
  per-layer dispatch/combine launches, NVSHMEM RDMA setup and grouped-GEMM cost more than they save
  when so little traffic crosses the network.
- DeepEP targets **large-scale** expert parallelism — tens of nodes, experts spread thin enough
  that every token must cross the fabric, where an ordinary all-to-all's message count and lack of
  compute/comm overlap make it collapse. That crossover is not reachable on 2 nodes for decode.

**Prefill is the exception, and this sweep does not go far enough to show it.** The concurrency
here stops at 32, where DeepEP has only just overtaken the baseline. Pushed to concurrency 256 in
the PD-disaggregated runs, DeepEP reaches **161.5k input tok/s — 3× the alternatives** at 16 GPUs.
See [`RESULTS-2p2d.md`](./RESULTS-2p2d.md). So the scale story is really *per stage*: DeepEP wins
prefill once the batch is large, and loses decode at 16 GPUs regardless.

**What these runs do establish:** the DeepEP dispatch/combine kernels are correct and run at
IB-class bandwidth over EFA with no InfiniBand present, and the build plugs into a real
DeepSeek-R1 SGLang server end to end. Before this port, DeepEP could not use EFA at all — NVSHMEM
offered only the IBRC/IBGDA InfiniBand transports.

## Caveats — please read before quoting any number

1. **Single-seed smoke benchmarks.** Each point is one pass of `sglang.bench_serving`. These are
   directional, not statistically tight. Re-run across seeds and compute intervals before treating
   any delta as real.
2. **The two tables below are different topologies.** The H200 table is **colocated** 2-node
   TP16/EP16 — the configuration `recipe/serve.sh` reproduces. The H100 table is **2P2D
   PD-disaggregated** (4 nodes: 2 prefill + 2 decode, fronted by `sglang_router
   --pd-disaggregation`, benched with `--pd-separated`), reproduced by `recipe/serve-pd.sh` and
   `recipe/benchmark-pd.sh`, **not** by `recipe/serve.sh`. Do not read across the two tables as a
   hardware A/B — the topologies differ.
3. **UCCL rows are not included here.** UCCL-EP in the same 2P2D topology, on both H100 and H200,
   is in [`RESULTS-2p2d.md`](./RESULTS-2p2d.md); `ep-backend-comparison` has the matched-config
   kernel-level comparison.
4. **The H100 2P2D table below moved the KV cache over TCP, not EFA** — it predates the
   `MOONCAKE_PROTOCOL=efa` fix. Baseline-vs-DeepEP there is still fair (both handicapped
   identically, and both stay below the concurrency where the TCP path collapses), but the absolute
   numbers are low. See [`RESULTS-2p2d.md`](./RESULTS-2p2d.md#required-setting-mooncake_protocolefa).

---

## H200 — colocated 2-node TP16/EP16 (matches `recipe/serve.sh`)

**Cluster:** 4× `p5en.48xlarge` (8×H200 141GB, sm_90), us-east-2, 16 EFA NICs/node, iface
`enp71s0`. SGLang 0.5.13.post1 + DeepEP `567632d` + NVSHMEM v3.7.0 (libfabric/EFA). Date:
2026-06-22. Serving 2 nodes / TP=16 / EP=16. All DeepEP kernel correctness checks passed first.

`pure TP` = `--tp-size 16` with no `--ep-size` (MoE weights TP-sharded; no dispatch/combine at
all) = `MOE_BACKEND=tp`. `Baseline` = EP with the ordinary fused-MoE all-to-all =
`MOE_BACKEND=baseline`. `DeepEP` = `MOE_BACKEND=deepep`.

### Prefill-only (`--random-output-len 1`, so TTFT ≈ prefill time)

TTFT ms (lower better) / input-token tok/s (higher better). **Bold = best in row.**

| Input × conc | pure TP | Baseline | DeepEP |
|---|---|---|---|
| 1024 × 1  | 88 / **11.0k**  | **87** / **11.0k** | 268 / 3.7k |
| 4096 × 1  | **158** / **25.1k** | 185 / 21.5k | 279 / 14.1k |
| 8192 × 1  | **181** / **44.0k** | 186 / 42.9k | 410 / 19.4k |
| 4096 × 8  | **992** / 30.8k | 1028 / 29.7k | 996 / **31.0k** |
| 4096 × 32 | 3282 / 34.2k | 3251 / 34.3k | **2938** / **38.8k** |

At low concurrency the baseline wins 2–3× on TTFT — EP's per-layer dispatch/combine overhead is
not amortized when few tokens move. At concurrency ≥8 DeepEP overtakes it on prefill throughput,
reaching 38.8k tok/s at conc 32.

### Decode-only (input 256, output 512, so TPOT dominates)

Output tok/s (higher better) / mean TPOT ms (lower better). **Bold = best in row.**

| Concurrency | pure TP | Baseline | DeepEP |
|---|---|---|---|
| 32  | — | **1175** / **27.1** | 634 / 50.0 |
| 64  | 2130 / **29.3** | **2047** / 29.8 | 1142 / 54.7 |
| 256 | 5616 / 43.4 | **5874** / **41.3** | 3870 / 63.6 |
| 512 | 8239 / 59.2 | **9048** / **53.5** | 6685 / 72.9 |

(pure TP has no conc=32 point in this sweep.) Pure TP ≈ baseline at the top; both beat DeepEP on
decode at this scale.

---

## H100 2P2D PD-disaggregated (different topology; see caveat 2)

**Cluster:** 4× `p5.48xlarge` (8×H100 80GB HBM3, sm_90), us-east-2, **32** EFA NICs/node, iface
`enp71s0`. Same image. Date: 2026-06-23. Server flags: `--mem-fraction-static 0.82
--chunked-prefill-size 16384 --watchdog-timeout 1200`. DeepEP kernel correctness: internode 32/32,
intranode 24/24 passed.

### Decode sweep (input 256, output 128)

tok/s (higher better) / TPOT ms (lower better). **Bold = best in row.**

| Concurrency | Baseline | DeepEP |
|---|---|---|
| 8  | **268.7** / **24.9** | 96.9 / 60.2 |
| 16 | **436.7** / **27.1** | 227.3 / 62.8 |
| 32 | **504.9** / **28.7** | 306.9 / 65.5 |

### Prefill-only sweep (output 1)

TTFT ms (lower better) / input tok/s (higher better). **Bold = best in row.**

| Input × conc | Baseline | DeepEP |
|---|---|---|
| 1024 × 1 | **360** / **2834** | 402 / 2538 |
| 4096 × 1 | 1462 / 2794 | **1267** / **3230** |
| 8192 × 1 | **2223** / **3684** | 2277 / 3596 |
| 4096 × 4 | **2451** / **6256** | 2744 / 5638 |

Baseline beats DeepEP on decode by ~1.6–2.8× throughput and is roughly par on prefill — the same
conclusion as the H200 colocated runs, reached on different hardware and a different topology.
Note the prefill sweep stops at concurrency 4 here, well short of where DeepEP's advantage appears
(see caveat 4 on the transport, and `RESULTS-2p2d.md` for the full concurrency ramp on H200).

---

## Kernel-level context (why the EFA port is nonetheless sound)

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
these kernels. Full kernel tables, the `NVSHMEM_NETDEVS_POLICY` ablation, and the H100/H200 deltas
live in the `deepep-benchmark` and `ep-backend-comparison` directories.

## To actually demonstrate a DeepEP win

Two routes, and the cheap one comes first:

1. **Push prefill concurrency at this same scale.** The sweep above stops at 32. At concurrency 256
   DeepEP hits 161.5k input tok/s versus ~44k for the baseline — a 3× win on 16 GPUs. Already
   measured: [`RESULTS-2p2d.md`](./RESULTS-2p2d.md).
2. **For decode, re-run on ≥8–16 nodes** so experts are spread across many nodes and the
   all-to-all becomes network-bound, with a larger `--ep-size` and `--deepep-mode low_latency`
   pinned. On 2 nodes the decode crossover is not reachable — though enabling DP-attention recovers
   ~25% of the gap, and does so for UCCL-EP too. The scaling behaviour of the kernels themselves
   out to 32 nodes / 256 ranks is already characterised in `ep-backend-comparison`.
