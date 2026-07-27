<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# DeepSeek-R1 on vLLM + DeepEP-over-EFA — measured results

End-to-end 2P2D serving results for the image in this directory. For the **kernel-level**
dispatch/combine bandwidth of the same DeepEP-EFA build — including the 8-to-32-node scaling runs
and the NCCL/UCCL cross-backend comparison — see
[`micro-benchmarks/expert-parallelism/deepep-benchmark`](../../../../../micro-benchmarks/expert-parallelism/deepep-benchmark)
and [`ep-backend-comparison`](../../../../../micro-benchmarks/expert-parallelism/ep-backend-comparison).
For the same model and fabric served by SGLang, see
[`3.test_cases/pytorch/sglang/dsr1-deepep-efa`](../../../sglang/dsr1-deepep-efa/benchmarks/RESULTS.md).

## Read this first

**What this sample establishes:** DeepSeek-R1 serves end to end on vLLM with DeepEP MoE
dispatch/combine running over EFA — a 4-node PD-disaggregated deployment, KV moving
prefill→decode over EFA via NixlConnector, with correct output verified through the router.
Before this port DeepEP could not use EFA at all, because NVSHMEM offered only the IBRC/IBGDA
InfiniBand transports.

**What it does not establish: a DeepEP win.** At 2 nodes per role / 16 GPUs, each GPU already owns
16 of 256 experts, the MoE fan-out is small and largely intra-node NVLink, and DeepEP's per-layer
dispatch/combine launches plus NVSHMEM RDMA setup cost more than they save. DeepEP targets
large-scale EP — tens of nodes, experts spread thin enough that every token must cross the fabric.
That crossover is not reachable here.

## Caveats — please read before quoting any number

1. **Single-seed smoke benchmarks.** Each point is one sweep pass. Directional, not statistically
   tight.
2. **Both decode tables below ran `--enforce-eager` on the decode role, i.e. with CUDA graphs
   OFF.** That is *not* a fair vLLM decode configuration and it inflates TPOT substantially
   (~120 ms here vs ~25–50 ms for SGLang in the sibling sample). `recipe/serve-pd.sh` therefore
   ships the *correct* config — `--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}` on
   decode, `--enforce-eager` on prefill — which is **not the config these numbers came from**. The
   cudagraph re-run was not completed before the cluster was released. Treat every decode number
   here as a **slow lower bound**, and do not compare it against the SGLang sample's decode table.
3. **vLLM DeepEP prefill data is incomplete** (see the H100 table). The out=1 PD path repeatedly
   drove the DeepEP prefill engine to `EngineDeadError` (`sample_tokens timed out`).
4. **The two tables are different hardware and different sweeps**, not a controlled A/B.

---

## H100 — 2P2D, vLLM 0.23.0, DeepEP vs baseline

**Cluster:** 4× `p5.48xlarge` (8×H100 80GB HBM3, sm_90), us-east-2, 32 EFA NICs/node, iface
`enp71s0`. Prefill role = nodes 1+2 (TP16/EP16), decode role = nodes 3+4 (TP16/EP16). KV via
NixlConnector over EFA; `vllm-router --vllm-pd-disaggregation`. `--enforce-eager` on **both**
roles (see caveat 2). Date: 2026-06-23.

`vLLM-DeepEP` = `VLLM_ALL2ALL_BACKEND=deepep_high_throughput` (prefill) /
`deepep_low_latency` (decode) + `VLLM_USE_DEEP_GEMM=1` = `MOE_BACKEND=deepep`.
`vLLM-baseline` = vLLM's default all-to-all, still `--enable-expert-parallel` =
`MOE_BACKEND=baseline`.

### Decode sweep (input 256, output 128)

Output tok/s (higher better) / mean TPOT ms (lower better). **Bold = best in row.**

| Concurrency | vLLM-DeepEP | vLLM-baseline |
|---|---|---|
| 8  | 59.4 / **119.6** | **60.9** / 119.8 |
| 16 | 115.5 / 120.2 | **124.9** / **119.7** |
| 32 | 176.4 / 142.9 | **193.5** / **137.8** |

All requests successful in both. **At this scale the all-to-all backend barely matters for vLLM
decode** — baseline ≈ DeepEP, within a few percent. Both ran cudagraph-off, so decode is
engine-bound and the backend difference is masked; this is the main reason caveat 2 matters.

### Prefill-only sweep (output 1)

| Input × conc | vLLM-baseline TTFT | vLLM-DeepEP TTFT |
|---|---|---|
| 1024 × 1 | 320 ms | not obtained |
| 4096 × 1 | 321 ms | not obtained |
| 8192 × 1 | 368 ms | not obtained |

The DeepEP prefill engine died with `EngineDeadError` (`sample_tokens timed out`) on the out=1 PD
path, repeatably. Baseline completed. This is an open issue in this sample, not a measured
DeepEP-is-slower result.

---

## H200 — 2P2D, vLLM 0.23.0, DeepEP only (run cut short)

**Cluster:** 4× `p5en.48xlarge` (8×H200 141GB, sm_90), 16 EFA NICs/node. Same topology and same
`--enforce-eager` caveat. Verified end-to-end: a real completion returned through the router with
the prefill→decode path confirmed by request id. Date: 2026-06-22.

Decode, input 256, concurrency 32, output length varied:

| Output len | Output tok/s | Mean TPOT |
|---|---|---|
| 256  | 199 | 135 ms |
| 512  | 243 | 128 ms |
| 1024 | 246 | 128 ms |
| 2048 | 203 | 117 ms |

No baseline arm and no prefill sweep — the cluster was released mid-run. Included only because it
independently confirms the image works on H200 as well as H100.

---

## Operational notes (vLLM-specific, hard-won)

These cost real debugging time and are encoded in the recipe scripts:

- **Multi-node TP needs Ray.** `vllm serve --nnodes/--node-rank` (the SGLang-style launch) fails
  with `collective_rpc should not be called on follower node`. Use one Ray cluster per role, then
  `vllm serve --tensor-parallel-size 16 --distributed-executor-backend ray` once on the head.
- **Never restart an engine inside a live Ray cluster.** `pkill`-ing vLLM leaks the GPU placement
  group, and the next `vllm serve` cannot obtain 16 GPUs. Tear the role's Ray cluster down and
  recreate it between configs (`recipe/serve-pd.sh stop` then `raystart`).
- **`vllm-router` is a separate PyPI package**, not part of the vLLM image. The bundled
  `examples/disaggregated/.../disagg_proxy_demo.py` collapses at concurrency ≥ 16. This
  Dockerfile bakes `vllm-router` in.
- **The load generator must not pass `--pd-separated`.** That flag speaks `sglang_router`'s PD
  protocol; `vllm-router` is a plain OpenAI-compatible endpoint. (The SGLang sample in this repo
  *does* need it — do not copy that invocation across.)
- **JSON-valued flags must not cross a nested shell.** `--kv-transfer-config` and
  `--compilation-config` lose their quotes through `docker exec bash -c`, and vLLM rejects them
  with `Invalid JSON: key must be a string`. `recipe/serve-pd.sh` writes the launch line to a file
  inside the container instead. The original workaround for this was to fall back to
  `--enforce-eager` on decode, which is exactly how caveat 2 came about.
- **vLLM's DeepGEMM warmup takes ~15 min per role** with DeepEP enabled (~1664 kernels) and has
  no pre-cache path equivalent to SGLang's `compile_deep_gemm`. Mount a persistent
  `/root/.cache` (the recipe does) so the cost is paid once per host.
- **Fabric Manager version skew** on fresh nodes (FM newer than the driver) surfaces as CUDA error
  802 behind a misleading NCCL/TCPStore cascade. Match FM to the driver and reset wedged GPUs
  before blaming the transport.

## To actually demonstrate a DeepEP win

Two things are needed, in this order: (1) re-run decode with CUDA graphs on the decode role, which
`recipe/serve-pd.sh` already configures, to get a fair vLLM baseline at all; then (2) scale to
**≥8–16 nodes** so experts spread across many nodes and the all-to-all becomes network-bound. The
scaling behaviour of the kernels themselves out to 32 nodes / 256 ranks is already characterised
in `ep-backend-comparison`.
