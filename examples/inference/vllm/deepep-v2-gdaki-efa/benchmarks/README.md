<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0 -->
# Benchmarks — vLLM + DeepEP-V2 over EFA on the GDAKI (GPU-initiated) transport

Measured 2026-08-14 on 2× and 4× p5en.48xlarge (H200), `Qwen/Qwen3-30B-A3B-FP8`, DP16/EP16 and
DP32/EP32. `recipe/benchmark.sh` fires N concurrent `/v1/chat/completions` per level and reports
aggregate output tok/s + per-request latency percentiles. Raw JSONL lands in `raw/` (gitignored).

## Environment provenance

| | |
|---|---|
| Instance | 2× / 4× p5en.48xlarge (H200), cross-node EFA (16 EFA NICs/node) |
| Transport | DeepEP-V2 `ElasticBuffer`, NCCL-GIN **GDAKI** (`NCCL_GIN_TYPE=3`, `OFI_NCCL_GIN_GDAKI=1`), `efa-direct` |
| Model | `Qwen/Qwen3-30B-A3B-FP8`, DP16/EP16 and DP32/EP32 (`--enable-expert-parallel --all2all-backend deepep_v2`) |
| vLLM | `0.22.1rc1.dev283+ge2f993dc4` (first commit with the `deepep_v2` backend, [PR#41183](https://github.com/vllm-project/vllm/pull/41183)) |
| Stack | torch 2.11.0+cu130, nvidia-nccl-cu13 2.30.4, DeepEP `b306af06`+PR#612 (upstream, open), aws-ofi-nccl `a3d2680` `--enable-gdaki` (SHA pin), rdma-core post-#1701, libfabric post-#12591 — zero local source patches |
| Node efa.ko | **3.0.0g / 3.1.0g (heterogeneous)** on the validation cluster → `OFI_NCCL_GDAKI_EFA_HW_COUNTER=off` (< 3.3.0) |
| Probe | `recipe/benchmark_probe.py` (stdlib urllib+threads, 127.0.0.1 loopback), `max_tokens=128`, `temperature=0.0` (greedy), concurrency 1/8/16/32/64 |
| Date | 2026-08-14 |

## Serve results — eager (`--enforce-eager`), DP16/EP16

121/121 HTTP 200 across the sweep; 31/31 ramp + 13/13 manual gate requests; zero `GDAKI-CQE … status 9` events.

| conc | ok | agg tok/s | p50 latency (128-tok greedy) |
|---:|---:|---:|---:|
| 1  | 1/1   | 5.5   | 23.24 s |
| 8  | 8/8   | 43.2  | 23.70 s |
| 16 | 16/16 | 87.2  | 23.48 s |
| 32 | 32/32 | 174.9 | 23.40 s |
| 64 | 64/64 | 351.1 | 23.29 s |

Near-linear concurrency scaling at flat per-request latency.

## Serve results — non-eager (historical; measured with the then-unmerged upstream guard, [vLLM #52632](https://github.com/vllm-project/vllm/pull/52632)), DP16/EP16

Ramp 31/31 (c=1→16, no wedge), coherence 200, sweep 121/121 HTTP 200, zero status-9.

| conc | ok | agg tok/s | p50 latency | vs eager |
|---:|---:|---:|---:|---:|
| 1  | 1/1   | 5.5   | 23.18 s | ±0% |
| 8  | 8/8   | 44.3  | 22.92 s | +2.5% |
| 16 | 16/16 | 83.4  | 23.97 s | −4.4% |
| 32 | 32/32 | 172.3 | 23.56 s | −1.5% |
| 64 | 64/64 | 260.5 | 31.11 s | **−25.8%** |

**Reading:** non-eager matches eager through concurrency 32, then falls ~26% behind at 64 on this shape
(the non-eager wall jumps at c=64 while eager stays flat). This replicates the proxy package's
eager-vs-non-eager finding on *its* transport (parity through c=32, a c=64 penalty). Mechanism not
root-caused here. Guidance: **eager is the zero-patch default; if you flip to non-eager for production,
measure at your target concurrency** — the crossover is workload-dependent.

## Transport A/B — GDAKI (TYPE=3) vs CPU-proxy (TYPE=2), same node-set, env-flip only

**The clean comparison.** Same nodes, same image, same day, same probe — **only the transport env
flipped** (`NCCL_GIN_TYPE` 3↔2, `OFI_NCCL_GIN_GDAKI` 1↔0). This removes the different-pair AND
different-image variables. Both arms eager, both runtime-env value-verified in the worker processes.

### DP16/EP16 (2 nodes) — both arms 121/121 HTTP 200

| conc | GDAKI (TYPE=3) tok/s | CPU-proxy (TYPE=2) tok/s | GDAKI Δ |
|---:|---:|---:|---:|
| 1  | 5.5   | 5.4   | +1.9% |
| 8  | 43.2  | 42.5  | +1.6% |
| 16 | 87.2  | 84.8  | +2.8% |
| 32 | 174.9 | 172.4 | +1.5% |
| 64 | 351.1 | 341.9 | +2.7% |

p50 latency at c=64: GDAKI 23.29 s vs proxy 23.92 s (proxy ~0.6 s higher).

### DP32/EP32 (4 nodes) — both arms 121/121 HTTP 200

Scale-up was launcher-args only (replicas 4 + `SERVE_DP 32`), same as the proxy package found. Same 4
nodes for both arms, same image, env-flip only.

| conc | GDAKI-EP32 tok/s | proxy-EP32 tok/s | GDAKI Δ |
|---:|---:|---:|---:|
| 1  | 3.2   | 3.1   | +3.2% |
| 8  | 25.4  | 25.1  | +1.2% |
| 16 | 50.4  | 50.3  | +0.2% |
| 32 | 101.3 | 100.1 | +1.2% |
| 64 | 204.9 | 199.3 | +2.8% |

**Across the full 2×2 (2 scales × 2 transports, 4 independent sweeps, 10 paired comparisons) GDAKI
reads ≥ proxy in all 10, +0.2–3.2%.** The consistent direction across scales bounds the single-sweep
variance concern; the magnitude stays small — the serve is compute/latency-bound at these shapes, so
the transport delta is modest (the kernel-level combine gap is mostly hidden behind overlap). EP32
per-request p50 is ~40 s vs ~23 s at EP16 for this 30B model — EP32's value here is model capability
(experts that don't fit 2 nodes), not throughput at this shape; same conclusion as the proxy package.

## Kernel-level microbench (context only — a different lineage; do not conflate)

A same-day, same-pair DeepEP-V2 `test_ep` microbench (`N=128` decode, 16 ranks, `torch.equal`-class
correctness green) measured the GPU-initiated **combine** at −10.3% vs CPU-proxy, reproducing a
July-2026 banked envelope (−11.5%). The GPU-initiated **dispatch** win did **not** reproduce on this
efa.ko-heterogeneous (3.0.0g/3.1.0g) node pair. Note: those microbench numbers use a research *fork*
of the GDAKI slice, a **separate lineage** from the upstream `a3d2680 --enable-gdaki` plugin this
sample ships; the serve-level numbers above are the ones measured with THIS image. Keep the lineages
distinct when quoting.

## Honest caveats (do not drop when publishing)

- **Single sweep per arm/scale — variance not statistically bounded.** The 2×2 consistency (10/10
  GDAKI ≥ proxy) demonstrates a direction, not a tight interval.
- **Wire-proof is functional, not a byte-tally.** The validation nodes ran efa.ko 3.0.x/3.1.x, which
  expose **no** `/sys/.../hw_counters` — transport proof is the `efa-direct` boot banner (×8 ranks) +
  coherent EP16/EP32 cross-node output, not a `rdma_write_bytes` delta. A byte-level GDAKI hw-counter
  validation needs node efa.ko ≥ 3.3.0 (see `../README.md` node preconditions).
- Fixed 128-token greedy decode; the ~23–31 s wall reflects 128-token generation, not a
  latency-optimized single token. These are **at-scale throughput + relative-latency** datapoints, not
  tuned TTFT baselines.
- Both modes/arms: 0 errors, 0 crashes, all HTTP 200. `raw/` holds the per-level JSONL from the sweeps.
