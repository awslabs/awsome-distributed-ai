<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0 -->
# Benchmarks — vLLM + DeepEP-V2 over EFA: CPU-proxy serving numbers (2026-08-14); the sample is built for GDAKI, not yet shown on hardware

Measured 2026-08-14 on 2× and 4× p5en.48xlarge (H200), `Qwen/Qwen3-30B-A3B-FP8`, DP16/EP16 and
DP32/EP32. `recipe/benchmark.sh` fires N concurrent `/v1/chat/completions` per level and reports
aggregate output tok/s + per-request latency percentiles. Raw JSONL lands in `raw/` (gitignored).

> **Read this first: every table here is CPU proxy, not GDAKI. In the env-flip A/B both arms ran the
> same transport configuration, so the A/B shows no difference.** The tables were recorded with `NCCL_GIN_TYPE=3`
> on NCCL 2.30.4 + aws-ofi-nccl `a3d2680` (+ the #1351 gdrcopy cherry-pick, which does not touch the GIN
> tables). NCCL 2.30.4 can load only a plugin's v13, v12 or v11 GIN table (`src/plugin/gin.cc:27` at
> `v2.30.4-1`) and takes the GIN type from that table, not from `NCCL_GIN_TYPE`
> (`src/gin/gin_host.cc:48-71`). This plugin's v13 table is the CPU-proxy one
> (`NCCL_NET_DEVICE_GIN_PROXY`, `src/rdma/gin/nccl_ofi_gin_api.cpp:468` at `a3d2680`). Evidence: a
> same-day `NCCL_DEBUG=INFO` boot log from one non-eager DP16/EP16 serve of an image built from the same
> pins shows `NET/Plugin: Loaded gin plugin Libfabric (v13)` (line 351) and `Assigned GIN plugin
> Libfabric to comm` (line 354), and the same pair on all 8 ranks it captured (one node, 8 of the 16).
> The numbers stay as recorded.

## Environment provenance

| | |
|---|---|
| Instance | 2× / 4× p5en.48xlarge (H200), cross-node EFA (16 EFA NICs/node) |
| Transport | DeepEP-V2 `ElasticBuffer` over the NCCL-GIN **CPU proxy** (the plugin's v13 table), `efa-direct`. `NCCL_GIN_TYPE=3` (GDAKI) was set, but NCCL 2.30.4 does not apply it to an external plugin (see the note above). The measured env also exported `OFI_NCCL_GIN_GDAKI=1`, `OFI_NCCL_GIN_MAX_REQUESTS=512`, `DEEP_EP_BACKEND=nccl` and `EP_EFA_MAX_QPS=2` / `EP_EFA_RDMA_GBS=25.0`; at the pinned aws-ofi-nccl `a3d2680` the first three are not parameters (GDAKI support is detected at configure time; the dev-line `OFI_NCCL_GIN_GDAKI` knob had been removed) and nothing reads the fourth, and the two `EP_EFA_*` knobs belong to the deepseek PR#612 tree the tables were measured on — none of the five exists (or is set) in the current recipe. |
| Model | `Qwen/Qwen3-30B-A3B-FP8`, DP16/EP16 and DP32/EP32 (`--enable-expert-parallel --all2all-backend deepep_v2`) |
| vLLM | `0.22.1rc1.dev283+ge2f993dc4` (first commit with the `deepep_v2` backend, [PR#41183](https://github.com/vllm-project/vllm/pull/41183)) |
| Stack | torch 2.11.0+cu130, nvidia-nccl-cu13 2.30.4, DeepEP `deepseek-ai/DeepEP@b306af06` + PR#612 (open, unmerged) applied (**the measured tree**; the shipped recipe now pins the `amazon-contributing/DeepEP` fork at `97d8f9bc`, which carries both #612 halves in-code — see `../README.md`), aws-ofi-nccl `a3d2680` + the [#1351](https://github.com/aws/aws-ofi-nccl/pull/1351) cherry-pick `c2e773df` (gdrcopy forced-PCIe override; no GIN change), configured with `--enable-gdaki` (a no-op at `a3d2680`); a local v13-host-uc shim patch on the DeepEP tree (since removed); rdma-core post-#1701, libfabric post-#12591 |
| Node efa.ko | **3.0.0g / 3.1.0g (heterogeneous)** on the 2026-08-14 cluster → `OFI_NCCL_GDAKI_EFA_HW_COUNTER=off` (< 3.3.0) |
| Probe | `recipe/benchmark_probe.py` (stdlib urllib+threads, 127.0.0.1 loopback), `max_tokens=128`, `temperature=0.0` (greedy), concurrency 1/8/16/32/64 |
| Date | 2026-08-14 |

## Serve results (CPU proxy) — eager (`--enforce-eager`), DP16/EP16

121/121 HTTP 200 across the sweep; 31/31 ramp + 13/13 manual gate requests; zero `GDAKI-CQE … status 9` events (no GDAKI path ran).

| conc | ok | agg tok/s | p50 latency (128-tok greedy) |
|---:|---:|---:|---:|
| 1  | 1/1   | 5.5   | 23.24 s |
| 8  | 8/8   | 43.2  | 23.70 s |
| 16 | 16/16 | 87.2  | 23.48 s |
| 32 | 32/32 | 174.9 | 23.40 s |
| 64 | 64/64 | 351.1 | 23.29 s |

Near-linear concurrency scaling at flat per-request latency.

## Serve results (CPU proxy) — non-eager (historical; measured with the then-unmerged upstream guard, [vLLM #52632](https://github.com/vllm-project/vllm/pull/52632)), DP16/EP16

Ramp 31/31 (c=1→16, no wedge), coherence 200, sweep 121/121 HTTP 200, zero status-9 (no GDAKI path ran).

| conc | ok | agg tok/s | p50 latency | vs eager |
|---:|---:|---:|---:|---:|
| 1  | 1/1   | 5.5   | 23.18 s | ±0% |
| 8  | 8/8   | 44.3  | 22.92 s | +2.5% |
| 16 | 16/16 | 83.4  | 23.97 s | −4.4% |
| 32 | 32/32 | 172.3 | 23.56 s | −1.5% |
| 64 | 64/64 | 260.5 | 31.11 s | **−25.8%** |

**Reading:** non-eager matches eager through concurrency 32, then falls ~26% behind at 64 on this shape
(the non-eager wall jumps at c=64 while eager stays flat). This replicates the proxy package's
eager-vs-non-eager finding (parity through c=32, a c=64 penalty). Mechanism not
root-caused here. Guidance: **eager is the zero-patch default; if you flip to non-eager for production,
measure at your target concurrency** — the crossover is workload-dependent.

## Env-flip A/B (NCCL_GIN_TYPE 3 vs 2): CPU proxy, both arms the same transport configuration; the A/B shows no difference

Both arms ran the plugin's v13 CPU-proxy GIN table on NCCL 2.30.4 (see the note at the top). The two env
values that were flipped, `NCCL_GIN_TYPE` 3↔2 and `OFI_NCCL_GIN_GDAKI` 1↔0, have no effect on this stack
(NCCL 2.30.4 does not apply the first to an external plugin, and `a3d2680` does not read the second), so
both arms had the same transport configuration and the deltas below are between two runs of it. As
recorded in the run notes: same nodes, same image, same day, same probe, both arms eager, both
runtime-env value-verified in the worker processes. Whether both arms (at either scale) used the
identical image build is not recorded: no log kept with these tables identifies the build each arm ran.

### DP16/EP16 (2 nodes) — both arms 121/121 HTTP 200

| conc | TYPE=3 tok/s | TYPE=2 tok/s | Δ |
|---:|---:|---:|---:|
| 1  | 5.5   | 5.4   | +1.9% |
| 8  | 43.2  | 42.5  | +1.6% |
| 16 | 87.2  | 84.8  | +2.8% |
| 32 | 174.9 | 172.4 | +1.5% |
| 64 | 351.1 | 341.9 | +2.7% |

p50 latency at c=64: TYPE=3 23.29 s vs TYPE=2 23.92 s (TYPE=2 ~0.6 s higher).

### DP32/EP32 (4 nodes) — both arms 121/121 HTTP 200

Scale-up was launcher-args only (replicas 4 + `SERVE_DP 32`), same as the proxy package found. Same 4
nodes for both arms, same image, env-flip only.

| conc | TYPE=3 EP32 tok/s | TYPE=2 EP32 tok/s | Δ |
|---:|---:|---:|---:|
| 1  | 3.2   | 3.1   | +3.2% |
| 8  | 25.4  | 25.1  | +1.2% |
| 16 | 50.4  | 50.3  | +0.2% |
| 32 | 101.3 | 100.1 | +1.2% |
| 64 | 204.9 | 199.3 | +2.8% |

**Across the full 2×2 (2 scales × 2 env settings, 4 independent sweeps, 10 paired comparisons): CPU
proxy, both arms the same transport configuration; the A/B shows no difference.** The per-row deltas
(+0.2–3.2%) are between two runs of one transport configuration, one sweep each, and are not evidence of
a transport effect. EP32
per-request p50 is ~40 s vs ~23 s at EP16 for this 30B model — EP32's value here is model capability
(experts that don't fit 2 nodes), not throughput at this shape; same conclusion as the proxy package.

## Kernel-level microbench (not this sample; not cited)

A same-day DeepEP-V2 `test_ep` microbench ran on the same node pair with a separate research fork of
the GDAKI slice, a different lineage from the upstream aws-ofi-nccl `a3d2680` plugin this sample
ships. The record kept with these tables does not include the log lines in
[How to confirm GDAKI is active](../README.md#how-to-confirm-gdaki-is-active), so whether GDAKI was
active in it is not established here, and its numbers are not cited. The serve-level numbers above
were measured with this sample's 2026-08-14 image (see the provenance table), not with the current
pin set.

## Honest caveats (do not drop when publishing)

- **Single sweep per arm/scale; variance not statistically bounded.** CPU proxy, both arms the same
  transport configuration; the A/B shows no difference, and its per-row deltas are not a transport effect.
- **Wire-proof is functional, not a byte-tally.** The 2026-08-14 nodes ran efa.ko 3.0.x/3.1.x, which
  expose **no** `/sys/.../hw_counters` — transport proof is the `efa-direct` boot banner (×8 ranks) +
  coherent EP16/EP32 cross-node output, not a `rdma_write_bytes` delta. A byte-level GDAKI hw-counter
  validation needs node efa.ko ≥ 3.3.0 (see `../README.md` node preconditions).
- Fixed 128-token greedy decode; the ~23–31 s wall reflects 128-token generation, not a
  latency-optimized single token. These are **at-scale throughput + relative-latency** datapoints, not
  tuned TTFT baselines.
- Both modes/arms: 0 errors, 0 crashes, all HTTP 200. `raw/` holds the per-level JSONL from the sweeps.
