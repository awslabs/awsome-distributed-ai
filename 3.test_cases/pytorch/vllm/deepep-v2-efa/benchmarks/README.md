<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0 -->
# Benchmarks — vLLM + DeepEP-V2 over EFA (eager + non-eager)

Concurrency sweep of the live DP16/EP16 serve, both execution modes, same pods / same day / same probe.
`recipe/benchmark.sh` fires N concurrent `/v1/chat/completions` per level and reports aggregate output
tok/s + per-request latency percentiles. Raw JSONL lands in `raw/` (gitignored).

## Environment provenance
| | |
|---|---|
| Instance | 2× p5en.48xlarge (H200), cross-node EFA |
| Transport | DeepEP-V2 `ElasticBuffer`, NCCL-GIN CPU-proxy (`NCCL_GIN_TYPE=2`, `OFI_NCCL_GIN_GDAKI=0`), `efa-direct` |
| Model | `Qwen/Qwen3-30B-A3B-FP8`, DP16/EP16 (`--enable-expert-parallel --all2all-backend deepep_v2`) |
| vLLM | `0.22.1rc1.dev283+ge2f993dc4` (first commit with the `deepep_v2` backend, [PR#41183](https://github.com/vllm-project/vllm/pull/41183)) |
| Stack | torch 2.11.0+cu130, nvidia-nccl-cu13 2.30.4, DeepEP `b306af06`+PR#612, aws-ofi-nccl GIN `9c44d34`+#1354 |
| Serve fingerprint | `vllm-0.22.1rc1.dev283+ge2f993dc4-dp16-ep-1f3ed125` |
| Probe | `recipe/benchmark.sh` (stdlib urllib+threads, 127.0.0.1 loopback), `max_tokens=128`, `temperature=0.0` (greedy), concurrency 1/8/16/32/64 |
| Date | 2026-08-14 |

## Results

### Eager (`--enforce-eager`) — 121/121 HTTP 200
| conc | agg tok/s | wall s | p50 s | codes |
|---|---|---|---|---|
| 1 | 4.8 | 26.91 | 26.91 | 200 |
| 8 | 37.1 | 27.63 | 27.63 | 200 |
| 16 | 74.6 | 27.45 | 27.43 | 200 |
| 32 | 151.9 | 26.97 | 26.95 | 200 |
| 64 | 301.8 | 27.15 | 27.11 | 200 |

### Non-eager (default compilation + the 3-commit fix stack) — 153/153 HTTP 200
| conc | agg tok/s | wall s | p50 s | codes |
|---|---|---|---|---|
| 1 | 5.0 | 25.48 | 25.48 | 200 |
| 8 | 39.9 | 25.69 | 25.68 | 200 |
| 16 | 76.9 | 26.64 | 26.19 | 200 |
| 32 | 153.9 | 26.61 | 26.17 | 200 |
| 64 | 239.7 | 34.18 | 33.91 | 200 |

### Eager vs non-eager (agg tok/s; delta = non-eager relative to eager)
| conc | eager | non-eager | delta |
|---|---|---|---|
| 1 | 4.8 | 5.0 | +4.2% |
| 8 | 37.1 | 39.9 | +7.5% |
| 16 | 74.6 | 76.9 | +3.1% |
| 32 | 151.9 | 153.9 | +1.3% |
| 64 | 301.8 | 239.7 | −20.6% (wall 27.15s → 34.18s) |

**Reading:** non-eager matches-or-slightly-beats eager through concurrency 32, then falls ~21% behind at
64 on this shape (the non-eager wall jumps 27→34 s at c=64 while eager stays flat). Mechanism not
root-caused here — candidates are CUDA-graph capture-size coverage vs per-engine batch shape at high
concurrency. Guidance: **eager is the zero-patch default; if you flip to non-eager for production,
measure at your target concurrency** — the crossover is workload-dependent.

## Honest caveats
- Pods were ~1 day old at measurement (accumulated-state can inflate latency; a fresh-pod baseline would
  differ). Fixed 128-token greedy decode. **Single sweep per mode — no variance bars; the c=64 non-eager
  delta was reproduced once.** These are at-scale **throughput + relative-latency** datapoints, not tuned
  TTFT baselines: the ~26–34 s wall reflects 128-token generation, not a latency-optimized single token.
- Both modes: 0 errors, 0 crashes, all HTTP 200. `raw/` holds the per-level JSONL from the sweeps.
