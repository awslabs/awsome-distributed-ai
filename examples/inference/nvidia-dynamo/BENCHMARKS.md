# Benchmarks

All runs use the official AIPerf concurrency sweep on the HyperPod GPU workers
(**1x NVIDIA L40S 48GB per worker**, `ml.g6e.4xlarge`), **ISL 200 / OSL 256**,
forced output length (`ignore_eos` + `min_tokens`). Disaggregated runs transfer
the KV cache prefill→decode over **NIXL (TCP** on g6e — no RDMA).

Metrics: **TTFT** = time to first token (avg, ms) · **ITL** = inter-token latency
(avg, ms) · **tok/s** = aggregate output throughput.

Reproduce: `./scripts/08-benchmark.sh <scenario>` → saves to `bench_results/<scenario>/`.

## GPT-OSS-20B

### Aggregated (`gpt-oss-agg`)
| Concurrency | TTFT (ms) | ITL (ms) | Output tok/s |
|---:|---:|---:|---:|
| 1  | 595   | 7.13  | 105.6 |
| 2  | 1,180 | 7.18  | 162.9 |
| 5  | 2,772 | 10.50 | 192.6 |
| 10 | 2,907 | 13.45 | 398.3 |
| 50 | 604   | 13.29 | 608.1 |

### Disaggregated (`gpt-oss-disagg`)
| Concurrency | TTFT (ms) | ITL (ms) | Output tok/s |
|---:|---:|---:|---:|
| 1  | 1,871 | 7.16  | 68.7  |
| 2  | 570   | 9.18  | 175.1 |
| 5  | 2,136 | 13.29 | 230.1 |
| 10 | 1,715 | 15.65 | 446.9 |
| 50 | 1,502 | 15.62 | 462.2 |

**Takeaways (GPT-OSS-20B):** ITL 7–16 ms (≈60–140 tok/s/user) on L40S.
Disaggregation gives lower TTFT under load (e.g. 1,715 vs 2,907 ms at c=10)
at the cost of slightly higher ITL; aggregated wins ITL. Both scale throughput with
concurrency (continuous batching). gpt-oss runs the **triton** attention backend
(L40S constraint) with native MXFP4 weights.

## Qwen3.6-27B-FP8

> Hybrid (Gated DeltaNet linear-attention + full-attention) multimodal model, FP8.
> Decode is dominated by the SSM/linear-attention state, so per-token latency is
> higher than a standard-attention model. The disaggregated decode worker reserves a
> per-slot mamba/SSM state — on a 48GB L40S you **must cap `--max-running-requests`**
> (16 here) and lower `--mem-fraction-static` or it OOMs.

### Aggregated (`qwen3.6-agg`)
| Concurrency | TTFT (ms) | ITL (ms) | Output tok/s |
|---:|---:|---:|---:|
| 1  | 1,416 | 52.0 | 17.4  |
| 2  | 338   | 46.9 | 41.6  |
| 5  | 384   | 48.5 | 100.3 |
| 10 | 1,941 | 50.4 | 97.0  |

### Disaggregated (`qwen3.6-disagg`)
| Concurrency | TTFT (ms) | ITL (ms) | Output tok/s |
|---:|---:|---:|---:|
| 1  | 721   | 48.8 | 19.5  |
| 2  | 867   | 45.5 | 40.8  |
| 5  | 2,427 | 47.1 | 87.3  |
| 10 | 4,426 | 49.4 | 140.5 |

**Takeaways (Qwen3.6-27B):** ITL ≈45–52 ms (≈20 tok/s/user) — the hybrid linear/SSM
decode is compute-heavy on L40S. Disaggregation improves low-concurrency TTFT
(721 vs 1,416 ms at c=1) and pushes higher aggregate throughput at c=10 (140 vs 97
tok/s). The FP8 weights (~27 GB) fit a single L40S; the hybrid design keeps the
attention KV cache small (only ~1 in 4 layers is full-attention).

## Notes
- Higher concurrency raises aggregate throughput but per-user latency degrades — the
  usual continuous-batching tradeoff.
