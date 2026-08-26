<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0 -->
# Benchmarks — TensorRT-LLM NcclEP over EFA

## Status: correctness measured, performance NOT yet published

What **was** measured (2026-08-07, 2× p5en.48xlarge / H200, same component versions as this image
with the rc24 NcclEP modules grafted onto an rc9 container — see the sample README "Known
limitations"):

- A real `trtllm-serve` (Qwen/Qwen3-30B-A3B, single-node EP8) reached `Application startup
  complete`, logged `NCCL EP group created` on all 8 ranks, and answered `/v1/chat/completions`
  correctly (including an arithmetic check — MoE misrouting produces plausible text but wrong
  arithmetic).
- The 16-rank cross-node probe (`recipe/run-kernel-test.sh`) passed 16/16 with zero
  illegal-memory-access and the `efa-direct` provider banner on every rank, under **both** the
  upstream LOW_LATENCY/RANK_MAJOR default and (on a patched image) HIGH_THROUGHPUT/FLAT, with
  combine relmax ≈ 0.003–0.005 against the oracle.

What was **not** measured, and is therefore not claimed here:

- **No throughput/latency tables are published in this folder yet.** Run `recipe/benchmark.sh`
  on your own deployment; treat a single pass as directional.
- **No alternative-backend baseline was measured** — nothing here compares NcclEP against
  TRT-LLM's default all-to-all or any other EP backend, so no relative-performance claim is made.
- No p5/H100 numbers; no multi-node *served* numbers (single-node serve boundary — README trap 8).

## Methodology (what `benchmark.sh` runs and why)

`recipe/benchmark_probe.py` sweeps fixed concurrency levels against the live OpenAI endpoint:

- **Requests per level = concurrency × 5** so p50/p90/p99 describe a distribution, not one shot.
- **Successes only** enter the percentiles and the token numerator; any failure fails the whole
  run (exit ≠ 0) — a partially-dead serve must not report a *better* p50 because refused
  connections return fast.
- **`ignore_eos: true`** pins generated tokens == `max_tokens`, fixing the tok/s denominator by
  construction.
- **Unique prompt prefix per request** (the index goes first) so a prefix/KV cache cannot serve
  request N's prefill from request 1's.
- A 200 response **without a `usage` block is a failure** (`200-no-usage`), not a zero-token
  success that silently deflates throughput.
- The probe's **exit code is the sole pass/fail authority** — `benchmark.sh` adds no separate
  grep gate that could disagree with it.
- Default levels are `[1, 2, 4]`, sized to the measured serve shape (`--max_batch_size 4`);
  raise `SERVE_MAX_BATCH_SIZE` / `SERVE_MAX_NUM_TOKENS` in `serve.sh` before sweeping higher,
  and record both.

## Provenance knobs (record these next to any numbers you publish)

| Knob | Default | Why it belongs in the table |
|---|---|---|
| `TRTLLM_NCCL_EP_ALGO` (+ layout) | `LOW_LATENCY` (+RANK_MAJOR) | different EP kernels entirely; patched image required for non-default (README trap 4) |
| `NCCL_EP_NUM_QP_PER_RANK` | 32 | QP fan-out per rank on EFA — throughput-relevant |
| `SERVE_TP` / `SERVE_EP` | 8 / 8 | the parallel shape |
| `SERVE_MAX_BATCH_SIZE` / `SERVE_MAX_NUM_TOKENS` / `SERVE_MAX_SEQ_LEN` | 4 / 2048 / 2048 | serving-engine ceilings that bound every level |
| model | `Qwen/Qwen3-30B-A3B` | expert count/hidden size change the all-to-all shape |
| instance type / EFA NICs | p5en.48xlarge / 16 | fabric budget per node |

Raw sweeps land in `benchmarks/raw/<timestamp>/` (gitignored — publish tables here, keep raw
JSONL out of the repo).
