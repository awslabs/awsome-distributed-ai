<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Qwen3-30B-A3B — NCCL vs DeepEP+UCCL vs DeepEP+NVSHMEM — Benchmark Results (p5.48xlarge / H100)

Three-way MoE token-dispatcher comparison on **8 × `p5.48xlarge` (H100-80GB, sm_90, 32× EFA /
node)**, using **Qwen3-30B-A3B** (128 experts, top-8, 48 layers, hidden 2048). This is the H100
companion to the B300 / Qwen3-235B results in [`../../qwen3-235b/benchmarks/RESULTS.md`](../../qwen3-235b/benchmarks/RESULTS.md).
Metric of record: mean training-iteration time over 8 steady iters (first 4 of 12 dropped);
`Δ` vs the NCCL all-to-all baseline (− = DeepEP faster, + = slower).

| Field | Value |
|---|---|
| Date | 2026-06-22 |
| Hardware | 8 × `p5.48xlarge` (64× H100-80GB, sm_90, 32× EFA / node ≈ 3.2 Tbps), EKS `ml-shared` |
| Model | Qwen3-30B-A3B — 128 experts, top-8, 48 layers, hidden 2048, bf16 |
| World | 64 ranks, TP8, ETP1, balanced routing, **recompute off** (30B fits H100), gbs 256 |
| EP sweep | **EP16 = TP8/PP4/DP2** (EP spans 2 nodes) · **EP32 = TP8/PP2/DP4** (spans 4) — both **EP>8 ⇒ internode** EFA |
| Images | `…-uccl-0dc87eb-sm90` · `…-deepep-nvshmem-567632d-cu13-sm90` (Hopper rebuilds) |
| Campaign | `qwen3-30b-p5-h100-20260622` (17/18 cells valid; `deepep-uccl-ep32-mb4-on` preempted) |

## Measured result — mean iter s · MODEL TFLOP/s/GPU · tok/s

### EP16 (TP8/PP4/DP2) — all-to-all spans 2 nodes

| mb · overlap | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM | UCCL Δ | NVSHMEM Δ |
|---|---:|---:|---:|---:|---:|
| 4 · off | **8.60 · 43.9 · 122.0k** | 14.81 · 25.5 · 70.8k | 9.16 · 41.1 · 114.4k | +72.3% | +6.6% |
| 1 · off | **27.89 · 13.5 · 37.6k** | 37.23 · 10.2 · 28.2k | 28.69 · 13.1 · 36.6k | +33.5% | +2.8% |
| 4 · on  | **6.87 · 54.9 · 152.5k** | 11.21 · 33.6 · 93.5k | 7.08 · 53.2 · 148.0k | +63.1% | +3.0% |

### EP32 (TP8/PP2/DP4) — all-to-all spans 4 nodes

| mb · overlap | NCCL all-to-all | DeepEP+UCCL | DeepEP+NVSHMEM | UCCL Δ | NVSHMEM Δ |
|---|---:|---:|---:|---:|---:|
| 4 · off | **9.24 · 40.8 · 113.5k** | 26.51 · 14.5 · 39.6k | 17.21 · 22.8 · 60.9k | +186.8% | +86.2% |
| 1 · off | **27.14 · 13.9 · 38.6k** | 48.22 · 7.8 · 21.7k | 34.66 · 11.0 · 30.3k | +77.7% | +27.7% |
| 4 · on  | **6.29 · 60.0 · 166.8k** | _preempted_ | 11.37 · 36.6 · 92.2k | — | +80.8% |

## What the numbers say

1. **On H100 with Qwen3-30B, NCCL all-to-all is the fastest dispatcher in every cell** — by
   large, unambiguous margins. DeepEP+NVSHMEM is the better of the two DeepEP backends; UCCL is
   consistently the slowest.
2. **The DeepEP penalty grows sharply with EP degree.** At **EP16** DeepEP+NVSHMEM is within
   ~3–7% of NCCL (and UCCL +33–72%); at **EP32** it blows out to **+28% to +86%** for NVSHMEM and
   **+78% to +187%** for UCCL. Doubling the expert-parallel fan-out (2→4 nodes) is where the
   DeepEP/EFA path falls apart on this fabric.
3. **This is the small-message, comm-bound regime.** Qwen3-30B (hidden 2048) makes each
   all-to-all message small and the step compute-light (NCCL tops out ~44–60 MODEL TFLOP/s/GPU),
   so DeepEP's per-dispatch proxy/kernel-launch overhead dominates and the % deltas are
   *amplified* relative to a compute-heavy model. Treat these as the **worst case for DeepEP**
   (small messages on H100), complementary to the B300/Qwen3-235B numbers (larger messages,
   compute-heavier) where the gaps are smaller.

## Cross-hardware picture (with [`../../qwen3-235b`](../../qwen3-235b/benchmarks/RESULTS.md))

| | B300 / Qwen3-235B (hidden 4096, EP32 PP2) | H100 / Qwen3-30B (hidden 2048, EP32 PP2) |
|---|---|---|
| NCCL all-to-all | fastest | fastest |
| DeepEP+UCCL (mb4) | +21.5% | +186.8% |
| DeepEP+NVSHMEM (mb4) | +34.0% | +86.2% |

Same qualitative conclusion on both fabrics — **NCCL all-to-all wins** — but the DeepEP penalty
is far larger on H100/30B (smaller messages, lower-bandwidth fabric, compute-light model). No
EFA *training* configuration tested here favors the DeepEP/UCCL path over NCCL all-to-all.

## Validity & caveats

- **EFA active on all 64 ranks** for every counted run (`efa_ok`), 0 stalls, 8 steady iters;
  per-arm transport confirmed (`uccl_ok` / `nvshmem_ok`).
- **`deepep-uccl-ep32-mb4-overlap=on` preempted** (a node was lost mid-run, rendezvous < 8/8) —
  the only missing cell. NVSHMEM arms exit non-zero at NVSHMEM finalize after training; gate on
  `efa_ok` + `n_steady==8`, not exit code.
- **Comm-bound amplification** (point 3): the % deltas are larger than a compute-heavy model
  would show. They measure the dispatcher's *exposed* cost, not end-to-end MFU on a tuned LLM.
- Mock data + random-init + forced balancing; single run per cell (within-run σ small).

## Reproduce

See [`../README.md`](../README.md). `CTX=… NS=… MODEL=qwen3-30b NNODES=8 EPS="32 16"
CELLS="4:off 1:off 4:on" RECOMPUTE="" INSTANCE_TYPE=p5.48xlarge EFA_PER_NODE=32
UCCL_IMG=…-sm90 NVSHMEM_IMG=…-sm90 bash ../qwen3-235b/benchmarks/run-qwen3-campaign.sh`.
