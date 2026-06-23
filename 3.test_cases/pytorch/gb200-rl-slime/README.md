# RL Post-Training on GB200 (slime, FP8 rollout + BF16 train)

GRPO-style RL post-training on **P6e-GB200**, led by **slime** (with verl and OpenRLHF documented as alternatives). RL colocates a training engine and a rollout/inference engine — exactly where GB200's combined strengths pay off: FP8 rollout on Blackwell Tensor Cores, BF16 policy training, and fast weight/KV movement over the 72-GPU NVLink domain.

> **GB200, not B300.** Grace/NVL72/aarch64. The repo's existing veRL GRPO recipe targets non-GB200 hardware; this is the Grace/NVL72 port (aarch64 containers, EP/TP inside the 72-GPU domain).

## Why slime leads here

slime's train/rollout split maps cleanly onto the NVLink domain, and its SGLang rollout backend supports FP8 KV cache on Blackwell. The recipe:

1. **Container**: rebuild `slime` aarch64/sbsa on an NGC Blackwell base (CUDA ≥ 13, torch 2.11, SGLang ≥ 0.5.12 unified cu130 tag, TE ≥ 2.13).
2. **NCCL**: apply the Grace-Blackwell NCCL fixes; source the canonical EFA env block (`micro-benchmarks/nccl-tests/gb200-env.sh`).
3. **Smoke first**: a single-node B200/GB200 run (Qwen3-4B on DAPO-math-17k) to reproduce-or-clear the known slime #1487 hang **before** scaling to NVL72.
4. **Precision**: FP8 rollout via SGLang passthrough (`--sglang-kv-cache-dtype fp8_e4m3`); keep **training BF16** to dodge the MXFP8 power-of-two-scale training bug.
5. **Placement**: Ray head/workers pinned inside the NVLink domain; TP+EP ≤ 72.

## Alternatives (documented, not led)

- **verl** — most-validated aarch64 GB200 container path (PR #5596); the fallback if slime's hang blocks you.
- **OpenRLHF** — simple DeepSpeed comparison point; weak MoE/Megatron story.

## What's here

| File | Purpose |
|---|---|
| `gb200-slime.Dockerfile` | arm64 NGC Blackwell base; slime + SGLang ≥ 0.5.12 + TE ≥ 2.13 |
| `run-grpo.sh` | GRPO launch: BF16 train + FP8 SGLang rollout, NVLink-domain placement |
| `smoke.sh` | single-node Qwen3-4B / DAPO-math-17k smoke (clears #1487 before scale) |
| `slurm/grpo.sbatch`, `kubernetes/grpo-rayjob.yaml` | per-orchestrator |

## MoE showcase

The headline run: Qwen3-30B-A3B → DeepSeek-V3-class, large TP+EP across the 72-GPU domain, BF16 train + FP8 rollout — turning "GB200 has a big NVLink domain and FP8 tensor cores" into a measurable RL throughput number.

## Testability

Single-node smoke is **runnable** today (clears the hang, confirms the container/precision path). Full NVL72 RL is **authored-to-spec** until a Capacity Block is held. Run the health-gate NCCL correctness check first.

## Version pins

slime v0.3.0 (AWS test case pins v0.2.4) · SGLang 0.5.12(.post1) unified cu130 · CUDA ≥ 13 · torch 2.11 · TE ≥ 2.13 · NGC PyTorch 26.02 · vLLM ≥ 0.9.0 / FlashInfer ≥ 0.2.5 (sm_100) · arm64.
