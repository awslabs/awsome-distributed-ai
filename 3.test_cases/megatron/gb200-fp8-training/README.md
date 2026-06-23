# FP8 / FP4 Training on GB200 (Megatron-LM + Transformer Engine)

FP8 (and FP4-eval) pretraining on **P6e-GB200** UltraServers with Megatron-LM and Transformer Engine. A single `PRECISION` knob selects the recipe; parallelism defaults are chosen so the bandwidth-heavy collectives stay inside the 72-GPU NVLink domain and only DP/PP cross EFA.

> **GB200, not B300.** This is the Grace-Blackwell / NVL72 / aarch64 path. The repo's `3.test_cases/megatron/megatron-bridge` EP-backend work targets B300 HGX (x86, 8-GPU islands) — keep the two distinct: this sample assumes 4 GPUs per instance and one 72-GPU NVLink domain across 18 instances.

## Precision knob

`PRECISION` maps to Megatron + TE flags:

| `PRECISION` | Megatron flags | Notes |
|---|---|---|
| `bf16` | `--bf16` | baseline |
| `fp8-mxfp8` | `--fp8-format hybrid --fp8-recipe mxfp8` | **default on B200** (microscaled FP8) |
| `fp8-tensorwise` | `--fp8-format hybrid --fp8-recipe tensorwise` | classic per-tensor FP8 |
| `fp8-delayed` | `--fp8-format hybrid --fp8-recipe delayed` | delayed scaling |
| `fp4-nvfp4` | `--fp4 nvfp4` | **evaluation only** — FP4 *training* is still maturing; use for throughput/eval, not converged runs |

FP8 "comes for free" via TE mixed precision — no model changes. Treat `fp4-nvfp4` as a throughput/accuracy probe, not a production training recipe.

## Parallelism on one UltraServer (`u-p6e-gb200x72`)

Keep **TP** (and **EP** for MoE) ≤ the 72-GPU NVLink domain so their collectives ride NVLink/NVSwitch (NVLS). Place **DP** and **PP** across EFA *between* UltraServers. The launcher sources the canonical EFA NCCL env block from `micro-benchmarks/nccl-tests/gb200-env.sh` and relies on IMEX for the intra-UltraServer domain.

## What's here

| File | Purpose |
|---|---|
| `gb200-fp8.Dockerfile` | arm64 NGC PyTorch ≥ 25.04 base (TE ≥ 2.16, CUDA 12.8+ for sm_100); Megatron-LM pinned to a commit |
| `train.sh` | precision-knob launcher (sources gb200-env.sh, sets Megatron args) |
| `models.md` | model-size → TP/PP/DP/EP table for one and two UltraServers |
| `slurm/train.sbatch` | sbatch + enroot/pyxis |
| `kubernetes/train-mpijob.yaml` | EKS / HyperPod-EKS MPIJob with ComputeDomain + EFA |

## Run

```bash
# Slurm
PRECISION=fp8-mxfp8 MODEL=llama3-8b sbatch slurm/train.sbatch
# Kubernetes
kubectl apply -f kubernetes/train-mpijob.yaml      # edit PRECISION/MODEL env in the manifest
```

## Testability

**Authored-to-spec** for converged runs (needs a Capacity Block). A single-UltraServer short run (a few hundred steps, small model) is **runnable** to confirm the toolchain, the precision knob, and intra-domain NVLS. Always run the `4.validation_and_observability/6.gb200-health-gate` NCCL correctness gate before a real run.

## Version pins

NGC PyTorch ≥ 25.04 (arm64; pin a tested tag e.g. 25.04 / 26.01) · TransformerEngine ≥ 2.16 (NVFP4) · Megatron-LM pinned commit + Megatron-Core 0.17.x · CUDA 12.8+ / cuDNN 9.3+ (sm_100) · NCCL 2.30.4 · EFA installer 1.48.0.
