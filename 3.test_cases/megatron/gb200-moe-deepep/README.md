# MoE / Expert Parallelism on GB200 (DeepEP over NVL72)

Mixture-of-Experts training on **P6e-GB200** with expert parallelism, plus an all-to-all dispatch/combine microbenchmark. The point of this sample is the AWS-specific transport reality for expert parallelism, which differs sharply from the NVIDIA reference.

> **GB200, not B300.** 4 GPUs per instance, one 72-GPU NVLink domain across 18 instances, aarch64, IMEX. The repo's `megatron-bridge` EP-backend-comparison work targets B300 HGX (x86, 8-GPU islands) — this is the complementary Grace/NVL72 port.

## The transport reality (verified)

- **DeepEP's IBGDA / GPU-initiated path does NOT work on EFA** — EFA has no GPUDirect Async. So the IB low-latency kernels DeepEP relies on off-AWS are unavailable.
- **DeepEP v1 works on EFA via NVSHMEM's libfabric transport** (CPU-proxy) — this is the path that runs today on AWS.
- **EPv2 NCCL-GIN is the forward EFA path** — NCCL ≥ 2.29.3 GPU-Initiated Networking + aws-ofi-nccl GIN; this is where cross-node EP is heading on AWS.
- **Mainline DeepEP assumes 8-GPU NVLink islands** (`NUM_MAX_NVL_PEERS == 8`) and does **not** fit GB200's 4-GPU-per-instance / 72-GPU-domain layout in standard internode mode. The GB200 path is the DeepEP **`hybrid-ep`** branch (and Megatron-Core's MNNVL-native **`hybridep`** dispatcher backend).

**The rule:** keep **EP ≤ 72** so expert all-to-all stays inside the NVLink domain (NVLS, no EFA hop, SHARP irrelevant there). Scale beyond one UltraServer with **PP/DP** over EFA — cross-UltraServer EP rides EFA with no in-network SHARP and is the expensive path.

## What's here

| File | Purpose |
|---|---|
| `gb200-moe.Dockerfile` | arm64: NVSHMEM ≥ 3.7, NCCL ≥ 2.29.3 (GIN), DeepEP `hybrid-ep`, aws-ofi-nccl GIN, CUDA 13 |
| `ep-bench.sh` | dispatch/combine bandwidth: intra (1 UltraServer, NVLink) vs cross (2 UltraServers, EFA) — measures the crossover |
| `train-moe.sh` | short Megatron-Core MoE run, `flex` dispatcher with `deepep` vs `hybridep` backend |
| `slurm/`, `kubernetes/` | run on each orchestrator |

## Run

```bash
# EP microbenchmark (intra vs cross domain)
SCOPE=intra sbatch slurm/ep-bench.sbatch      # 1 UltraServer
SCOPE=cross sbatch slurm/ep-bench.sbatch      # 2 UltraServers (EFA all-to-all)
# MoE training (compare dispatcher backends)
BACKEND=hybridep sbatch slurm/train-moe.sbatch
```

The microbenchmark asserts, via `NCCL_DEBUG=INFO` / NVSHMEM logs, that intra-domain EP issues **no EFA traffic** while cross-UltraServer EP does — that is the load-bearing claim made measurable.

## Testability

The microbenchmark is **runnable**: one UltraServer for the intra leg, two for the cross leg. The short MoE training run is runnable on one UltraServer for a small (DeepSeek-V3-shaped / Mixtral-shaped) config. Single-node `sm_100` compile + intranode dispatch/combine is CI-checkable without a full domain.

## Version pins

DeepEP `hybrid-ep` branch · NVSHMEM ≥ 3.7.0 · NCCL ≥ 2.29.3-1 (GIN Device API) · aws-ofi-nccl GIN build · GDRCopy 2.5.2 · CUDA 13 (cu130) · Megatron-Core MoE `flex` dispatcher (`deepep` / `hybridep` backends) · arm64, sm_100 (GB200) / sm_103 (GB300).
