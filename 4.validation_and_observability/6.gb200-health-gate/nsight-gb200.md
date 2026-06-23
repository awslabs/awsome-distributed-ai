# Nsight Systems profiling on P6e-GB200 (Grace / aarch64)

Nsight Systems 2025.x ships an **aarch64** build — use it; the x86 build will not run on Grace. Profile a representative step to confirm communication/compute overlap and that NVLS is being used intra-domain.

## Capture

```bash
# Inside the GB200 container (one rank traces; others run normally).
nsys profile \
  --trace=cuda,nvtx,osrt,nccl \
  --gpu-metrics-devices=all \
  --output=gb200_step_%q{SLURM_PROCID} \
  --force-overwrite=true \
  <your training/inference launch command>
```

`--trace=nccl` annotates collective calls so you can confirm, on the timeline, that intra-domain collectives use NVLS (NVLink SHARP) and that cross-UltraServer traffic is the only thing hitting EFA.

## What to look for

- **NVLS on the NCCL track** for intra-domain allreduce (in-fabric reduction; minimal GPU-side reduce kernels).
- **Communication hidden under compute** — the GB200 NVLink domain has the bandwidth to overlap; exposed comm gaps mean a tuning problem, not a hardware limit.
- **C2C traffic** when using Grace offload (KV-offload / unified memory samples) — confirm prefetch is hiding page-fault latency.

## GPU metrics

`--gpu-metrics-devices=all` samples SM occupancy, Tensor Core activity, and memory throughput. On Blackwell, confirm Tensor Core activity matches the precision you intend (FP4/FP8) — flat Tensor Core utilization during a supposedly-FP4 matmul means a silent upcast (cross-check with the `transformer-engine-gemm` roofline tool).

## Version pins

Nsight Systems 2025.x (aarch64/Grace build). Pull the matching CLI from the NGC container or the standalone aarch64 package.
