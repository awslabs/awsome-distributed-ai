# Transformer Engine GEMM — FP4 / FP8 throughput on GB200 (Blackwell)

A precision/throughput microbenchmark for **GB200 / P6e-GB200** that measures NVFP4 and MXFP8 GEMM throughput on Blackwell Tensor Cores and reports it **against the dense roofline**. It validates the FLOPS claims the GB200 training and inference samples rely on, and it pins the precision toolchain (TransformerEngine / CUDA / cuBLAS / CUTLASS) those samples reuse.

Single-node: runs on the 4 GB200 GPUs of one `p6e-gb200.36xlarge`. The container is **arm64 (Grace)** even though the GEMM itself is GPU-only — GB200 hosts are ARM, not x86 (that is the B200/B300 HGX distinction).

## The one number people get wrong

Report measured throughput against the **dense** FP4 peak (~10 PFLOPS dense FP4 per GB200-binned B200), **never** the sparse 18–20 PFLOPS marketing figure. A realistic large-GEMM result is roughly **1.46–1.66× FP4-over-MXFP8**, well below the 2× peak ratio. A result near the sparse number means you are reading the wrong roofline; a result far *below* MXFP8 means the kernel is not actually hitting FP4 Tensor Cores (silent upcast) — the same failure mode the FP4 inference sample guards against.

## What's here

| File | Purpose |
|---|---|
| `transformer-engine-gemm.Dockerfile` | arm64 build: TransformerEngine ≥ 2.16, CUDA 12.8 (sm_100) / 12.9 (sm_103), cuBLAS, CUTLASS sm_100a |
| `run_gemm_bench.sh` | wraps TE's `benchmark_gemm.py` over a shape sweep; emits measured TFLOPS per recipe |
| `roofline.py` | compares measured dense TFLOPS to the dense FP4/FP8/BF16 roofline; flags "FP4 not engaged" |
| `slurm/te-gemm.sbatch` | single-node (4-GPU) Slurm run |
| `kubernetes/te-gemm.yaml` | single-node Job (EKS / HyperPod-EKS) |
| `buildspec.yaml` | CodeBuild image build |

## Running it

```bash
# Slurm (one p6e-gb200.36xlarge)
sbatch slurm/te-gemm.sbatch
# Kubernetes
kubectl apply -f kubernetes/te-gemm.yaml
# Directly, inside the container:
./run_gemm_bench.sh                         # sweeps BF16 / FP8 (delayed,current,block) / MXFP8 / NVFP4
python3 roofline.py results.json            # dense-roofline comparison + FP4-engaged check
```

TE exposes the recipes as `Float8BlockScaling`, `MXFP8BlockScaling`, and `NVFP4BlockScaling`; `run_gemm_bench.sh` drives each via `benchmark_gemm.py --recipe ...` and records `tflops = 2*M*N*K / time`.

## Testability

**Runnable** on one `p6e-gb200.36xlarge` (4× GB200, `sm_100`). No multi-node, no EFA — this is a hardware-capability benchmark, so the numbers are directly measurable today.

## Version pins (2026-06)

TransformerEngine 2.16+ (NVFP4/MXFP8 grouped-GEMM matured) · CUDA 12.8 (first Blackwell toolkit, first cuBLASLt micro-scaled FP4/FP8) for `sm_100`, CUDA 12.9 for `sm_103` · cuBLAS 12.8+ · CUTLASS 3.8+ built `-arch=sm_100a` · arm64 (Grace).
