# Qwen3-8B Pre-Training: H200 vs B300 (NeMo/Megatron)

Pre-training **Qwen3-8B** (8.2B dense parameters) on 1T tokens comparing two GPU generations — p5en.48xlarge (H200) and p6-b300.48xlarge (B300) — using NeMo/Megatron on 2-node / 16-GPU topologies with EFA GDRDMA interconnect.

## Results

| Metric | H200 (p5en) | B300 (p6-b300) | Ratio |
|--------|-------------|----------------|-------|
| **TFLOP/s per GPU** | 497 | **976** | 1.96× |
| **Throughput** | 162K tok/s | **318K tok/s** | 1.96× |
| **Time to 1T tokens** | ~71 days | **~36 days** | 1.97× |
| Step time (100 iters) | 3.23s | 1.65s | 1.96× |
| Peak memory/GPU | ~138 GB / 141 GB | ~173 GB / 288 GB | — |
| MFU | 0.50 | 0.50 | — |

Both clusters are compute-saturated with perfect communication overlap. AllReduce and AllGather are fully hidden behind compute.

## Prerequisites

- **Slurm** workload manager with **PyXis + Enroot** container runtime
- **EFA networking** with GDRDMA support (for multi-node communication)
- **FSx for Lustre** shared filesystem mounted at `/fsx/`
- **Docker** (for building container images)

> **Don't have a cluster?** Deploy a fully functional HPC cluster in under 1 hour using [Amazon SageMaker HyperPod](https://awslabs.github.io/ai-on-sagemaker-hyperpod/). The guide walks you through deploying a ready-to-use cluster with Slurm, EFA, PyXis/Enroot, and FSx for Lustre pre-configured.

## Quick Start

> **Disk space:** The container build requires ~50 GB of disk space in TMPDIR.
> `enroot import` needs `sudo` and TMPDIR pointing to FSx (not `/tmp`, which is too small).

### H200 Cluster (p5en.48xlarge)

```bash
# 1. Build container
cd h200/ && docker build -t qwen3-8b-h200:latest .
sudo TMPDIR=/fsx/tmp ENROOT_TEMP_PATH=/fsx/tmp enroot import --output /fsx/ubuntu/qwen3-8b-pretraining/containers/nemo-efa-25.07.sqsh dockerd://qwen3-8b-h200:latest

# 2. Submit training job
sbatch h200/slurm/run.sh
```

### B300 Cluster (p6-b300.48xlarge)

```bash
# 1. Build container
cd b300/ && docker build -t qwen3-8b-b300:latest .
sudo TMPDIR=/fsx/tmp ENROOT_TEMP_PATH=/fsx/tmp enroot import --output /fsx/ubuntu/qwen3-8b-pretraining/containers/nemo-efa-26.02.sqsh dockerd://qwen3-8b-b300:latest

# 2. Submit training job
sbatch b300/slurm/run.sh
```

## Model Architecture: Qwen3-8B

| Parameter | Value |
|-----------|-------|
| Layers | 36 |
| Hidden dim (d_model) | 4096 |
| Q-heads | 32 |
| KV-heads | 8 (GQA) |
| FFN dim | 14336 (SwiGLU) |
| Vocab size | 151,936 |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Sequence length | 4096 |
| Precision | BF16 |
| Total params | 8.2B |

## Parallelism Strategy

**Pure Data Parallelism (DP=16)** — the model fits entirely on a single GPU.

| Component | Setting |
|-----------|---------|
| Tensor Parallel | 1 |
| Pipeline Parallel | 1 |
| Data Parallel | 16 |
| Distributed Optimizer | Yes (shards Adam states across DP ranks) |
| Overlap Grad Reduce | Yes |
| Overlap Param Gather | Yes |

**Why TP=1 is optimal:** At 8.2B params, the model + optimizer states fit on one GPU with distributed optimizer. Adding tensor parallelism introduces all-reduce communication for every transformer layer — validated experimentally: TP=2 was 11% slower (868 vs 976 TFLOP/s on B300).

## Best Configuration Per Cluster

| Parameter | H200 (p5en.48xlarge) | B300 (p6-b300.48xlarge) |
|-----------|---------------------|------------------------|
| GPUs | 16× H200 (141 GB HBM3) | 16× B300 (288 GB HBM3e) |
| Parallelism | TP=1, PP=1, DP=16 | TP=1, PP=1, DP=16 |
| Micro-batch size | 2 | 4 |
| Global batch size | 128 (grad_accum=4) | 128 (grad_accum=2) |
| Sequence length | 4096 | 4096 |
| Precision | BF16 | BF16 |
| Gradient checkpointing | Full recompute | None |
| Distributed optimizer | Yes (sharded Adam) | Yes (sharded Adam) |
| Overlap grad reduce | Yes | Yes |
| Framework | Megatron-Core (NeMo 25.07) | Megatron-Bridge (NeMo 26.02) |

## Key Findings

1. **Both clusters are compute-saturated with perfect communication overlap.** AllReduce and AllGather are fully hidden behind compute — verified by single-GPU benchmarks showing lower TFLOP/s due to reduced batch arithmetic intensity.

2. **Best software stack per GPU generation.** NeMo 25.07 (Megatron-Core v0.13.1) for H200; NeMo 26.02 (Megatron-Bridge v2.9.0) for B300. Each maximizes hardware utilization for its target architecture.

3. **Pure data parallelism is optimal** when the model fits in single-GPU memory. Distributed optimizer + overlapped grad reduce eliminate the memory penalty.

4. **Gradient checkpointing is the key differentiator:** required on H200 (141 GB) for MBS≥2, unnecessary on B300 (288 GB) for MBS≤4 — saving ~20% recompute overhead.

## Hardware

| | H200 Cluster | B300 Cluster |
|---|---|---|
| Instance | p5en.48xlarge | p6-b300.48xlarge |
| Nodes | 2 | 2 |
| GPUs per node | 8× H200 | 8× B300 |
| GPU Memory | 141 GB HBM3 | 288 GB HBM3e |
| Interconnect | EFA GDRDMA (3200 Gbps) | EFA GDRDMA (3200 Gbps) |
| Intra-node | NVLink | NVLink |

## Project Structure

```
├── README.md              ← You are here
├── h200/
│   ├── Dockerfile         ← NeMo 25.07 + EFA container
│   ├── train.py           ← Megatron-Core training script
│   └── slurm/
│       └── run.sh         ← Slurm submission script
├── b300/
│   ├── Dockerfile         ← NeMo 26.02 + EFA container
│   ├── train.py           ← Megatron-Bridge training script
│   └── slurm/
│       └── run.sh         ← Slurm submission script
└── docs/
    └── lessons-learned.md ← EFA, PyXis, and Megatron gotchas
```

## License

Apache 2.0
