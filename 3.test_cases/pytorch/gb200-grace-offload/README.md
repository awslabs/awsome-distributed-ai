# Training with Grace Unified-Memory Offload on GB200

Train models larger than HBM by offloading optimizer state / parameters to **Grace LPDDR5X** over the coherent **NVLink-C2C** link, instead of spilling to NVMe. Shipped as a **recipe variant** of the existing `deepspeed` and `FSDP` test cases, not a new framework.

> **GB200-specific.** The Grace coherent-memory offload story has no B200/B300 (HGX, x86) analog — those have no Grace CPU and no coherent CPU↔GPU link. This is Grace-Blackwell: ~480 GB LPDDR5X per superchip at ~500 GB/s over C2C (~1/16 of HBM bandwidth), aarch64.

## Honest framing

The nonuniform bandwidth is the whole design constraint: Grace memory is ~16× faster than NVMe but ~16× slower than HBM. So it is a **tier for cold/overflow state**, with explicit prefetch (`cudaMemPrefetchAsync` / `cudaMemAdvise`) to hide page-fault latency. Treat throughput targets here as **GH200-derived upper bounds** — the published SuperOffload / ZeRO-Offload numbers are from GH200; GB200's 2-GPU-per-Grace ratio means real GB200 numbers must be measured, and producing them is this sample's primary contribution.

## Two variants

| Variant | Mechanism |
|---|---|
| DeepSpeed SuperOffload (`ds_config_superoffload.json`) | `super_offload` to Grace coherent memory (DeepSpeed ≥ 0.18.0) |
| FSDP CPUOffload (`fsdp_offload.py` flags) | `CPUOffload(offload_params=True)` / FSDP2 `offload_policy` baseline |

## What's here

| File | Purpose |
|---|---|
| `gb200-grace-offload.Dockerfile` | arm64 base; DeepSpeed ≥ 0.18.0, PyTorch 2.x |
| `ds_config_superoffload.json` | DeepSpeed SuperOffload config (offload to Grace) |
| `run.sh` | launches the DeepSpeed or FSDP variant |
| `slurm/offload.sbatch`, `kubernetes/offload-job.yaml` | per-orchestrator |

## MPAM / resctrl note

Partitioning Grace memory bandwidth (MPAM via `resctrl`) needs a **privileged** container or host-level config on P6e-GB200 — straightforward on Slurm/bare-metal, explicit handling on EKS (privileged securityContext / host mount). The manifests flag where.

## Run

```bash
VARIANT=superoffload sbatch slurm/offload.sbatch     # or VARIANT=fsdp
kubectl apply -f kubernetes/offload-job.yaml
```

## Testability

**Runnable** on one `p6e-gb200.36xlarge` (single superchip is enough to exercise C2C offload). Numbers are GH200-derived until measured on GB200 — record measured GB200 throughput in the PR.

## Version pins

DeepSpeed ≥ 0.18.0 (SuperOffload, Oct 2025) · PyTorch 2.x (FSDP CPUOffload / FSDP2 offload_policy) · CUDA 12.x unified/managed memory + ATS · arm64 (Grace).
