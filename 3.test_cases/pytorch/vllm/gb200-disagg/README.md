# Disaggregated Prefill-Decode on GB200 (Dynamo / vLLM + NIXL over EFA)

Disaggregated LLM serving on **P6e-GB200**: separate prefill and decode worker pools with KV-cache handoff via **NIXL**. The GB200/AWS specifics are the whole reason this is a distinct sample from the repo's existing `dsv3-uccl-nixl` (which is x86 / H200 / p5en, UCCL-EP).

> **GB200, not B300/p5en.** Grace/NVL72/aarch64. KV transfer is NVLink/NVSwitch intra-UltraServer and **GPUDirect RDMA over EFA** (not InfiniBand) cross-UltraServer.

## The AWS KV-transport reality

- NIXL supports **EFA since v1.0.0**, but the path must be **explicitly pinned**: `backends: ["LIBFABRIC"]`. UCX is NIXL's off-AWS default and will not use EFA correctly here.
- **Open blocker — NIXL #1609:** on exactly p6e-gb200, the LIBFABRIC+VRAM dma-buf path fails with "Bad address" (unresolved in NIXL 1.0.0 / 1.1.0). **Default `FI_HMEM_CUDA_USE_DMABUF=0`** to work around it, and document the resulting perf caveat. Track the issue for the fix.
- Intra-UltraServer KV handoff rides NVLink/NVSwitch (no EFA); only cross-UltraServer handoff hits EFA.

## What's here

| File | Purpose |
|---|---|
| `gb200-disagg.Dockerfile` | arm64: vLLM 0.21.0 Blackwell wheel (FA4 default sm_100/sm_103), NIXL built `-Dlibfabric_path=/opt/amazon/efa`, EFA 1.48.0 |
| `nixl-config.yaml` | KV-connector config: `backends: ["LIBFABRIC"]`, `FI_HMEM_CUDA_USE_DMABUF=0` |
| `kubernetes/` | prefill / decode / proxy Deployments (EKS) |
| `slurm/disagg.sbatch` | heterogeneous prefill + decode nodes; NIXL via DECODE_IP/PREFILL_IP/KV_TRANSFER_PORT |
| `topology.md` | co-resident vs split prefill/decode layout across the NVL36x2 domain |

## Run

```bash
# Kubernetes (EKS) -- prefill, decode, proxy
kubectl apply -f kubernetes/
# Slurm -- heterogeneous job
sbatch slurm/disagg.sbatch
```

## Testability

Single-UltraServer (co-resident or intra-domain split) is **runnable**. Cross-UltraServer KV over EFA is **authored-to-spec** and currently gated on NIXL #1609 (ship with `FI_HMEM_CUDA_USE_DMABUF=0`).

## Version pins

NVIDIA Dynamo 1.2.1 · vLLM 0.21.0 (NixlConnector; FA4 default sm_100/sm_103 since 0.20.0) · NIXL 1.1.0 (EFA floor 1.0.0) · aws-efa-installer 1.48.0 (NIXL+EFA GA floor 1.47.0) · CUDA ≥ 12.8 (GB200 repro on 12.9) · arm64, u-p6e-gb200x72.
