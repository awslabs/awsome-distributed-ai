# Prefill / decode topology on the GB200 NVL36x2 domain

Two layouts, toggled by where you place the worker pools relative to the 72-GPU NVLink domain.

## Co-resident (single UltraServer) — runnable

Prefill and decode workers share one 72-GPU NVLink domain. KV handoff rides NVLink/NVSwitch — no EFA, lowest latency. Best for moderate load and for validating the pipeline before scaling.

```
[ u-p6e-gb200x72 ]  prefill pool (e.g. 36 GPU) + decode pool (e.g. 36 GPU), KV over NVLink
```

## Split (two UltraServers) — authored-to-spec, gated on NIXL #1609

Prefill on one UltraServer, decode on another. KV handoff crosses **EFA** via NIXL's LIBFABRIC backend (GPUDirect RDMA). Independent scaling of prefill vs decode pools; the cross-domain transfer is the path affected by #1609 (`FI_HMEM_CUDA_USE_DMABUF=0`).

```
[ US-A: prefill 72 GPU ]  --(NIXL / LIBFABRIC over EFA)-->  [ US-B: decode 72 GPU ]
```

## Choosing

- Latency-critical / moderate scale → co-resident (NVLink KV).
- Need to scale prefill and decode independently → split, accept the EFA KV path and its current #1609 caveat.

Set `PREFILL_IP` / `DECODE_IP` / `KV_TRANSFER_PORT` (Slurm) or the prefill/decode Deployment selectors (EKS) to match the chosen layout.
