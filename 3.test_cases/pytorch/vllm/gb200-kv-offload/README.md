# KV-Cache Offload to Grace Host Memory on GB200

Extend usable KV-cache capacity on **P6e-GB200** by offloading cold/older KV blocks from GPU HBM to **Grace host memory (LPDDR5X)**, keeping active KV in HBM. Single node (4 GB200 GPUs + Grace), arm64.

> **Honest framing (the research corrected a common claim):** this is **generic host-DRAM offload that happens to land in Grace LPDDR5X** — it is *not* a special coherent-tiering feature. Grace memory is ~480 GB at ~500 GB/s, roughly **1/16 of HBM bandwidth**, so it is a **lower tier for cold/old tokens**, not a place to keep hot KV. Active decode KV stays in HBM; offload buys capacity (longer contexts, more concurrent sequences), not speed.

## Scope — one sample, deliberately

- **KV-offload-to-Grace**: vLLM CPU/host KV offload (optionally via LMCache / NIXL tiering) targeting Grace memory.
- **No CPU-hosted speculative decoding.** There is **no framework support** for running a speculative-decode draft model on the Grace CPU today, so this sample does **not** build one. If you want speculative decoding, use **GPU-resident EAGLE-3** as a separate, optional vLLM knob (documented below) — not a Grace-CPU draft.

## What's here

| File | Purpose |
|---|---|
| `gb200-kv-offload.Dockerfile` | arm64: vLLM 0.11–0.14 with KV offload, optional LMCache 0.4.7 / NIXL 1.3.0 |
| `serve.sh` | vLLM serve with host-KV offload sized to Grace memory |
| `kubernetes/serve-job.yaml` | EKS / HyperPod-EKS (vLLM family is k8s-first) |
| `slurm/serve.sbatch` | Slurm variant (no precedent in the vllm family — deliberate addition) |

## Run

```bash
# Kubernetes (primary)
kubectl apply -f kubernetes/serve-job.yaml
# Slurm
sbatch slurm/serve.sbatch
# Optional GPU-resident speculative decoding (NOT Grace-CPU): add to serve.sh
#   --speculative-config '{"method":"eagle3","model":"<eagle3-head>","num_speculative_tokens":3}'
```

## Tuning intuition

- Set `--cpu-offload-gb` (or LMCache's host pool) to a fraction of the ~480 GB Grace pool — leave headroom for the OS and the training/rollout co-tenants.
- Offload helps **capacity-bound** workloads (long context, high concurrency, prefix reuse). For latency-bound decode of short contexts it adds fetch latency — measure both.

## Testability

**Runnable** on one `p6e-gb200.36xlarge`. Report the capacity gain (max concurrent sequences / context length at fixed HBM) and the decode-latency delta when KV is served from Grace vs HBM.

## Version pins

vLLM 0.11.0–0.14.0 (host KV offload) · LMCache 0.4.7 · NIXL 1.3.0 · NVIDIA Dynamo (KVBM) 1.2.1 · arm64 (Grace).
