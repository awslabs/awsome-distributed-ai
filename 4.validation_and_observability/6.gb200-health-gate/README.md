# GB200 Health Gate & Observability

Pre-run validation and observability for **P6e-GB200** NVL72 domains. The headline is a **mandatory NCCL correctness gate**: on NVL36x2, fabric-level faults (a miscabled NVSwitch port pair, a degraded NVLS multicast bind) can pass single-asset diagnostics yet silently corrupt cross-asset-group collectives. A bandwidth number will not catch this — only an allreduce **correctness** check across the whole 72-GPU domain will.

> GB200-specific: 72-GPU NVLink domain, NVLink-C2C, NVSwitch fabric, IMEX. Not the B200/B300 HGX layout.

## What's here

| File | Purpose |
|---|---|
| `nccl-correctness-gate.sh` | Runs `all_reduce_perf -c 1` across the domain, parses the `#wrong` column, **FAILS on any nonzero**. busbw stays advisory (no fabricated GB200 threshold). |
| `topology-clique.sh` | Derives the NVLink domain from EC2 `DescribeInstanceTopology` (`capacityBlockId`) and the `nvidia.com/gpu.clique` label — not `nvidia-smi topo -m`. |
| `gb200-dcgm-metrics.csv` | DCGM field list extended for GB200: NVSwitch, NVLink-C2C, and per-link NVLink counters. |
| `grafana-gb200-panels.md` | Panel definitions; C2C link-status is the early-warning signal. |
| `nsight-gb200.md` | Nsight Systems (aarch64 / Grace) profiling recipe for GB200. |
| `slurm/health-gate.sbatch`, `kubernetes/health-gate-job.yaml` | Run the gate on each orchestrator. |

## The correctness gate (mandatory before production)

```bash
# Slurm
sbatch slurm/health-gate.sbatch            # 18 nodes / 72 GPUs; exits nonzero if any #wrong != 0
# Kubernetes
kubectl apply -f kubernetes/health-gate-job.yaml
```

`nccl-correctness-gate.sh` is the load-bearing piece: it parses the last field (`#wrong`) of every nccl-tests data row and fails the job if any is nonzero. Wire it as a required step in your launcher (training/inference) so a fabric-corrupting node can never silently poison a run. This is the L11-style check that single-asset diagnostics (which validate only intra-asset-group NVLink paths) miss.

## Observability

DCGM field IDs added for GB200 in `gb200-dcgm-metrics.csv`:

- **NVLink**: per-link bandwidth/error counters (1011/1012, 1040–1075)
- **NVSwitch**: 780–783
- **NVLink-C2C**: 285–287, 1076–1079 — C2C link status is the **earliest** warning that a Grace↔Blackwell link is degrading.

Requires **DCGM 4.6.0** (C2C fields available since 4.2.3; NVLink-5 coverage by 4.6.0). Feed into the existing Prometheus/Grafana stack (`4.validation_and_observability/4.prometheus-grafana`); panel definitions in `grafana-gb200-panels.md`.

## Testability

The correctness gate and DCGM scrape are **runnable** on one `u-p6e-gb200x72` Capacity Block. Cross-domain (144-GPU) correctness needs two UltraServers.

## Version pins

DCGM 4.6.0 · nccl-tests `main` (the `-c`/`#wrong` interface) · EC2 `DescribeInstanceTopology` · Nsight Systems 2025.x (aarch64/Grace) · EKS K8s 1.33+ / AMI v20251103 (driver 580, CDMM).
