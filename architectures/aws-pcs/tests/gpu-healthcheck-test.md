# Test: GPU Cluster Health Check

Validates GPU hardware, EFA, and NVLink on a PCS GPU node group using the
repository's [GPU Cluster Health Check Suite](../../../4.validation_and_observability/2.gpu-cluster-healthcheck).

**This page documents only the PCS-specific deltas.** For the suite itself — what
each check does, the severity classification (PASS / MONITOR / REBOOT / ISOLATE),
the lightweight vs intensive check levels, instance profiles, and the per-check
reference — see the suite's own
[README](../../../4.validation_and_observability/2.gpu-cluster-healthcheck/README.md).
Don't duplicate that here.

**Prerequisites:** a deployed GPU CNG (P5/P5e/P5en/P6-B200/P6-B300) and the suite
available on shared `/fsx` (so every compute node sees the same scripts).

---

## PCS-specific deltas

1. **Stage the suite on `/fsx`** (shared) so all GPU nodes run the same copy:
   ```bash
   cd /fsx && git clone https://github.com/awslabs/awsome-distributed-ai.git --depth 1
   HC=/fsx/awsome-distributed-ai/4.validation_and_observability/2.gpu-cluster-healthcheck
   ```

2. **Drive it through the PCS Slurm queue** — use `srun`/`sbatch` against your GPU
   partition name (e.g. `gpu-p5`, `gpu-b200`), not the suite's bare host invocation:
   ```bash
   export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH
   # lightweight suite (checks 0-3) on one GPU node
   srun -p <gpu-partition> -N1 -n1 --exclusive bash $HC/gpu-healthcheck.sh --suite lightweight
   # multi-node NCCL (check 5) — runs inside the Slurm allocation
   srun -p <gpu-partition> -N2 -n2 --exclusive bash $HC/gpu-healthcheck.sh --check 5
   ```
   The suite's `slurm/` directory also ships `sbatch` wrappers and a
   `prolog-gpu-healthcheck.sh` you can wire into PCS as a Slurm prolog (see the
   suite README's Slurm section) — no PCS-specific change needed there.

3. **When to run on a PCS cluster:** after any GPU CNG deploy or instance
   replacement (lightweight), and before long multi-day training runs
   (lightweight + check 5). Check 5 overlaps with [Test 6 NCCL](./compute-test.md)
   but adds the suite's per-instance bandwidth thresholds.

---

## Verified on real hardware

Lightweight suite (checks 0-3) on **p6-b200** (ap-south-1, via `srun -p gpu`):
**4/4 PASS** —

| Check | Result |
|---|---|
| 0 nvidia-smi | PASS — 8 GPUs, no Xid/SXid, no ECC/retired-page errors |
| 1 DCGM L2 | PASS — all Level-2 diagnostics |
| 2 EFA enumeration | PASS — 8 EFA PCI / 10 RDMA devices (`fi_info` warning is non-blocking: libfabric lives in the container, not on the host) |
| 3 Topology | PASS — 8 GPUs, connectivity validated (a B200 "unsupported P2P path" warning is expected and non-blocking) |

EFA device count matching the instance profile and the multi-node NCCL bandwidth
threshold are also covered by [compute-test.md](./compute-test.md) (NCCL
all_reduce hit 377 GB/s on 4× p6-b200).
