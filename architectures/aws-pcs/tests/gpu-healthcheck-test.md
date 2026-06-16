# Test: GPU Cluster Health Check

Validates GPU hardware, EFA networking, and NVLink interconnects using the
[GPU Cluster Health Check Suite](../../4.validation_and_observability/2.gpu-cluster-healthcheck)
from this repository.

**Prerequisites**:
- Cluster with a GPU CNG deployed (P5, P5e, P5en, or P6-B200)
- At least 2 GPU nodes for multi-node NCCL check
- The health check scripts fetched to `/fsx` (shared storage)

---

## Setup

```bash
# Fetch the health check suite to shared /fsx
cd /fsx
git clone https://github.com/awslabs/awsome-distributed-ai.git --depth 1 --sparse
cd awsome-distributed-ai
git sparse-checkout set 4.validation_and_observability/2.gpu-cluster-healthcheck
cd 4.validation_and_observability/2.gpu-cluster-healthcheck
```

Or simply copy the directory from the repo checkout if already available.

---

## Part A — Lightweight suite (single-node, ~15 min)

Run on each GPU node to validate hardware is healthy before submitting
training jobs:

```bash
export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH

# Single node
srun -N 1 -n 1 -p gpu-p5 --exclusive \
  bash /fsx/awsome-distributed-ai/4.validation_and_observability/2.gpu-cluster-healthcheck/gpu-healthcheck.sh \
    --suite lightweight
```

Or via sbatch for multiple nodes:
```bash
sbatch -N 2 -p gpu-p5 --exclusive \
  /fsx/awsome-distributed-ai/4.validation_and_observability/2.gpu-cluster-healthcheck/slurm/sbatch-lightweight.sh
```

### Expected results (per node)

| Check | What it validates | Pass criteria |
|---|---|---|
| 0: nvidia-smi | GPU count, driver, Xid errors | All GPUs detected, no Xid errors in dmesg |
| 1: DCGM L2 | PCIe BW, memory stress, SM stress | All DCGM sub-tests PASS |
| 2: EFA enumeration | EFA device count, RDMA, libfabric | EFA count matches instance profile (32 for p5, 16 for p5en, 8 for p6-b200) |
| 3: Topology | NVLink status, PCIe groups | All NVLinks UP, no uncorrectable errors |

### Severity output

Results are classified as:
- **PASS** — all checks green
- **MONITOR** — minor anomaly, non-blocking
- **REBOOT** — correctable with instance reboot
- **ISOLATE** — hardware fault, instance should be replaced

---

## Part B — NCCL multi-node (check 5, ~10-20 min)

Validates multi-node GPU communication over EFA. Overlaps with Test 6 (NCCL)
but adds per-instance bandwidth threshold validation:

```bash
srun -N 2 -n 2 -p gpu-p5 --exclusive \
  bash /fsx/awsome-distributed-ai/4.validation_and_observability/2.gpu-cluster-healthcheck/gpu-healthcheck.sh \
    --check 5
```

### Expected

- EFA provider selected (`Selected Provider is efa`)
- Bus bandwidth meets instance-type threshold (e.g. p5: >700 GB/s peak for 2-node all_reduce at 128MB)
- No NCCL timeouts or EFA transport errors

---

## Part C — Slurm prolog integration (production use)

For production clusters, configure the lightweight checks as a Slurm prolog
so every GPU job validates hardware before running:

```bash
# In slurm.conf (via PCS custom settings):
Prolog=/fsx/awsome-distributed-ai/4.validation_and_observability/2.gpu-cluster-healthcheck/slurm/prolog-gpu-healthcheck.sh
```

A non-zero prolog exit drains the node and requeues the job.

---

## When to run

| Scenario | Suite | Frequency |
|---|---|---|
| After initial GPU CNG deployment | Lightweight (full) | Once |
| Before multi-day training runs | Lightweight + check 5 | Per-run |
| Production steady-state | Prolog (checks 0, 2) | Every job |
| Node suspected faulty | Intensive (all checks) | On-demand |
| After instance replacement | Lightweight | Once |

---

## Verdict checklist

| Check | Result |
|---|---|
| Lightweight suite completes on all GPU nodes | ✅ |
| No ISOLATE/REBOOT severity results | ✅ |
| EFA device count matches instance profile | ✅ |
| NVLink all UP, no uncorrectable errors | ✅ |
| Multi-node NCCL bandwidth meets threshold | ✅ |

---

## Reference

- [GPU Cluster Health Check Suite README](../../4.validation_and_observability/2.gpu-cluster-healthcheck/README.md)
- Supported instance profiles: see `instance-profiles.conf` in the suite
