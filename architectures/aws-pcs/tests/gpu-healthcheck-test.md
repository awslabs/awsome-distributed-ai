# Test 13: GPU Cluster Health Check

Validates GPU hardware, EFA, and NVLink on a PCS GPU node group using the
repository's [GPU Cluster Health Check Suite](../../../validation/gpu-cluster-healthcheck).

**This page documents only the PCS-specific deltas.** For the suite itself — what
each check does, the severity classification (PASS / MONITOR / REBOOT / ISOLATE),
the lightweight vs intensive check levels, instance profiles, and the per-check
reference — see the suite's own
[README](../../../validation/gpu-cluster-healthcheck/README.md).
Don't duplicate that here.

**Prerequisites:** a deployed GPU CNG (P5/P5e/P5en/P6-B200/P6-B300) and the suite
available on shared `/fsx` (so every compute node sees the same scripts).

---

## PCS-specific deltas

1. **Stage the suite on `/fsx`** (shared) so all GPU nodes run the same copy:

   ```bash
   cd /fsx && git clone https://github.com/awslabs/awsome-distributed-ai.git --depth 1
   HC=/fsx/awsome-distributed-ai/validation/gpu-cluster-healthcheck
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

### Intensive suite (checks 4-6) on p6-b200 ×2 (ap-south-1)

The intensive suite (`--suite intensive`) needs **exclusive nodes for 1-3 hr** (DCGM L4
alone is 45 min – 2.25 hr/node), so it's a maintenance-window / pre-long-run check, not a
per-deploy gate.

- **Check 6 — EFA loopback: PASS** on 2× p6-b200 (8 EFA domains tested per node, both nodes).
- **Check 4 — DCGM L4:** budget the full 45 min+/node; it disables MIG, stops concurrent
  GPU telemetry (`dcgm-exporter`), and runs the EUD + pulse-power stress, so it cannot share
  a node with other GPU work.

> **⚠️ Validate multi-node NCCL via [compute-test.md Test 6](./compute-test.md#test-6-nccl-multi-node-efa),
> not intensive check 5** — until the suite fix lands. The suite's
> `checks/5-nccl-allreduce.sh` defaults to an ECR image URI that enroot rejects (needs the
> `#` registry separator, `docker://public.ecr.aws#hpc-cloud/nccl-tests:<tag>`), and invokes
> `all_reduce_perf` by bare name when that image keeps the binaries under
> `/opt/nccl-tests/build/` (not on `PATH`). The canonical `nccl-tests-container.sbatch` in
> Test 6 handles both correctly and ran to 377 GB/s on this cluster.
>
> Measured consequence on **p5.48xlarge ×2** (Slurm 25.11, 2026-08-01): stock
> `--check 5` exits 1 after **2.7 s** and reports **severity RESET** on nodes that sustain
> **445.33 GB/s** all-reduce with `#wrong=0` immediately afterwards — i.e. a false RESET on
> healthy hardware. Both causes are fixed by
> `checks/5-nccl-allreduce.sh`'s `NCCL_CONTAINER` (`#` separator) and the new
> `NCCL_TESTS_BIN` override; once that is merged, check 5 is usable here and this warning
> can be reduced to a version note.

### Verified on p5.48xlarge (us-east-1, 2026-08-01, Slurm 25.11)

Lightweight suite on **p5.48xlarge** via `srun -p gpu`, per-check wall clock
(driver 595.71.05, DCGM 4.6.0, libfabric 2.4.0amzn1.0, 32 EFA devices):

| Check | Result | Wall clock |
|---|---|---|
| 0 nvidia-smi | PASS — 8× H100 80GB HBM3, no Xid/SXid | **3.12 s** |
| 1 DCGM L2 | PASS — all Level-2 diagnostics | **254.62 s** |
| 2 EFA enumeration | PASS — 32 EFA PCI / 32 RDMA / 32 uverbs | **2.24 s** |
| 3 Topology | PASS — 8 GPUs, NVLink validated | **3.27 s** |
| **lightweight total** | 4/4 PASS | **≈263 s** (**8.6 s** without DCGM L2) |
| 6 EFA loopback | PASS ×6 consecutive runs, 32 domains each | **138.70–139.10 s** |
| prolog (checks 0+2) | ×3 runs | **3.46 / 3.37 / 3.37 s** |

Two notes for PCS specifically:

- **The PCS-Ready DLAMI ships the EFA userspace** at `/opt/amazon/efa/bin`
  (`fi_info`, `fi_pingpong`, libfabric 2.4.0amzn1.0) but does **not** put it on `PATH`,
  so checks 2 and 6 log `fi_info not found` / `fi_pingpong not found on PATH` as a WARN.
  Check 6 then adds the directory itself and runs normally — the WARN is cosmetic, not a
  skipped test. (Contrast with EKS GPU AMIs, where `/opt/amazon/efa` is genuinely empty.)
- Under `sbatch`, `PATH` may be **unset**, so a script that does
  `export PATH=/some/dir:$PATH` ends up with only `/some/dir` and loses coreutils. Seed the
  standard directories explicitly in Slurm job scripts.
