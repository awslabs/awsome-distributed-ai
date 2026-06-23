# nvbandwidth on P6e-GB200 — NVLink, Grace–Blackwell C2C, and IMEX cross-instance fabric

[NVIDIA `nvbandwidth`](https://github.com/NVIDIA/nvbandwidth) measured-bandwidth microbenchmark, packaged for **GB200 / P6e-GB200 UltraServers**. It validates three fabric paths that are unique to the Grace-Blackwell NVL72 platform on AWS and have no analog on HGX B200/B300 boxes:

1. **NVLink-C2C** — the coherent Grace CPU ↔ Blackwell GPU link (~900 GB/s spec). Exercised by host↔device copies that ride C2C rather than PCIe.
2. **Intra-domain GPU↔GPU bisection** — the 72-GPU NVLink domain over NVSwitch (device-to-device copy engine / SM paths).
3. **IMEX cross-instance memory** — GPU memory access that crosses the EC2 instance boundary inside one UltraServer, which **requires NVIDIA IMEX** (a DRA ComputeDomain on EKS, or a prolog-configured IMEX service on Slurm). This path does not exist on a single-host HGX system.

## Why a separate fabric benchmark

`nccl-tests` measures collective bandwidth; it does not isolate the **C2C** link or the **cross-instance IMEX** path, and it does not surface the silent-miscabling / Xid-149 failure modes documented for NVL36x2. `nvbandwidth` measures point-to-point copy bandwidth per direction, so a per-direction floor catches a degraded link (one miscabled NVSwitch port pair, a C2C link in a warning state) that a collective average would hide.

## What's here

| File | Purpose |
|---|---|
| `nvbandwidth.Dockerfile` | arm64 (Grace) CUDA build of nvbandwidth with multi-node (`-DMULTINODE=1`) + OpenMPI |
| `slurm/nvbandwidth.sbatch` | Slurm run: IMEX preflight (`nvidia-imex-ctl -N` → `Domain State: UP`) then `-np 72` mpirun |
| `kubernetes/nvbandwidth.yaml` | DRA `ComputeDomain` + Job, clique-pinned, for EKS / HyperPod-EKS |
| `test_nvbandwidth.py` | pytest gate: asserts IMEX domain UP, parses the GB/s matrix, fails below a per-direction floor |
| `buildspec.yaml` | CodeBuild image build |

## Running it

**Slurm (one UltraServer, runnable):**
```bash
sbatch slurm/nvbandwidth.sbatch          # --nodes=18 (72 GPUs)
```
The job first runs `nvidia-imex-ctl -N` and aborts unless every node reports `Domain State: UP`, then runs the multi-node device-to-device and host-to-device (C2C) tests.

**Kubernetes (EKS / HyperPod-EKS):**
```bash
kubectl apply -f kubernetes/nvbandwidth.yaml
```

**Second exerciser (NVSHMEM):** for a put/get view of the same fabric, the `shmem_put_bw` microbenchmark from [`micro-benchmarks/nvshmem`](../nvshmem) complements the copy-engine numbers here; build it from that directory's Dockerfile and launch with the same clique pinning.

## Interpreting results

| Path | Healthy (order of magnitude) | Red flag |
|---|---|---|
| Grace↔Blackwell C2C (host↔device) | hundreds of GB/s, approaching the ~900 GB/s spec | PCIe-class (tens of GB/s) → C2C not engaged / falling back to PCIe |
| Intra-domain GPU↔GPU (device↔device) | full NVLink-5 per-GPU bandwidth | one pair markedly low → miscabled NVSwitch port pair; correlate with Xid 149 |
| Cross-instance (IMEX) | NVLink-class within the domain | failure / 0 → IMEX domain not `UP`; check the ComputeDomain / prolog |

`test_nvbandwidth.py` encodes the floor as a configurable threshold (default anchored near the observed ~821 GB/s C2C read point); it does **not** hard-code a fabricated GB200 number — set `NVBW_MIN_GBPS` to your measured baseline.

## Testability

| Path | Status |
|---|---|
| Single-UltraServer C2C / intra-domain / IMEX | **Runnable** on one `u-p6e-gb200x72` Capacity Block |
| Per-direction floors | calibrate on first run; the ~821 GB/s C2C anchor is an observation, not a spec guarantee |

## Version pins

CUDA ≥ 12.3 (multi-node floor) · NVIDIA driver R570+ · OpenMPI (EFA build) · nvbandwidth `main` (2026-06) · arm64 (Grace), `sm_100` (GB200) / `sm_103` (GB300). On EKS: K8s ≥ 1.33, NVIDIA DRA driver for the ComputeDomain.
