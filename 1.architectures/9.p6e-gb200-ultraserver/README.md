# P6e-GB200 UltraServer Reference Architecture (NVL72, IMEX)

Stand up a **P6e-GB200 UltraServer** — 18 `p6e-gb200.36xlarge` instances federated into one **72-GPU NVLink domain** (NVL36x2) — on **both** orchestrators, with the platform's mandatory IMEX wiring. This is the architecture that every GB200 training/inference test case in this repo launches onto.

> **GB200 ≠ B200/B300.** This is Grace-Blackwell (ARM Grace CPU + Blackwell, aarch64, coherent NVLink-C2C, a 72-GPU NVLink domain spanning 18 instances, mandatory IMEX). It is **not** the HGX B200/B300 layout (x86, 8-GPU NVLink island per node, no Grace, no IMEX). See `architectures/aws-pcs/assets/add-cng-p6-b200.yaml` for the HGX path.

## Platform constraints (read first)

- **IMEX is mandatory.** Cross-instance NVLink memory coherency inside the UltraServer requires NVIDIA IMEX — a DRA `ComputeDomain` on EKS, or a prolog-configured IMEX service on Slurm. Without it the 72-GPU domain does not form.
- **CDMM disables MIG.** Coherent Driver-based Memory Management (NVIDIA driver 580+) is enabled on GB200; **MIG and vGPU are not supported**.
- **Capacity-Blocks only.** P6e-GB200 is acquired through EC2 Capacity Blocks for ML (Dallas Local Zone as of 2026-06) — no on-demand, no spot.
- **Clique = domain.** The NVLink domain is labelled by `nvidia.com/gpu.clique` (EKS) and corresponds to the `capacityBlockId` from the EC2 Instance Topology API; co-schedule the whole job to one clique.

## EKS / HyperPod-EKS path (`eks/`)

| File | Purpose |
|---|---|
| `gpu-operator-values.yaml` | GPU Operator Helm values: `migManager.enabled=false`, `driver.enabled=false`, `toolkit.enabled=false` (AMI ships driver 580 / CDMM) |
| `nvidia-dra-driver-values.yaml` | NVIDIA DRA driver Helm values: `computeDomains.enabled=true`, `gpus.enabled=false` (IMEX ComputeDomain support) |
| `computedomain-example.yaml` | A `ComputeDomain` + a clique-pinned validation Job (runs `nvbandwidth` over the domain) |
| `capacity-block-nodegroup.md` | Managed/self-managed node group on a `CAPACITY_BLOCK` launch template, AL2023-ARM-NVIDIA EKS AMI |

Prereqs: EKS ≥ 1.33, NVIDIA DRA driver 25.8.0+, GPU Operator 25.3.4+, EFA device plugin 0.5.14+, AL2023-ARM-NVIDIA EKS AMI v20251103+.

## ParallelCluster / HyperPod-Slurm path (`parallelcluster/`)

| File | Purpose |
|---|---|
| `cluster-p6e-gb200.yaml` | ParallelCluster 3.14.0 config: a `CAPACITY_BLOCK` SlurmQueue of 18 `p6e-gb200.36xlarge`, `DisableSimultaneousMultithreading: true`, `Efa.Enabled: true`, `PrologFlags: "Alloc,NoHold"` |
| `91_nvidia_imex_prolog.sh` | Head-node `OnNodeStart` custom action that configures and starts the IMEX service across the allocation |
| `validate-imex.sh` | `nvidia-imex-ctl -N` check expecting `Domain State: UP` under `--exclusive` |

HyperPod-Slurm alternative: where Slurm ≥ 24.05 is available, use the `switch/nvidia_imex` plugin instead of the prolog (set in `slurm.conf`); the rest of the config is the same.

## Validate the domain

After either path is up, confirm the NVLink domain formed before running real work:
```bash
# Slurm
sbatch parallelcluster/validate-imex.sh           # expects "Domain State: UP" on all 18 nodes
sbatch ../../micro-benchmarks/nvbandwidth/slurm/nvbandwidth.sbatch
# EKS
kubectl apply -f eks/computedomain-example.yaml
kubectl apply -f ../../micro-benchmarks/nccl-tests/kubernetes/nccl-tests-gb200.yaml
```
Then make the NCCL correctness gate in `4.validation_and_observability` mandatory before production (it catches silent cross-asset-group NVLink corruption that passes single-node diagnostics).

## Testability

**Authored-to-spec** unless a P6e-GB200 Capacity Block is held in the Dallas Local Zone. The IaC, version pins, and validation commands are drawn from AWS/NVIDIA Tier-1 docs and are internally consistent; a live 18-instance domain is the only way to confirm end-to-end clique placement, IMEX channel establishment, and the EFA cross-UltraServer path. Treat as a spec-grade build plan; mark claims verified once a Capacity-Block run confirms them.
