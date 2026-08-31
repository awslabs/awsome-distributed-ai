# DeepEP Benchmark — Slurm

Run the DeepEP micro-benchmarks on a Slurm cluster with [Pyxis](https://github.com/NVIDIA/pyxis)
and [Enroot](https://github.com/NVIDIA/enroot). Build the container image first by following the
[top-level README](../README.md#building-the-deepep-image).

## 1. Import the container image

Convert the Docker image into an Enroot squash file. The `.sbatch` scripts reference
`./deepep.sqsh` relative to the directory you submit from, so import it into this `slurm/`
directory:

```bash
cd slurm
enroot import -o ./deepep.sqsh dockerd://${DEEPEP_CONTAINER_IMAGE_NAME_TAG}
```

## 2. Submit the benchmarks

Each job is exclusive and uses one task per node (the test spawns 8 local GPU ranks per node
internally via `torch.multiprocessing`).

```bash
# Single-node intranode (NVLink) test
sbatch test_intranode.sbatch

# Two-node internode (RDMA over EFA) test
sbatch test_internode.sbatch

# Two-node low-latency (decode-path) test
sbatch test_low_latency.sbatch
```

The internode and low-latency scripts set `MASTER_ADDR` to the first node in the allocation and
export `WORLD_SIZE=$SLURM_NNODES`; each rank reads `RANK=$SLURM_PROCID`. Results print to the
Slurm job output (`slurm-<jobid>.out`).

## Environment variables

The scripts export the EFA / NVSHMEM settings the libfabric transport needs:

| Variable | Value |
|----------|-------|
| `FI_PROVIDER` | `efa` |
| `NVSHMEM_REMOTE_TRANSPORT` | `libfabric` |
| `NVSHMEM_LIBFABRIC_PROVIDER` | `efa` |
| `NVSHMEM_NETDEVS_POLICY` | `EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE` |

These are also baked into the image `ENV`; the sbatch scripts re-export them so the transport
config is visible and overridable per-job. `NVSHMEM_NETDEVS_POLICY` controls NIC↔PE assignment on
multi-NIC EFA nodes. See the [benchmark README](../README.md#how-the-efa-support-works) for the full rationale.

For the Kubernetes/EKS equivalent of these jobs, see [`../kubernetes/`](../kubernetes/).
