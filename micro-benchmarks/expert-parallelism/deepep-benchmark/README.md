# DeepEP Benchmark

[DeepEP](https://github.com/deepseek-ai/DeepEP) is a communication library for
Mixture-of-Experts (MoE) **expert parallelism** — its dispatch and combine kernels perform the
GPU all-to-all that routes tokens to experts and gathers the results back. This directory builds
DeepEP with an **EFA (libfabric) NVSHMEM transport** and runs its three micro-benchmarks on AWS
GPU clusters:

| Benchmark | Nodes | Measures |
|-----------|-------|----------|
| `test_intranode` | 1 | intra-node dispatch/combine over NVLink |
| `test_internode` | 2 | inter-node dispatch/combine (RDMA over EFA) |
| `test_low_latency` | 2 | low-latency kernels used on the decode path |

Run them on **Slurm** ([`slurm/`](./slurm/)) or **Kubernetes / EKS** ([`kubernetes/`](./kubernetes/)).

## ⚠️ Version constraints

> This particular version works only with NVSHMEM >= [3.7.0-0](https://github.com/NVIDIA/nvshmem/tree/v3.7.0-0)
> and DeepEP v1 with the **NVSHMEM backend** <= [567632d of Feb 3, 2026](https://github.com/deepseek-ai/DeepEP/tree/567632dd59810d77b3cc05553df953cc0f779799).

This targets DeepEP's legacy NVSHMEM code path (pre-EPv2). EPv2 restructures the kernel sources
and switches to the NCCL GIN backend; it is out of scope here.

## How the EFA support works

DeepEP's internode kernels were written for [IBGDA](https://docs.nvidia.com/nvshmem/) (GPU-initiated
RDMA over InfiniBand), which EFA does not provide. The image works around this:

- **NVSHMEM is built from source** with the libfabric transport
  (`NVSHMEM_LIBFABRIC_SUPPORT=ON`, IBGDA/IBRC off) so RDMA runs over EFA.
- [`setup_deepep_efa.sh`](./setup_deepep_efa.sh) **patches DeepEP** to replace its IBGDA device
  calls with NVSHMEM host-proxy QP APIs (`nvshmemx_qp_*`), maps the dispatch/combine tail updates
  onto a combined put+signal, and raises `NVSHMEM_MAX_TEAMS` for NVSHMEM 3.7.
- The build runs in a **Python virtual environment** (`/opt/deepep`); PyTorch's bundled
  `nvidia-nvshmem-cu13` is uninstalled so it does not clash with the from-source NVSHMEM 3.7.0-0.
- The image is built on **CUDA 13** (PyTorch `cu130`). This is required for Blackwell (B200/B300):
  CUDA ≤ 12.9 CUPTI fails on those GPUs (`CUPTI_ERROR_INVALID_DEVICE`), which breaks the
  internode/low-latency tuning phase that profiles kernels.

At runtime the NVSHMEM libfabric transport is selected by these environment variables (set in the
image and re-exported by the launchers):

```
FI_PROVIDER=efa
NVSHMEM_REMOTE_TRANSPORT=libfabric
NVSHMEM_LIBFABRIC_PROVIDER=efa
```

## Prerequisites

- EFA-enabled GPU nodes.
- **GPU architecture:** the image is built for **both Hopper (`sm_90`) and Blackwell (`sm_100`)**
  by default, so the same image runs on `p5`/`p5en` and `p6-b300` nodes. To build a smaller
  single-arch image, override `--build-arg GPU_ARCH=90` (Hopper only) or `GPU_ARCH=100`
  (Blackwell only).

## Building the DeepEP image

```bash
GDRCOPY_VERSION=v2.5.2
EFA_INSTALLER_VERSION=1.48.0
NVSHMEM_VERSION=v3.7.0-0
DEEPEP_COMMIT=567632d
GPU_ARCH="90;100"  # Hopper + Blackwell by default; use "90" or "100" for a single-arch image
TAG="efa${EFA_INSTALLER_VERSION}-nvshmem${NVSHMEM_VERSION}-deepep${DEEPEP_COMMIT}"
DEEPEP_CONTAINER_IMAGE_NAME_TAG="deepep:${TAG}"
```

```bash
docker build --progress=plain -f ./deepep.Dockerfile \
      --build-arg="GDRCOPY_VERSION=${GDRCOPY_VERSION}" \
      --build-arg="EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}" \
      --build-arg="NVSHMEM_VERSION=${NVSHMEM_VERSION}" \
      --build-arg="DEEPEP_COMMIT=${DEEPEP_COMMIT}" \
      --build-arg="GPU_ARCH=${GPU_ARCH}" \
      -t ${DEEPEP_CONTAINER_IMAGE_NAME_TAG} \
      .
```

## Running the benchmark

- **Slurm** (Pyxis/Enroot): see [`slurm/README.md`](./slurm/README.md).
- **Kubernetes / EKS** (MPIJob): see [`kubernetes/README.md`](./kubernetes/README.md).
