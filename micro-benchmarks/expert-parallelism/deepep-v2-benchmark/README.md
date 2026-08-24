# DeepEP V2 Benchmark (NCCL GIN / EFA-GDA)

[DeepEP](https://github.com/deepseek-ai/DeepEP) is a communication library for
Mixture-of-Experts (MoE) **expert parallelism** where its dispatch and combine kernels perform the
GPU all-to-all that routes tokens to experts and gathers the results back.

This directory provides:

- **[`setup_deepep_gin.sh`](./setup_deepep_gin.sh)**, which installs the
**DeepEP V2** Python package, the version whose internode transport is **NCCL GIN
(GPU-Initiated Networking)**, into your container or environment (for example a vLLM image
that already ships a DeepEP): it uninstalls any existing `deep_ep`, validates that the NCCL it
is pointed at is GIN-capable (`include/nccl_device.h`, errors out otherwise), and builds +
installs DeepEP against your torch. No arguments needed for the default flow, see `--help`.

Everything else here is reference material for using the script and testing that the
installation works:

- **[`deepep.Dockerfile`](./deepep.Dockerfile)** — a reference image showing the full stack the
  script needs (the EFA installer, which bundles the EFA-GDA-capable aws-ofi-nccl plugin, a
  GIN-capable NCCL, torch), built from a bare CUDA base; it can also be used directly for testing.
- **[`slurm/`](./slurm/)** — launchers that run DeepEP's own `tests/elastic/test_ep.py` to
  validate the installation:

| Benchmark | Nodes | Measures |
|-----------|-------|----------|
| `test-intranode` | 1 | intra-node dispatch/combine over NVLink |
| `test-internode` | 2 | inter-node dispatch/combine (GPU-initiated RDMA over EFA) |

## ⚠️ Version constraints

> - **NCCL >= 2.31** — the GIN device API (`nccl_device.h`) DeepEP V2 links against.
>   The Dockerfile builds NCCL from source and fails the build if the header is missing.
> - **EFA installer >= 1.50** — provides everything EFA-GDA needs in the container (the
>   userspace runtime and the aws-ofi-nccl plugin with the EFA-GDA GIN backend, type 5,
>   this benchmark measures). Installers up to 1.49.0 will not work; the image build fails
>   loudly on a too-old installer. DeepEP's kernels can also run over the CPU-proxy GIN
>   backend, which has no such floor.
> - A host EFA kernel driver **>= 3.3.0**, the first release with the completion-counter API
>   (`efadv_create_comp_cntr`). Run the EFA installer **on the node itself** (not in the
>   container) to install/upgrade the kernel driver; the container ships only the userspace
>   stack. Check with `modinfo efa | grep ^version`; stock AMIs may ship older.
> - The **gdrcopy kernel module (`gdrdrv`) loaded on compute nodes** — the launchers
>   bind-mount `/dev/gdrdrv`, and the plugin's GIN initialization opens a gdr handle.
>   Check with `lsmod | grep gdrdrv`.
> - **DeepEP V2** from [amazon-contributing/DeepEP](https://github.com/amazon-contributing/DeepEP) `main`.

## How the EFA support works

DeepEP V2's internode kernels communicate through **NCCL GIN**: the dispatch/combine kernels
call NCCL's device-side communication API, and a GIN backend carries the traffic. Backends come
in two kinds: GPU-initiated ones, where the NIC work queues are mapped into GPU memory and the
kernels post RDMA themselves, and the CPU-proxy backend (available on EFA as well as IB/RoCE),
where the GPU hands work to a proxy thread. On EFA, the **aws-ofi-nccl** plugin provides the
GPU-initiated backend, **EFA-GDA** (GIN type 5). The launchers pin `NCCL_GIN_TYPE=5` to select
it, if left unset, NCCL selects the CPU-proxy backend when running on EFA.

| Variable | Value | Purpose |
|----------|-------|---------|
| `NCCL_GIN_TYPE` | `5` | Pins the EFA-GDA GIN backend; unset falls back to the CPU-proxy backend. |
| `EP_NCCL_ROOT_DIR` | `$NCCL_HOME` | Points DeepEP's build/runtime checks at the GIN-capable NCCL. |
| `LD_PRELOAD` | `$NCCL_HOME/lib/libnccl.so.2` | Guarantees the GIN-capable NCCL wins over any other on the loader path. |

## Prerequisites

- EFA-enabled GPU nodes with the EFA kernel driver installed (section below).
- **GPU architecture:** the image builds for **Hopper (`sm_90`) and Blackwell (`sm_100`)** by
  default, so one image runs on `p5`/`p5en` and `p6`. Earlier DeepEP revisions required
  **CUDA >= 13.1** on p6/Blackwell (ptxas 13.0.88 rejects the 32-bit `st.bulk` size operand);
  DeepEP main carries the fix ([amazon-contributing/DeepEP#3](https://github.com/amazon-contributing/DeepEP/pull/3)),
  so CUDA 13.0 works as well. Override
  `--build-arg TORCH_CUDA_ARCH_LIST` / `--build-arg NVCC_GENCODE` for a single-arch image.
- Docker with BuildKit, and enroot + pyxis on the cluster for the Slurm flow.

## Using the script in your own container

```bash
# inside a container that already has torch (and possibly an older deep_ep):
./setup_deepep_gin.sh --nccl-root /path/to/gin-capable/nccl
```

The script refuses to run if the NCCL at `--nccl-root` lacks the GIN device API, so it cannot
silently produce a DeepEP that falls back to a slower path. Runtime still requires the
EFA-GDA-capable aws-ofi-nccl GIN plugin (bundled with EFA installer >= 1.50): run the EFA
installer **inside your container** too (with `--skip-kmod`, as the reference Dockerfile does)
to provide the userspace stack and the plugin, then point `NCCL_NET_PLUGIN` and
`NCCL_GIN_PLUGIN` at `/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so`. The host installer run
(previous section) provides only the kernel driver.

## Building the reference image

### External use (public NCCL release + official EFA installer)

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain -f ./deepep.Dockerfile \
  -t deepep-v2:gin .
```

Override `NCCL_REF` or `EFA_INSTALLER_VERSION` for different versions:

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain -f ./deepep.Dockerfile \
  --build-arg NCCL_REF=v2.31.2-1 \
  --build-arg EFA_INSTALLER_VERSION=1.50.0 \
  -t deepep-v2:gin .
```

## Run (Slurm + enroot)

### 1. Convert the Docker image to a squashfs container

```bash
enroot import -o ~/deepep-v2.sqsh dockerd://deepep-v2:gin
```

### 2. Submit the Slurm jobs

```bash
SQSH=~/deepep-v2.sqsh sbatch slurm/test-intranode.sbatch   # 1 node, NVLink
SQSH=~/deepep-v2.sqsh sbatch slurm/test-internode.sbatch   # 2 nodes, EFA-GDA
```

Each job runs `tests/elastic/test_ep.py`, which checks dispatch/combine correctness and prints
per-rank bandwidth lines like:

```
* EP:   3/16 | dispatch: 81 GB/s (SO), 263 GB/s (SU), ... | copy: 3068 GB/s, ...
```

`SO` is scale-out (inter-node over EFA), `SU` scale-up (NVLink). `RUN_EXIT=0` on every rank
means the correctness checks passed.
