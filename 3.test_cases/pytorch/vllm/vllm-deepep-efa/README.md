# vLLM serving with DeepEP expert-parallel all-to-all over EFA

Builds a vLLM serving image whose **MoE expert all-to-all runs over EFA** via DeepEP, so that
`--all2all-backend deepep_low_latency` / `deepep_high_throughput` work when the expert-parallel
group spans more than one node.

This is the serving counterpart to
[`micro-benchmarks/expert-parallelism/deepep-benchmark`](../../../../micro-benchmarks/expert-parallelism/deepep-benchmark/),
which validates DeepEP's dispatch/combine kernels on EFA in isolation. Both images are built from
the **same** [`setup_deepep_efa.sh`](../../../../micro-benchmarks/expert-parallelism/deepep-benchmark/setup_deepep_efa.sh)
— this directory adds no second copy of the patch logic.

## ⚠️ Expert parallelism must be wider than one node to exercise this

`EP` must exceed the GPUs on a node, or DeepEP's all-to-all never touches the fabric and this image
buys nothing over the stock one. On an 8-GPU node, `TP=4 × DP=2` gives `EP=8` — exactly one node —
so every expert is local and dispatch/combine stays on NVSwitch. Getting expert traffic onto EFA
needs e.g. `EP=16` across two nodes.

Note this is independent of *disaggregated prefill*: a 1P+1D or 1P+4D topology is multi-node, but if
each role is its own 8-GPU deployment then each role still holds all 256 experts locally and only
the KV transfer crosses the network.

## Version constraints

Inherited from the DeepEP EFA patch, which is validated against exactly one commit:

| Component | Version | Why pinned |
|---|---|---|
| DeepEP | [`567632d`](https://github.com/deepseek-ai/DeepEP/tree/567632dd59810d77b3cc05553df953cc0f779799) | The patch applies to this tree only; `setup_deepep_efa.sh` refuses others without `--force`. Targets the legacy NVSHMEM path, not EPv2/NCCL-GIN. |
| NVSHMEM | ≥ `3.7.0-0` | The patch calls `nvshmemx_qp_*` host-proxy APIs, which do not exist earlier. |
| CUDA | 13 | Matches the base image's torch `cu130`; also required for Blackwell CUPTI. |
| vLLM | `v0.21.0` | Matches the existing benchmarks in this repo. Override with `VLLM_VERSION`. |

## Why this layers on the vLLM release image

`vllm/vllm-openai:v0.21.0` already ships the full **CUDA 13.0 devel toolkit** (`nvcc` 13.0.88),
**torch 2.11.0+cu130**, `ninja`, and CUDA's `cccl` headers — every prerequisite
`setup_deepep_efa.sh` checks for except EFA itself. So the image only adds four layers:

1. build prerequisites (`git`, `cmake`; distro RDMA packages removed so the EFA installer's own
   `libibverbs`/`librdmacm` are not shadowed)
2. **GDRCopy** — NVSHMEM's low-latency path into GPU memory on the proxy
3. **EFA installer 1.48.0** — provides the `libfabric` that NVSHMEM's transport binds to
4. **NVSHMEM 3.7.0 + DeepEP v1 (EFA-patched)** via `setup_deepep_efa.sh`

Rebuilding vLLM from source (as [`dsv3-uccl-nixl`](../dsv3-uccl-nixl/) does, because it also needs
UCCL-EP and NIXL) would add hours of compile time and risk drift from the published release the
existing benchmarks were measured against.

**One trap worth knowing:** the base image ships `nvidia-nvshmem-cu13` **3.4.5** via pip. That is
below the 3.7.0 floor *and* it shadows the standalone install at runtime, so the build removes it in
the same layer and a smoke test asserts it is gone.

### Three things the vLLM base image lacks that `nvidia/cuda:*-devel` provides

`deepep-benchmark`'s own Dockerfile builds `FROM nvidia/cuda:13.0.1-devel-ubuntu22.04`, which sets up
a few things the vLLM image does not. Each of these breaks the DeepEP build in a way whose error
message does not name the real cause, so all three are handled explicitly and commented in place:

| Symptom | Cause | Fix |
|---|---|---|
| `fatal error: cusparse.h: No such file or directory` compiling `internode.cu` | `/usr/local/cuda/include` is a **partial** toolkit: `cublas_v2.h` and `curand.h` are present but `cusparse.h`, `cusolverDn.h` and `cufft.h` are not. DeepEP includes torch's `ATen/cuda/CUDAContextLight.h`, which needs `cusparse.h`. | `CPATH` also points at the pip tree `nvidia/cu13/include`, which has all of them. |
| `/usr/bin/ld: cannot find -lcuda` linking `deep_ep_cpp` | `nvidia/cuda:*-devel` sets `LIBRARY_PATH=/usr/local/cuda/lib64/stubs`; the vLLM image sets it to nothing. The real `libcuda` is not in the image — the nvidia container runtime injects it at `docker run`, which never happens during `docker build`. | Stubs dir added to `LIBRARY_PATH` (**link** time only). |
| `libcuda.so.1 => not found` when a build-time smoke test imports `deep_ep` | Same missing driver, now at load time. The stubs dir does **not** help here: the file is named `libcuda.so` while its `SONAME` is `libcuda.so.1`, so the loader will not resolve it. | The two smoke-test layers prefix `LD_LIBRARY_PATH` with `/usr/local/cuda-13.0/compat`, which ships a real `libcuda.so.1`. |

Note the asymmetry in that last pair: the stubs go on `LIBRARY_PATH` only and the compat dir on
`LD_LIBRARY_PATH` only, **for single layers, never as image `ENV`**. Baking either into the image
would let a fake or version-mismatched driver shadow the one the runtime injects, and every CUDA call
would then fail on a real GPU node.

## Building

`docker build` context must be the `deepep-benchmark` directory, because the Dockerfile
bind-mounts `setup_deepep_efa.sh` from it:

```bash
cd 3.test_cases/pytorch/vllm/vllm-deepep-efa
./build.sh
```

`build.sh` builds the image and then runs `enroot import` to a squashfs on shared storage.
Environment overrides:

| Variable | Default | Notes |
|---|---|---|
| `TORCH_CUDA_ARCH_LIST` | `9.0` | Hopper (p5/p5en). Blackwell: `10.0`. Portable: `9.0;10.0`. `9.0` alone also enables DeepEP's aggressive PTX instructions. |
| `VLLM_VERSION` | `v0.21.0` | Base image tag. |
| `SQSH_PATH` | `/fsx/$USER/containers/<image>-<tag>.sqsh` | **Must be shared storage** — see below. |
| `DO_IMPORT` | `true` | Set `false` to skip the enroot step (e.g. when pushing to ECR instead). |

Or by hand:

```bash
cd micro-benchmarks/expert-parallelism/deepep-benchmark
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -f ../../../3.test_cases/pytorch/vllm/vllm-deepep-efa/vllm-deepep-efa.Dockerfile \
    --build-arg TORCH_CUDA_ARCH_LIST=9.0 \
    -t vllm-deepep:v0.21.0-deepep567632d-efa1.48.0 .
```

The build ends in smoke tests that fail the build rather than a multi-node job: `deep_ep` imports,
vLLM's `has_deep_ep()` returns true, `deep_ep_cpp`'s NVSHMEM resolves inside
`/opt/amazon/nvshmem/lib`, NVSHMEM is ≥ 3.7.0, no pip NVSHMEM `.so` files remain, and the libfabric
transport plugin is present.

## Validating on real nodes

The build-time tests cannot check the one thing that matters most — that expert traffic actually
moves over EFA — because the nvidia container runtime only injects the host driver at `docker run`,
never during `docker build`. `validate_2node.sbatch` covers that on two GPU nodes:

```bash
sbatch validate_2node.sbatch /fsx/$USER/containers/vllm-deepep-efa.sqsh
```

It asserts the four transport vars survived `enroot import` into the squashfs, that the driver and
EFA endpoints are visible inside the container, and then runs DeepEP's own `test_internode.py`.

Measured on 2 × p5en.48xlarge (16 × H200), job `11495`, all steps `COMPLETED 0:0` with zero errors
and zero recv timeouts:

| | RDMA (EFA) | NVLink |
|---|---|---|
| Best dispatch (BF16) | **71.30 GB/s** | 232.71 GB/s |
| Best combine | **63.38 GB/s** | 206.88 GB/s |

Those RDMA figures land within 0.6% of the standalone micro-benchmark's published result on the same
hardware (71.73 dispatch / 63.27 combine in
[`benchmarks/inference-.../ep-dispatch-combine-bf16/p5en-2nodes`](../../../../micro-benchmarks/expert-parallelism/deepep-benchmark/)),
which is the intended check: adding vLLM to the image costs nothing on the DeepEP path.

Note this validation uses **one Slurm task per node**. DeepEP's tests spawn their own 8 ranks per
node via `torch.multiprocessing`, so `--ntasks-per-node 8` makes every task build its own 8-rank
group and the run dies on `num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode` or `EADDRINUSE`.

### Slurm / pyxis notes

- **Use `enroot import`, not `mksquashfs`.** A hand-rolled squashfs of a container export drops the
  image's baked-in `ENV`, which here includes the four NVSHMEM/libfabric variables the transport
  depends on — producing a runtime failure that looks like a fabric problem.
- **Write the `.sqsh` to shared storage.** On ParallelCluster `/scratch` is per-login-node (and the
  docker image store lives there), so an image built on one login node is invisible from the other
  and a `.sqsh` under `/scratch` is invisible to compute nodes. Pin the whole build to one login
  node and put the `.sqsh` on `/fsx`.
- **Use a private enroot cache, on `/fsx`.** The cluster-wide `ENROOT_CACHE_PATH` is world-shared
  with other users' layers at mode `640`, so `tar` fails mid-import on a collision. `build.sh` sets a
  private one under `/fsx/$USER/enroot-tmp/`. It deliberately does **not** use `/scratch`: the image
  is ~35 GB, enroot unpacks every layer into `ENROOT_TEMP_PATH` before writing the squashfs, and
  `/scratch` also holds the docker image store and is routinely >90% full — the import then dies
  partway with an ENOSPC that reads as image corruption.

## Runtime environment

The four transport variables are baked into the image as `ENV`, so every rank inherits them:

| Variable | Value | Purpose |
|---|---|---|
| `FI_PROVIDER` | `efa` | Selects the EFA provider in libfabric. |
| `NVSHMEM_REMOTE_TRANSPORT` | `libfabric` | Routes NVSHMEM's inter-node RDMA through libfabric instead of IBGDA. |
| `NVSHMEM_LIBFABRIC_PROVIDER` | `efa` | Tells that transport to use EFA. |
| `NVSHMEM_NETDEVS_POLICY` | `EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE` | Pairs each PE with the NIC under its PCIe switch on multi-NIC nodes. |

**`NVSHMEM_REMOTE_TRANSPORT=libfabric` is the one that matters most.** DeepEP's internode kernels
were written for IBGDA (GPU-initiated RDMA over InfiniBand), which EFA does not provide. Without it
NVSHMEM falls back to IBGDA and the run dies with `DeepEP error: CPU recv timeout` — a
missing-configuration symptom that is easy to misread as an EFA or cluster defect. Re-export them at
the launch surface anyway if you want them visible per-job.

## Verifying DeepEP is actually in the data path

Do not assume the flag took effect — vLLM silently falls back. Check the server log:

```bash
grep -iE "all2all|AgRs|deepep" server.log
```

`Using DeepEPLLAll2AllManager` (or `DeepEPHTAll2AllManager`) means DeepEP is live.
`Using AgRsAll2AllManager` means it is **not** — that is vLLM 0.21.0's default
`allgather_reducescatter`. Two known traps:

- `--all2all-backend naive` logs *"has been removed. Falling back to 'allgather_reducescatter'"*.
  vLLM 0.21.0 has no naive backend.
- `Using MoEPrepareAndFinalizeNaiveDPEPModular` is the MoE prepare/finalize module, a **different
  layer**. It contains the word "Naive" and says nothing about which all-to-all backend is running.

Also note `deepep_low_latency` sizes its RDMA buffer from `--max-num-batched-tokens`, so a large
token budget can OOM the workers at startup with
`CUDA error ... 'out of memory'` from `uccl_ep.cc`. Lower `--max-num-batched-tokens` rather than
concluding DeepEP is broken.
