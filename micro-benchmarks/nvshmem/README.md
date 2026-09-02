# NVSHMEM

NVIDIA NVSHMEM is NVIDIA’s implementation of the OpenSHMEM [PGAS](https://en.wikipedia.org/wiki/Partitioned_global_address_space) model for GPU clusters. It provides an easy-to-use CPU-side interface to allocate pinned memory that is symmetrically distributed across a cluster of NVIDIA GPUs. NVSHMEM can significantly reduce communication and coordination overheads by allowing programmers to perform these operations from within CUDA kernels and on CUDA streams.

One of the options for using the NVSHMEM is to implement high-throughput and low-latency MoE dispatch and combine GPU kernels. [DeepEP](https://github.com/deepseek-ai/DeepEP) and [pplx-kernels](https://github.com/ppl-ai/pplx-kernels) are examples of such implementations.

The goal of this document is to provide a guide on how to build NVSHMEM with NCCL with AWS EFA support and run the performance tests. This document reuses NCCL Tests Docker image as a base image and adds NVSHMEM on top. This is done because NVSHMEM is built with NCCL.

### Building NCCL Tests Docker image

For more details on how to build the NCCL Tests Docker image, please refer to the [NCCL Tests README](../nccl-tests/README.md).

```bash
GDRCOPY_VERSION=v2.5.1
EFA_INSTALLER_VERSION=1.43.2
AWS_OFI_NCCL_VERSION=v1.16.3
NCCL_VERSION=v2.27.7-1
NCCL_TESTS_VERSION=v2.16.9
TAG="efa${EFA_INSTALLER_VERSION}-ofi${AWS_OFI_NCCL_VERSION}-nccl${NCCL_VERSION}-tests${NCCL_TESTS_VERSION}"
NCCL_CONTAINER_IMAGE_NAME_TAG="nccl-tests:${TAG}"
```

```bash
docker build --progress=plain -f ../nccl-tests/nccl-tests.Dockerfile \
       --build-arg="EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}" \
       --build-arg="AWS_OFI_NCCL_VERSION=${AWS_OFI_NCCL_VERSION}" \
       --build-arg="NCCL_VERSION=${NCCL_VERSION}" \
       --build-arg="NCCL_TESTS_VERSION=${NCCL_TESTS_VERSION}" \
       -t ${NCCL_CONTAINER_IMAGE_NAME_TAG} \
       .
```

### Building NVSHMEM Docker image on top of NCCL Tests Docker base image

```bash
NVSHMEM_VERSION=3.3.9
TAG="efa${EFA_INSTALLER_VERSION}-ofi${AWS_OFI_NCCL_VERSION}-nccl${NCCL_VERSION}-tests${NCCL_TESTS_VERSION}-nvshmem${NVSHMEM_VERSION}"
NVSHMEM_CONTAINER_IMAGE_NAME_TAG="nvshmem:${TAG}"
```

```bash
docker build --progress=plain -f nvshmem.Dockerfile \
       --build-arg="EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}" \
       --build-arg="AWS_OFI_NCCL_VERSION=${AWS_OFI_NCCL_VERSION}" \
       --build-arg="NCCL_VERSION=${NCCL_VERSION}" \
       --build-arg="NCCL_TESTS_VERSION=${NCCL_TESTS_VERSION}" \
       --build-arg="NVSHMEM_VERSION=${NVSHMEM_VERSION}" \
       -t ${NVSHMEM_CONTAINER_IMAGE_NAME_TAG} \
       .
```

### Slurm

To run the NCCL tests on Slurm, you will need to convert the container into a Squash file using Enroot.

Convert the container image to a squash file via Enroot. If you have the built image locally use the following command:

```bash
enroot import -o ./nvshmem.sqsh dockerd://${NVSHMEM_CONTAINER_IMAGE_NAME_TAG}
```

# Perf Test

NVSHMEM provides rich set of performance tests for different operations launched on both device and host.

Common arguments:

* `-b, --min_size <minbytes>` - Minimum message size in bytes
* `-e, --max_size <maxbytes>` - Maximum message size in bytes  
* `-f, --step <step factor>` - Step factor for message sizes
* `-n, --iters <number>` - Number of iterations
* `-w, --warmup_iters <number>` - Number of warmup iterations
* `-c, --ctas <number>` - Number of CTAs to launch (used in some device pt-to-pt tests)
* `-t, --threads_per_cta <number>` - Number of threads per block (used in some device pt-to-pt tests)
* `-d, --datatype <type>` - Data type: int, int32_t, uint32_t, int64_t, uint64_t, long, longlong, ulonglong, size, ptrdiff, float, double, fp16, bf16
* `-o, --reduce_op <op>` - Reduction operation: min, max, sum, prod, and, or, xor
* `-s, --scope <scope>` - Thread group scope: thread, warp, block, all
* `-i, --stride <number>` - Stride between elements
* `-a, --atomic_op <op>` - Atomic operation: inc, add, and, or, xor, set, swap, fetch_inc, add, and, or, xor, compare_swap
* `--bidir` - Run bidirectional test
* `--msgrate` - Report message rate (MMPs)
* `--dir <direction>` - Direction (read/write) for put/get operations
* `--issue <mode>` - Issue mode (on_stream/host) for some host pt-to-pt tests

## Device

### Collective

Device collective tests are located in `/opt/nvshmem/bin/perftest/device/collective/`:

* alltoall_latency
* barrier_latency
* bcast_latency
* fcollect_latency
* redmaxloc_latency
* reducescatter_latency
* reduction_latency
* sync_latency

### Point-to-Point

Device point-to-point tests are located in `/opt/nvshmem/bin/perftest/device/pt-to-pt/`:

* shmem_atomic_bw:
* shmem_atomic_latency
* shmem_atomic_ping_pong_latency
* shmem_g_bw
* shmem_g_latency
* shmem_get_bw
* shmem_get_latency
* shmem_p_bw
* shmem_p_latency
* shmem_p_ping_pong_latency
* shmem_put_atomic_ping_pong_latency
* shmem_put_bw
* shmem_put_latency
* shmem_put_ping_pong_latency
* shmem_put_signal_ping_pong_latency
* shmem_signal_ping_pong_latency
* shmem_st_bw

## Host

### Collectives

Host collective tests are located in `/opt/nvshmem/bin/perftest/host/collective/`:

* alltoall_on_stream
* barrier_all_on_stream
* barrier_on_stream
* broadcast_on_stream
* fcollect_on_stream
* reducescatter_on_stream
* reduction_on_stream
* sync_all_on_stream

### Point-to-Point

Host point-to-point tests are located in `/opt/nvshmem/bin/perftest/host/pt-to-pt/`:

* bw
* latency
* stream_latency

### Example of running shmem_put_bw benchmark on 2 GPUs on a single node and 2 GPUs on two different nodes

NVSHMEM shmem_put_bw benchmark requires 2 processing elements (PEs), so there are two options:

benchmark 2 GPUs on a single node over NVLink:

```bash
srun --mpi=pmix --cpu-bind=none --container-image ./nvshmem.sqsh --nodes=1 --ntasks-per-node=2 /opt/nvshmem/bin/perftest/device/pt-to-pt/shmem_put_bw
```

benchmark 2 GPUs on two different nodes over AWS EFA:

```bash
srun --mpi=pmix --cpu-bind=none --container-image ./nvshmem.sqsh --nodes=2 --ntasks-per-node=1 /opt/nvshmem/bin/perftest/device/pt-to-pt/shmem_put_bw
```
