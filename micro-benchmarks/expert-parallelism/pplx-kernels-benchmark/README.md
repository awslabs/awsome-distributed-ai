# Perplexity Kernels Benchmark
<https://github.com/ppl-ai/pplx-kernels>

Updated to [NVSHMEM 3.4.5-0](https://github.com/NVIDIA/nvshmem/commit/df2814155acfba6227534dd81a8bf338da9e55f2) and PPLX-KERNELS [Aug 6, 2025](https://github.com/ppl-ai/pplx-kernels/commit/12cecfda252e4e646417ac263d96e994d476ee5d)

## Git clone NVSHMEM

3.4.5-0:

```
git clone https://github.com/NVIDIA/nvshmem.git && cd ./nvshmem && git checkout df2814155acfba6227534dd81a8bf338da9e55f2 && cd ..
```

devel brach:

```
git clone https://github.com/NVIDIA/nvshmem.git && cd ./nvshmem && git checkout devel && cd ..
```

## Building Perplexity Kernels Docker image

```bash
GDRCOPY_VERSION=v2.5.1
EFA_INSTALLER_VERSION=1.43.2
NCCL_VERSION=v2.27.7-1
NCCL_TESTS_VERSION=v2.16.9
NVSHMEM_VERSION=3.4.5-0
TAG="efa${EFA_INSTALLER_VERSION}-nccl${NCCL_VERSION}-tests${NCCL_TESTS_VERSION}-nvshmem${NVSHMEM_VERSION}"
PPLX_CONTAINER_IMAGE_NAME_TAG="pplx-kernels:${TAG}"
```

```bash
docker build --progress=plain -f ./pplx-kernels.Dockerfile \
       --build-arg="EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}" \
       --build-arg="NCCL_VERSION=${NCCL_VERSION}" \
       --build-arg="NCCL_TESTS_VERSION=${NCCL_TESTS_VERSION}" \
       --build-arg="NVSHMEM_VERSION=${NVSHMEM_VERSION}" \
       -t ${PPLX_CONTAINER_IMAGE_NAME_TAG} \
       .
```

```bash
enroot import -o ./pplx-kernels.sqsh dockerd://${PPLX_CONTAINER_IMAGE_NAME_TAG}
```

## Running Perplexity Kernels Benchmark

```bash
sbatch pplx-kernels.sbatch
```

## Check the logs

```bash
tail -f -n +0 slurm-XXX.out
```

## Core dump

1. run `ulimit -c unlimited` and check that `srun -N <num of nodes> bash -c "ulimit -c"` should print `unlimited` <num of nodes> times
2. run `srun -N <num of nodes> sudo bash -c "mkdir -p /tmp/coredump && echo '/tmp/coredump/core.%e.%p' > /proc/sys/kernel/core_pattern"`
3. run `sbatch pplx-kernels.sbatch`
