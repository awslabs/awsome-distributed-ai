# DeepEP Benchmark

## !!! This particular version works only with NVSHMEM >= [3.7.0-0](https://github.com/NVIDIA/nvshmem/tree/v3.7.0-0) and DeepEP v1 with NVSHMEM backend <= [567632d of Feb 3, 2026](https://github.com/deepseek-ai/DeepEP/tree/567632dd59810d77b3cc05553df953cc0f779799) !!!

## Building DeepEP Docker image

```bash
GDRCOPY_VERSION=v2.5.2
EFA_INSTALLER_VERSION=1.48.0
NVSHMEM_VERSION=v3.7.0-0
DEEPEP_COMMIT=567632d
TAG="efa${EFA_INSTALLER_VERSION}-nvshmem${NVSHMEM_VERSION}-deepep${DEEPEP_COMMIT}"
DEEPEP_CONTAINER_IMAGE_NAME_TAG="deepep:${TAG}"
```

```bash
docker build --progress=plain -f ./deepep.Dockerfile \
      --build-arg="GDRCOPY_VERSION=${GDRCOPY_VERSION}" \
      --build-arg="EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}" \
      --build-arg="NVSHMEM_VERSION=${NVSHMEM_VERSION}" \
      --build-arg="DEEPEP_COMMIT=${DEEPEP_COMMIT}" \
      -t ${DEEPEP_CONTAINER_IMAGE_NAME_TAG} \
      .
```

```bash
enroot import -o ./deepep.sqsh dockerd://${DEEPEP_CONTAINER_IMAGE_NAME_TAG}
```

## Running DeepEP Benchmark

```bash
sbatch test_intranode.sbatch
```

```bash
sbatch test_internode.sbatch
```

```bash
sbatch deepep-test_low_latency.sbatch
```
