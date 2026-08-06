#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Build the vLLM + DeepEP + EFA image and (optionally) import it to a squashfs
# for pyxis/enroot on Slurm.
#
# The build context is the deepep-benchmark directory so that the bind-mounted
# setup_deepep_efa.sh is the repo's single copy rather than a duplicate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_CONTEXT="${REPO_ROOT}/micro-benchmarks/expert-parallelism/deepep-benchmark"

VLLM_VERSION="${VLLM_VERSION:-v0.21.0}"
EFA_INSTALLER_VERSION="${EFA_INSTALLER_VERSION:-1.48.0}"
NVSHMEM_VERSION="${NVSHMEM_VERSION:-3.7.0}"
DEEPEP_COMMIT="${DEEPEP_COMMIT:-567632d}"
# Hopper (p5/p5en) by default. Blackwell: "10.0". Portable: "9.0;10.0".
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

IMAGE="${IMAGE:-vllm-deepep}"
TAG="${TAG:-v${VLLM_VERSION#v}-deepep${DEEPEP_COMMIT}-efa${EFA_INSTALLER_VERSION}}"
FULL_IMAGE="${IMAGE}:${TAG}"

# Where to write the .sqsh. Must be on SHARED storage: /scratch is per-login-node
# on ParallelCluster, so a squashfs written there is invisible to compute nodes.
SQSH_DIR="${SQSH_DIR:-/fsx/${USER}/containers}"
SQSH_PATH="${SQSH_PATH:-${SQSH_DIR}/${IMAGE}-${TAG}.sqsh}"

DO_IMPORT="${DO_IMPORT:-true}"

if [[ ! -f "${BUILD_CONTEXT}/setup_deepep_efa.sh" ]]; then
    echo "ERROR: setup_deepep_efa.sh not found in ${BUILD_CONTEXT}" >&2
    exit 1
fi

echo "==> Building ${FULL_IMAGE}"
echo "    base            vllm/vllm-openai:${VLLM_VERSION}"
echo "    EFA installer   ${EFA_INSTALLER_VERSION}"
echo "    NVSHMEM         ${NVSHMEM_VERSION}"
echo "    DeepEP commit   ${DEEPEP_COMMIT}"
echo "    CUDA arch       ${TORCH_CUDA_ARCH_LIST}"
echo "    context         ${BUILD_CONTEXT}"

DOCKER_BUILDKIT=1 docker build --progress=plain \
    --platform linux/amd64 \
    -f "${SCRIPT_DIR}/vllm-deepep-efa.Dockerfile" \
    --build-arg "VLLM_VERSION=${VLLM_VERSION}" \
    --build-arg "EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}" \
    --build-arg "NVSHMEM_VERSION=${NVSHMEM_VERSION}" \
    --build-arg "DEEPEP_COMMIT=${DEEPEP_COMMIT}" \
    --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
    -t "${FULL_IMAGE}" \
    "${BUILD_CONTEXT}"

echo "==> Built ${FULL_IMAGE}"
docker image ls "${IMAGE}" --format '    {{.Repository}}:{{.Tag}}  {{.Size}}'

if [[ "${DO_IMPORT}" != "true" ]]; then
    echo "==> DO_IMPORT=false; skipping enroot import"
    exit 0
fi

if ! command -v enroot >/dev/null 2>&1; then
    echo "==> enroot not found; skipping import (this is normal off-cluster)"
    exit 0
fi

# enroot import (NOT mksquashfs): it reads the image through dockerd and
# preserves the layer metadata, environment and entrypoint. A hand-rolled
# mksquashfs of a container export loses the ENV baked into the image -- which
# here includes the four NVSHMEM/libfabric variables the transport depends on.
#
# Private cache: the cluster-wide default ENROOT_CACHE_PATH is world-shared and
# other users' layers are mode 640, so tar fails mid-import on any collision.
#
# Both live on /fsx, not /scratch. This image is ~35 GB and enroot unpacks every
# layer into TEMP before building the squashfs, so the import needs roughly the
# image size again in scratch space. /scratch is a shared per-login-node ephemeral
# volume that also holds the docker image store, and it is routinely >90% full --
# an import there fails partway with a confusing ENOSPC.
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-/fsx/${USER}/enroot-tmp/cache}"
export ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-/fsx/${USER}/enroot-tmp/tmp}"
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_TEMP_PATH}" "$(dirname "${SQSH_PATH}")"

echo "==> Importing to ${SQSH_PATH}"
rm -f "${SQSH_PATH}"
enroot import -o "${SQSH_PATH}" "dockerd://${FULL_IMAGE}"

ls -la "${SQSH_PATH}"
echo "==> Done."
echo "    Use with: srun --container-image=${SQSH_PATH} ..."
