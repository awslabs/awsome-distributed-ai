#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Import the PREBUILT public NCCL-tests image into a squashfs on /fsx, ready for
# Pyxis (02-nccl-tests.sbatch). No local docker build — pulls the published image:
#   public.ecr.aws/hpc-cloud/nccl-tests:efa1.48.0-ncclv2.30.4-1-testsv2.18.3
#
# Run on a GPU node (it has enroot + the public ECR is anonymous-pullable):
#   srun --partition=gpu-p5 --nodes=1 --exclusive bash 02-import-nccl-image.sh
#
# IMPORTANT: enroot's default cache/temp live on the small root disk (~72G,
# largely full on the DLAMI) which overflows during import; but FSx Lustre does
# NOT support the overlayfs that `enroot import` mounts ("failed to mount
# overlay: Invalid argument"). The p5 nodes have a ~1 TB /dev/shm tmpfs, so we
# stage all enroot temp/cache/data there and write only the final .sqsh to /fsx.

set -euxo pipefail

# Use a tag that actually exists in public.ecr.aws/hpc-cloud/nccl-tests. The
# README's example tag may be stale; 'latest' always resolves. Override IMAGE_TAG
# to pin a specific build (see `enroot import`-able tags via the public ECR API).
: "${IMAGE_TAG:=latest}"
: "${OUT:=/fsx/validation/nccl-tests.sqsh}"

# enroot URI form is docker://[USER@][REGISTRY#]REPO[:TAG]. The registry must be
# separated from the repo with '#', otherwise enroot defaults to Docker Hub and
# the public-ECR path 401s.
: "${IMAGE_URI:=docker://public.ecr.aws#hpc-cloud/nccl-tests:${IMAGE_TAG}}"

SHM_BASE="/dev/shm/enroot-$(id -u)"
export ENROOT_TEMP_PATH="${SHM_BASE}/tmp"
export ENROOT_CACHE_PATH="${SHM_BASE}/cache"
export ENROOT_DATA_PATH="${SHM_BASE}/data"
mkdir -p "${ENROOT_TEMP_PATH}" "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}"
trap 'rm -rf "${SHM_BASE}"' EXIT

mkdir -p "$(dirname "${OUT}")"
rm -f "${OUT}"
enroot import -o "${OUT}" "${IMAGE_URI}"
echo "Wrote ${OUT}"
ls -la "${OUT}"
