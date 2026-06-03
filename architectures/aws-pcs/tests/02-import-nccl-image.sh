#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Import the prebuilt public NCCL-tests image into a squashfs on /fsx, ready for
# Pyxis (02-nccl-tests.sbatch). No local docker build — pulls the published image:
#   public.ecr.aws/hpc-cloud/nccl-tests
#
# Run this DIRECTLY ON THE LOGIN/HEAD NODE (not as a batch job). The login node
# has Enroot and a 300 GiB root disk (RootVolumeSize default), so the import works
# without any special staging. Only the final .sqsh is written to shared /fsx, so
# every compute node can use it:
#
#   ssh pcs-login            # or: aws ssm start-session --target <login-id>
#   bash 02-import-nccl-image.sh
#
# Note: FSx Lustre cannot host the overlayfs that `enroot import` mounts, so the
# import builds on the node-local root disk and writes only the resulting .sqsh to
# /fsx — do not point ENROOT_* at /fsx.

set -euxo pipefail

: "${IMAGE_TAG:=latest}"
: "${OUT:=/fsx/nccl-tests.sqsh}"

# enroot URI form is docker://[USER@][REGISTRY#]REPO[:TAG]. The registry must be
# separated from the repo with '#', otherwise enroot defaults to Docker Hub and
# the public-ECR path 401s.
: "${IMAGE_URI:=docker://public.ecr.aws#hpc-cloud/nccl-tests:${IMAGE_TAG}}"

mkdir -p "$(dirname "${OUT}")"
rm -f "${OUT}"
enroot import -o "${OUT}" "${IMAGE_URI}"
echo "Wrote ${OUT}"
ls -la "${OUT}"
