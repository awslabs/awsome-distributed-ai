#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Stage 3a: Build the aws-megatron-lm container on a GPU node and import it to a
# squash file on /fsx. The image is ~20 GB; building on the m6i.xlarge login node
# (2 vCPU) is too slow, so we build on a GPU node.

#SBATCH --job-name=build-megatron
#SBATCH --partition=gpu-p5
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=03-build-megatron_%j.out
#SBATCH --error=03-build-megatron_%j.out

set -euxo pipefail

: "${REPO:=/fsx/awsome-distributed-ai}"
: "${OUT:=/fsx/validation/aws-megatron-lm.sqsh}"

MEGATRON_DIR="${REPO}/3.test_cases/megatron/megatron-lm"
[[ -f "${MEGATRON_DIR}/aws-megatron-lm.Dockerfile" ]] || {
  echo "aws-megatron-lm.Dockerfile not found under ${MEGATRON_DIR}. Clone awsome-distributed-ai to /fsx or set REPO." >&2
  exit 1
}

cd "${MEGATRON_DIR}"
docker build -t aws-megatron-lm -f aws-megatron-lm.Dockerfile .

rm -f "${OUT}"
enroot import -o "${OUT}" dockerd://aws-megatron-lm:latest
echo "Wrote ${OUT}"
