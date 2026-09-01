#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Build the SGLang + DeepEP-on-EFA image, and optionally push it to ECR.
#
# No GPU is needed at build time (EFA installs with --skip-kmod and the libcuda
# stub is symlinked for linking), but the build compiles NVSHMEM, DeepEP and
# Mooncake from source — budget ~40-60 min on a 32-core box, and run it on an
# instance with plenty of cores.
#
# Reads IMAGE_URI (and REGISTRY / IMAGE / TAG / AWS_REGION for --push) from
# setup/env_vars.
#
# Usage:
#   source setup/env_vars
#   setup/build-push.sh              # build locally, tag as $IMAGE_URI
#   setup/build-push.sh --push       # also tag ${REGISTRY}${IMAGE}:${TAG} and push

set -euo pipefail

PUSH=false
[[ "${1:-}" == "--push" ]] && PUSH=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${IMAGE_URI:?source setup/env_vars first}"

echo "==> Building ${IMAGE_URI}"
DOCKER_BUILDKIT=1 docker build \
    --progress=plain \
    --platform linux/amd64 \
    -f "${PROJECT_DIR}/Dockerfile" \
    -t "${IMAGE_URI}" \
    "${PROJECT_DIR}"

echo "==> Built ${IMAGE_URI}"

if [[ "$PUSH" == "true" ]]; then
    : "${REGISTRY:?need REGISTRY}"; : "${IMAGE:?need IMAGE}"
    : "${TAG:?need TAG}"; : "${AWS_REGION:?need AWS_REGION}"
    FULL_IMAGE="${REGISTRY}${IMAGE}:${TAG}"

    echo "==> Ensuring ECR repository ${IMAGE} exists"
    if ! aws ecr describe-repositories --repository-names "${IMAGE}" \
            --region "${AWS_REGION}" >/dev/null 2>&1; then
        aws ecr create-repository --repository-name "${IMAGE}" \
            --region "${AWS_REGION}" >/dev/null
    fi

    echo "==> Logging in to ${REGISTRY}"
    aws ecr get-login-password --region "${AWS_REGION}" \
        | docker login --username AWS --password-stdin "${REGISTRY%/}"

    echo "==> Pushing ${FULL_IMAGE}"
    docker tag "${IMAGE_URI}" "${FULL_IMAGE}"
    docker push "${FULL_IMAGE}"

    echo "==> Done:"
    docker inspect --format='{{index .RepoDigests 0}}' "${FULL_IMAGE}" || true
    echo
    echo "Pull this on every node, then set IMAGE_URI=${FULL_IMAGE} in setup/env_vars."
fi
