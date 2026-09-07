#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f setup/env_vars ] || { echo "FATAL: cp setup/env_vars.example setup/env_vars and edit it first"; exit 2; }
source setup/env_vars
: "${REGISTRY:?set REGISTRY in setup/env_vars}"
: "${IMAGE_NAME:?set IMAGE_NAME in setup/env_vars}"
: "${IMAGE_TAG:?set IMAGE_TAG in setup/env_vars}"
: "${AWS_REGION:?set AWS_REGION in setup/env_vars}"
IMG="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${REGISTRY}"
# --platform: the image targets x86 p5en nodes; on an arm64 workstation an unplatformed build
# fails four layers in (Layer-5 manylinux x86_64 wheel), this fails at Layer 1 with the cause named.
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -t "${IMG}" .
docker push "${IMG}"
echo "pushed ${IMG}"
