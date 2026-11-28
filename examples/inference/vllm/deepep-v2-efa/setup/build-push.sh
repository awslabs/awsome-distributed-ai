#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f setup/env_vars ] || { echo "FATAL: cp setup/env_vars.example setup/env_vars and edit it first"; exit 2; }
source setup/env_vars
: "${REGISTRY:?set REGISTRY in setup/env_vars}"
IMG="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${REGISTRY}"
DOCKER_BUILDKIT=1 docker build -t "${IMG}" .
docker push "${IMG}"
echo "pushed ${IMG}"
