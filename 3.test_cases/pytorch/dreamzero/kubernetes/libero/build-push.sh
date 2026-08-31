#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Build the DreamZero training image (two-stage: upstream RLinf embodied-libero +
# EFA overlay) and push to your ECR. Requires docker buildx + AWS CLI logged in.
#
# Usage:
#   source ./env_vars         # sets ECR_URI, AWS_REGION, UPSTREAM_REF, DREAMZERO_REF
#   ./build-push.sh
set -euo pipefail

: "${ECR_URI:?set ECR_URI (e.g. <acct>.dkr.ecr.<region>.amazonaws.com/dreamzero)}"
: "${AWS_REGION:?set AWS_REGION}"
UPSTREAM_REPO="${UPSTREAM_REPO:-https://github.com/RLinf/RLinf.git}"
UPSTREAM_REF="${UPSTREAM_REF:-b3bbabb1f461}"
DREAMZERO_REPO="${DREAMZERO_REPO:-https://github.com/RLinf/dreamzero.git}"
DREAMZERO_REF="${DREAMZERO_REF:-ab790c198fbc}"
BUILD_TARGET="${BUILD_TARGET:-embodied-libero}"
# Immutable image tag. Defaults to one derived from the pinned DreamZero ref so the
# pushed image is reproducible (CONTRIBUTING: "do not use a latest tag"). The SAME
# value must be exported as IMAGE_TAG when rendering the manifests so the deployed
# image matches the pushed one -- see env_vars.example.
IMAGE_TAG="${IMAGE_TAG:-dz-${DREAMZERO_REF}}"

# Test-case root (this script is at kubernetes/libero/build-push.sh).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "== ECR login =="
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "${ECR_URI%%/*}"

echo "== Clone pinned sources into the build context =="
rm -rf RLinf DreamZero
git clone "$UPSTREAM_REPO" RLinf && git -C RLinf checkout "$UPSTREAM_REF"
git clone "$DREAMZERO_REPO" DreamZero && git -C DreamZero checkout "$DREAMZERO_REF"

echo "== Stage 1: upstream embodied-libero =="
docker buildx build --platform linux/amd64 --load \
  --build-arg BUILD_TARGET="$BUILD_TARGET" --build-arg NO_MIRROR=1 \
  -t "rlinf-upstream-${BUILD_TARGET}" \
  -f RLinf/docker/Dockerfile RLinf

echo "== Stage 2: EFA overlay (push) =="
docker buildx build --platform linux/amd64 --push \
  --build-arg BUILD_TARGET="$BUILD_TARGET" \
  -t "${ECR_URI}:${IMAGE_TAG}" \
  -f Dockerfile .

echo "== Done: ${ECR_URI}:${IMAGE_TAG} =="
echo "== Cleaning transient clones =="
rm -rf RLinf DreamZero
