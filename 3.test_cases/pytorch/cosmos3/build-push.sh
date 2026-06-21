#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Build the Cosmos 3 post-training image (AWS DLC base) and push to
# your Amazon ECR. Mirrors the build-push convention used elsewhere in
# awsome-distributed-ai (e.g. pytorch/verl).
#
# The Dockerfile is self-contained: the base comes from the public DLC ECR, the
# framework is cloned + `uv sync`ed, and FFmpeg/uv are fetched at build time. The
# only local build context is src/ and toml/ (COPYed in). No FSx staging needed.
#
# BUILD HOST: this image is x86_64-only (the FFmpeg shared build is linux64 and
# the CUDA wheels — flash-attn, transformer-engine — ship amd64 wheels only), so
# it targets linux/amd64. On an Apple-Silicon / ARM host the build runs under
# QEMU emulation and is impractically slow for a venv this large; build on an
# x86_64 host (Linux box, CI runner, or an in-region EC2 instance near ECR).
#
# Usage:
#   cp env_vars.example env_vars   # then edit env_vars (registry, tag, DLC base)
#   set -a; . ./env_vars; set +a
#   ./build-push.sh
set -euo pipefail

# Required (from env_vars):
: "${IMAGE_URI:?set IMAGE_URI (e.g. <acct>.dkr.ecr.<region>.amazonaws.com/cosmos3:dlc)}"
: "${IMAGE:?set IMAGE (ECR repository name, e.g. cosmos3)}"
: "${AWS_REGION:?set AWS_REGION}"
: "${REGISTRY:?set REGISTRY (e.g. <acct>.dkr.ecr.<region>.amazonaws.com)}"

# Pins (validated; override via env_vars or the environment if you must):
COSMOS_FRAMEWORK_REF="${COSMOS_FRAMEWORK_REF:-f358a625875310096cf01d75b11b7818c438aa1e}"
DLC_TAG="${DLC_TAG:-2.10.0-gpu-py313-cu130-ubuntu22.04-ec2}"

# This script lives at the cosmos3 test-case root; build context = that dir.
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "== ECR login =="
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "$REGISTRY"

echo "== Ensure ECR repository '${IMAGE}' exists =="
aws ecr describe-repositories --repository-names "$IMAGE" --region "$AWS_REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$IMAGE" --region "$AWS_REGION" >/dev/null

echo "== Build + push ${IMAGE_URI} (linux/amd64) =="
echo "   COSMOS_FRAMEWORK_REF=${COSMOS_FRAMEWORK_REF}  DLC_TAG=${DLC_TAG}"
docker buildx build --platform linux/amd64 --push \
  -f Dockerfile \
  --build-arg COSMOS_FRAMEWORK_REF="$COSMOS_FRAMEWORK_REF" \
  --build-arg DLC_TAG="$DLC_TAG" \
  -t "$IMAGE_URI" \
  .

echo "== Done: ${IMAGE_URI} =="
