#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Build and Push Docker Image to ECR
# =============================================================================
set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/../env_vars" ]; then
    source "${SCRIPT_DIR}/../env_vars"
fi

# Configuration
AWS_REGION=${AWS_REGION:-$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')}
ACCOUNT=${ACCOUNT:-$(aws sts get-caller-identity --query Account --output text)}
REGISTRY=${REGISTRY:-"${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/"}
IMAGE=${IMAGE:-"verl-grpo-lora"}
TAG=${TAG:-"latest"}

FULL_IMAGE="${REGISTRY}${IMAGE}:${TAG}"

echo "=============================================="
echo "Building Docker Image"
echo "=============================================="
echo "Registry: ${REGISTRY}"
echo "Image: ${IMAGE}:${TAG}"
echo "Full path: ${FULL_IMAGE}"
echo "=============================================="

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names "${IMAGE}" --region "${AWS_REGION}" 2>/dev/null || \
    aws ecr create-repository --repository-name "${IMAGE}" --region "${AWS_REGION}"

# Login to ECR
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${REGISTRY}"

# Build the image
cd "${SCRIPT_DIR}/"
docker build \
    --platform linux/amd64 \
    -t "${FULL_IMAGE}" \
    -f Dockerfile \
    .

# Push to ECR
docker push "${FULL_IMAGE}"

echo ""
echo "=============================================="
echo "Image pushed successfully!"
echo "=============================================="
echo "Image: ${FULL_IMAGE}"
echo ""
echo "To use this image, ensure your env_vars has:"
echo "  export REGISTRY=${REGISTRY}"
echo "  export IMAGE=${IMAGE}"
echo "  export TAG=${TAG}"