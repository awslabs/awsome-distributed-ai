#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Build and push Docker image for Qwen GRPO experiments
# Usage: ./build_push.sh [--profile compute-sa] [--region us-east-2]

set -euo pipefail

# Configuration
AWS_PROFILE="${AWS_PROFILE:-compute-sa}"
AWS_REGION="${AWS_REGION:-us-east-2}"
IMAGE_NAME="${IMAGE_NAME:-qwen-grpo-experiments}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --profile "${AWS_PROFILE}" --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_IMAGE="${ECR_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "============================================"
echo "Building Qwen GRPO Experiments Image"
echo "============================================"
echo "Profile:  ${AWS_PROFILE}"
echo "Region:   ${AWS_REGION}"
echo "Registry: ${ECR_REGISTRY}"
echo "Image:    ${FULL_IMAGE}"
echo "============================================"

# Create ECR repository if it doesn't exist
aws ecr describe-repositories \
    --profile "${AWS_PROFILE}" \
    --region "${AWS_REGION}" \
    --repository-names "${IMAGE_NAME}" 2>/dev/null || \
aws ecr create-repository \
    --profile "${AWS_PROFILE}" \
    --region "${AWS_REGION}" \
    --repository-name "${IMAGE_NAME}" \
    --image-scanning-configuration scanOnPush=true

# Login to ECR
aws ecr get-login-password \
    --profile "${AWS_PROFILE}" \
    --region "${AWS_REGION}" | \
docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Build image (linux/amd64 for EKS)
echo "Building Docker image..."
docker buildx build \
    --platform linux/amd64 \
    --tag "${FULL_IMAGE}" \
    --push \
    .

echo "============================================"
echo "Image pushed: ${FULL_IMAGE}"
echo "============================================"
