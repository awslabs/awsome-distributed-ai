#!/usr/bin/env bash
set -euo pipefail

# Build and push MTC-enabled Docker image
# This script builds an image with the amzn-sagemaker-checkpointing library

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../setup/env_vars"

# Configuration
MTC_TAG="${TAG}-mtc"
DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile"

echo "=================================="
echo "Building MTC-enabled Docker Image"
echo "=================================="
echo "Base Image: ${REGISTRY}${IMAGE}:${TAG}"
echo "Target Image: ${REGISTRY}${IMAGE}:${MTC_TAG}"
echo "Dockerfile: ${DOCKERFILE_PATH}"
echo ""

# Build the image
echo "Building image..."
if DOCKER_BUILDKIT=1 docker build \
    --platform linux/amd64 \
    --build-arg REGISTRY="${REGISTRY}" \
    --build-arg IMAGE="${IMAGE}" \
    --build-arg TAG="${TAG}" \
    -f "${DOCKERFILE_PATH}" \
    -t "${REGISTRY}${IMAGE}:${MTC_TAG}" \
    "${SCRIPT_DIR}/.."; then
    echo ""
    echo "✓ Build successful!"
    echo ""
else
    echo ""
    echo "✗ Build failed!"
    exit 1
fi

# Create ECR repository if it doesn't exist
echo "Checking ECR repository..."
REGISTRY_COUNT=$(aws ecr describe-repositories --region "${AWS_REGION}" 2>/dev/null | grep -c "\"${IMAGE}\"" || true)
if [ "${REGISTRY_COUNT}" == "0" ]; then
    echo "Creating ECR repository: ${IMAGE}"
    aws ecr create-repository --repository-name "${IMAGE}" --region "${AWS_REGION}"
else
    echo "✓ Repository already exists"
fi

# Login to ECR
echo ""
echo "Logging in to ECR..."
if aws ecr get-login-password --region "${AWS_REGION}" | \
   docker login --username AWS --password-stdin "${REGISTRY}"; then
    echo "✓ Login successful"
else
    echo "✗ Login failed!"
    exit 1
fi

# Push the image
echo ""
echo "Pushing image to ECR..."
echo "Image: ${REGISTRY}${IMAGE}:${MTC_TAG}"
if docker image push "${REGISTRY}${IMAGE}:${MTC_TAG}"; then
    echo ""
    echo "✓ Push successful!"
    echo ""
    echo "=================================="
    echo "MTC-enabled image ready:"
    echo "  ${REGISTRY}${IMAGE}:${MTC_TAG}"
    echo "=================================="
    echo ""
else
    echo ""
    echo "✗ Push failed!"
    exit 1
fi
