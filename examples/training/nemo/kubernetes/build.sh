#!/bin/bash

# Build the AWS-optimized NeMo container for P4 and P5 instances
# This script builds the Docker image with EFA support optimizations

set -e

IMAGE_NAME="aws-nemo"
TAG="25.04.01"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "This may take several minutes..."

CMD="docker build --progress=plain -t "${IMAGE_NAME}:${TAG}" -f Dockerfile ."
if [ ! "$verbose" == "false" ]; then echo -e "\n${CMD}\n"; fi
eval "${CMD}"

echo "Build completed successfully!"
echo "Image built: ${IMAGE_NAME}:${TAG}" 