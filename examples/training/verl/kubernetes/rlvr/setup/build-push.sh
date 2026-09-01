#!/bin/bash
set -e  # Exit on any error

echo "Building image ${REGISTRY}${IMAGE}:${TAG}"
if DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -f Dockerfile -t ${REGISTRY}${IMAGE}:${TAG} .; then
    echo "Done building image!"
    echo ""
else
    echo "Build failed!"
    exit 1
fi

echo "Pushing image to ECR..."
# Create registry if needed
REGISTRY_COUNT=$(aws ecr describe-repositories | grep "${IMAGE}" | wc -l)
if [ "$REGISTRY_COUNT" == "0" ]; then
        aws ecr create-repository --repository-name ${IMAGE}
fi

# Login to registry
echo "Logging in to $REGISTRY ..."
if aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY; then
    echo "Login successful"
else
    echo "Login failed!"
    exit 1
fi

# Push image to registry
echo "Pushing image ${REGISTRY}${IMAGE}:${TAG}"
if docker image push ${REGISTRY}${IMAGE}:${TAG}; then
    echo "Done pushing image!"
    echo ""
else
    echo "Push failed!"
    exit 1
fi