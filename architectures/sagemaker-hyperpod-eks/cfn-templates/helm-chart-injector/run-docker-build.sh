#!/bin/bash
# run-docker-build.sh

# Build the Docker image
docker build $DOCKER_NETWORK -t lambda-layer-builder .

# Run the container and copy the zip file
docker run --rm -v $(pwd)/output:/layer/output lambda-layer-builder bash -c "chmod +x build-layer.sh && ./build-layer.sh && cp lambda-layer.zip output/"

echo "Lambda layer zip file has been created in the output directory"

