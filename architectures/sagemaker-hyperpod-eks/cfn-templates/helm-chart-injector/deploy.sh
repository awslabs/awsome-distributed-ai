#!/bin/bash
# deploy.sh

# Create a directory for the output
mkdir -p output

# Build the Lambda layer using Docker
./run-docker-build.sh

# Package the Lambda function with dependencies
./package-function.sh

