#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Define the container name
CONTAINER_NAME="otel-collector"

ECR_ACCOUNT_ID=602401143452
VERSION=v1754424030352
IMAGE="$ECR_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/hyperpod/otel_collector:${VERSION}"

# Maximum number of retries
MAX_RETRIES=5
RETRY_DELAY=5  # Initial delay in seconds

# Check if the container exists and is running
if docker ps --filter "name=$CONTAINER_NAME" --filter "status=running" | grep -q "$CONTAINER_NAME"; then
    echo "Container $CONTAINER_NAME is already running."
    exit 0
else
    echo "Container $CONTAINER_NAME is not running or does not exist..."
    echo "Checking if $CONTAINER_NAME container exists but is not running. If yes, removing it..."
    docker rm -f $CONTAINER_NAME && echo "Container $CONTAINER_NAME has been removed."
    echo "Proceeding with script..."
fi

# Retry logic for pulling the image
attempt=0
while [ $attempt -lt $MAX_RETRIES ]; do
    echo "Attempting to pull image ($attempt/$MAX_RETRIES)..."

    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

    if sudo docker pull "$IMAGE"; then
        echo "Successfully pulled image."
        break
    else
        attempt=$((attempt + 1))
        if [ $attempt -lt $MAX_RETRIES ]; then
            echo "Pull failed. Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
            RETRY_DELAY=$((RETRY_DELAY * 2))  # Exponential backoff
        else
            echo "Failed to pull Docker image after $MAX_RETRIES attempts. Exiting..."
            exit 1
        fi
    fi
done

# Run the Docker container with appropriate configurations
if sudo docker run -d --restart always \
    --name=$CONTAINER_NAME \
    --net="host" \
    --pid="host" \
    -v /etc/otel:/etc/otel \
    -p 4317:4317 -p 4318:4318 -p 8889:8889 \
    $IMAGE \
    --config=/etc/otel/config.yaml ; then
    echo "Successfully started OTEL Collector on node"
    exit 0
else
    echo "Failed to run Docker container"
    exit 1
fi
