#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Define the container name
CONTAINER_NAME="node-exporter"

ECR_ACCOUNT_ID=602401143452
VERSION=v1.9.1
IMAGE="$ECR_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/hyperpod/node_exporter:${VERSION}"

# Maximum number of retries
MAX_RETRIES=5
RETRY_DELAY=5  # Initial delay in seconds

# Set additional flags for advanced metrics if ADVANCED is set to 1
ADDITIONAL_FLAGS=""
if [ "$ADVANCED" = "1" ]; then
    ADDITIONAL_FLAGS="--collector.cgroups --collector.ksmd --collector.meminfo_numa --collector.ethtool --collector.mountstats --collector.network_route --collector.processes --collector.tcpstat"
fi

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

# Add textfile collector for NCCL metrics if the inspector directory exists
NCCL_VOLUME=""
NCCL_FLAGS=""
if [ -d "/var/lib/node_exporter/nccl_inspector" ]; then
    NCCL_VOLUME="-v /var/lib/node_exporter:/var/lib/node_exporter:ro"
    NCCL_FLAGS="--collector.textfile --collector.textfile.directory=/var/lib/node_exporter/nccl_inspector"
fi

# Run the Docker container with appropriate configurations
if sudo docker run -d --restart always \
    --name=$CONTAINER_NAME \
    --net="host" \
    --pid="host" \
    -v "/:/host:ro,rslave" \
    $NCCL_VOLUME \
    $IMAGE \
    --path.rootfs=/host \
    $ADDITIONAL_FLAGS \
    $NCCL_FLAGS; then
    echo "Successfully started Node Exporter on node"
    exit 0
else
    echo "Failed to run Docker container"
    exit 1
fi
