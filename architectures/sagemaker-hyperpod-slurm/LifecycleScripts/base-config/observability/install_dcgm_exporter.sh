#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Define the container name and image version
CONTAINER_NAME="dcgm-exporter"
ECR_ACCOUNT_ID=602401143452
DCGM_EXPORTER_VERSION=4.1.1-4.0.4-ubi9
IMAGE="$ECR_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/hyperpod/dcgm_exporter:${DCGM_EXPORTER_VERSION}"
METRICS_CSV_DIR="/etc/dcgm-exporter"
METRICS_CSV_FILE="/etc/dcgm-exporter/dcgm-metrics.csv"
JOB_ID_MAP_DIR="/run/slurm/dcgm_job_mapping"

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

# Check for GPU, then proceed with script
if nvidia-smi > /dev/null 2>&1; then
    echo "NVIDIA GPU found. Proceeding with script..."

    # Get the instance-type from EC2 instance metadata
    TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
    INSTANCE_TYPE=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/instance-type)

    echo "Instance Type is recognized as $INSTANCE_TYPE, setting DCGM_EXPORTER_VERSION to $DCGM_EXPORTER_VERSION"

    # Retry logic for pulling the image
    attempt=0
    delay=$RETRY_DELAY
    while [ $attempt -lt $MAX_RETRIES ]; do
        echo "Attempting to pull image ($((attempt + 1))/$MAX_RETRIES)..."

        aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

        if sudo docker pull "$IMAGE"; then
            echo "Successfully pulled image."
            break
        else
            attempt=$((attempt + 1))
            if [ $attempt -lt $MAX_RETRIES ]; then
                echo "Pull failed. Retrying in $delay seconds..."
                sleep $delay
                delay=$((delay * 2))  # Exponential backoff
            else
                echo "Failed to pull Docker image after $MAX_RETRIES attempts. Exiting..."
                exit 1
            fi
        fi
    done

    # Run the DCGM Exporter Docker container with the custom metrics
    if sudo docker run -d --restart always \
        --name $CONTAINER_NAME \
        --gpus all \
        --net host \
        --cap-add SYS_ADMIN \
        -v "$METRICS_CSV_DIR:$METRICS_CSV_DIR" \
        -v "$JOB_ID_MAP_DIR:/etc/dcgm-exporter/hpc" \
        $IMAGE \
        -f "$METRICS_CSV_FILE" \
        --hpc-job-mapping-dir "/etc/dcgm-exporter/hpc"; then
        echo "Running DCGM exporter in a Docker container on port 9400..."
    else
        echo "Failed to run DCGM Exporter Docker container"
        exit 1
    fi
else
    echo "NVIDIA GPU not found. DCGM Exporter was not installed. If this is a controller node, you can safely ignore this warning. Exiting gracefully..."
    exit 0
fi
