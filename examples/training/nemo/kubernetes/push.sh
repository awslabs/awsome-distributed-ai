#!/bin/bash

# Push the AWS-optimized NeMo container to Amazon ECR
# This script creates the ECR repository, logs in, tags, and pushes the image

set -e

IMAGE_NAME="aws-nemo"
TAG="25.04.01"
REGION="${AWS_REGION:-us-east-1}"  # Use AWS_REGION env var or default to us-east-1

echo "Pushing Docker image: ${IMAGE_NAME}:${TAG} to ECR"

# Get AWS Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account ID: ${ACCOUNT_ID}"
echo "Region: ${REGION}"

# Check if ECR repository exists, create if it doesn't exist
echo "Checking if ECR repository exists..."
if aws ecr describe-repositories --repository-names "${IMAGE_NAME}" --region "${REGION}" >/dev/null 2>&1; then
    echo "ECR repository '${IMAGE_NAME}' already exists"
else
    echo "Creating ECR repository '${IMAGE_NAME}'..."
    aws ecr create-repository --repository-name "${IMAGE_NAME}" --region "${REGION}"
    echo "ECR repository created successfully"
fi

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Tag the image
ECR_IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:${TAG}"
echo "Tagging image: ${IMAGE_NAME}:${TAG} -> ${ECR_IMAGE}"
docker tag "${IMAGE_NAME}:${TAG}" "${ECR_IMAGE}"

# Push the image
echo "Pushing image to ECR..."
CMD="docker push ${ECR_IMAGE}"
if [ ! "$verbose" == "false" ]; then echo -e "\n${CMD}\n"; fi
eval "$CMD"

echo "Push completed successfully!"
echo "ECR Image URI: ${ECR_IMAGE}" 
