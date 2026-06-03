#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Build and push the QLoRA training Docker image to Amazon ECR.
#
# Usage:
#   ./0.build-image.sh
#
# Prerequisites:
#   - AWS CLI configured
#   - Docker running

set -euo pipefail

###########################
###### User Variables #####
###########################

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=${AWS_REGION:-us-east-1}
REPO_NAME=qwen3-qlora-training
IMAGE_TAG=latest

IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

###########################
###### Create ECR Repo ####
###########################

echo "Creating ECR repository (if not exists)..."
aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${AWS_REGION} 2>/dev/null || \
    aws ecr create-repository \
echo "Creating ECR repository (if not exists)..."
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${AWS_REGION}" 2>/dev/null || \
    aws ecr create-repository \
        --repository-name "${REPO_NAME}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

###########################
###### Build & Push #######
###########################

echo "Building Docker image: ${IMAGE_URI}"
docker build -t ${IMAGE_URI} .

echo "Pushing to ECR..."
docker push ${IMAGE_URI}

echo ""
echo "Image pushed successfully: ${IMAGE_URI}"
echo ""
echo "Export for use with deploy script:"
echo "  export IMAGE=${IMAGE_URI}"
