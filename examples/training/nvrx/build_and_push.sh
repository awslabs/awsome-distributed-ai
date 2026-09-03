#!/bin/bash
# build_and_push.sh -- Build and push the Docker image to ECR.
#
# Requires env_vars to be sourced first:
#   source env_vars
#   ./build_and_push.sh
#
# If BUILD_HOST is set in env_vars, builds happen remotely via SSH.
# Otherwise, builds happen locally (requires Docker + ECR access).

set -euo pipefail

# Verify required variables
for var in AWS_REGION ACCOUNT REGISTRY IMAGE_URI IMAGE_TAG; do
  if [ -z "${!var:-}" ]; then
    echo "ERROR: $var is not set. Run 'source env_vars' first."
    exit 1
  fi
done

REPO_NAME="nvrx-fsdp-training"
DLC_REGISTRY="763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "============================================"
echo "NVRx Docker Build & Push"
echo "============================================"
echo "Region:    ${AWS_REGION}"
echo "Account:   ${ACCOUNT}"
echo "Image:     ${IMAGE_URI}"
echo "Build host: ${BUILD_HOST:-local}"
echo "============================================"

# Create ECR repository if it doesn't exist
REPO_EXISTS=$(aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${AWS_REGION} 2>/dev/null | grep -c "${REPO_NAME}" || true)
if [ "${REPO_EXISTS}" == "0" ]; then
  echo "Creating ECR repository: ${REPO_NAME}"
  aws ecr create-repository --repository-name ${REPO_NAME} --region ${AWS_REGION}
fi

if [ -n "${BUILD_HOST:-}" ]; then
  # ---- Remote build via SSH ----
  echo ""
  echo "Building remotely on ${BUILD_HOST}..."
  BUILD_DIR="${BUILD_DIR:-~/nvrx-build}"

  # Ensure remote build directory structure exists
  ssh ${BUILD_HOST} "mkdir -p ${BUILD_DIR}/src"

  # SCP source files, Dockerfile, and requirements.txt
  echo "Copying source files to ${BUILD_HOST}:${BUILD_DIR}/..."
  scp Dockerfile requirements.txt prepare_dataset.py ${BUILD_HOST}:${BUILD_DIR}/
  scp src/*.py ${BUILD_HOST}:${BUILD_DIR}/src/

  # Build and push on remote host
  echo "Building and pushing image on ${BUILD_HOST}..."
  ssh ${BUILD_HOST} "cd ${BUILD_DIR} && \
    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${REGISTRY} && \
    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${DLC_REGISTRY} && \
    docker build --build-arg AWS_REGION=${AWS_REGION} -t ${IMAGE_URI} . && \
    docker push ${IMAGE_URI}"

else
  # ---- Local build ----
  echo ""
  echo "Building locally..."

  # ECR login
  aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${REGISTRY}
  aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${DLC_REGISTRY}

  # Build and push
  docker build --build-arg AWS_REGION=${AWS_REGION} -t ${IMAGE_URI} .
  docker push ${IMAGE_URI}
fi

echo ""
echo "============================================"
echo "Done! Image pushed: ${IMAGE_URI}"
echo "============================================"
