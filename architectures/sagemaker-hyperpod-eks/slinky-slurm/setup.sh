#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -euo pipefail

###########################
###### Default Values #####
###########################

REPO_NAME="dlc-slurmd"
IMAGE_TAG="25.11.1-ubuntu24.04"
AWS_REGION=$(aws configure get region 2>/dev/null || echo "us-west-2")
AWS_ACCOUNT_ID=""
INSTANCE_TYPE=""
INSTANCE_COUNT=4
INFRA=""
SKIP_BUILD=false
LOCAL_BUILD=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBUILD_STACK_NAME="slurmd-codebuild-stack"
S3_BUCKET=""

###########################
## Source Helper Library ###
###########################

source "${SCRIPT_DIR}/lib/deploy_helpers.sh"

###########################
###### Usage Function #####
###########################

usage() {
    cat <<EOF
Usage: $0 --instance-type <ml.X.Y> --infra <cfn|tf> [OPTIONS]

Build the Slurmd DLC container image, generate SSH keys, and produce
slurm-values.yaml from the template.

Required:
  --instance-type <type>    SageMaker instance type for GPU/EFA/GRES resolution
                            (e.g. ml.g5.8xlarge, ml.p5.48xlarge, ml.g6.12xlarge)
  --infra <cfn|tf>          Infrastructure method for CodeBuild stack

Optional:
  --instance-count <N>      Number of accelerated instances / Slurm worker
                            replicas (default: 4)
  --repo-name <name>        ECR repository name (default: dlc-slurmd)
  --tag <tag>               Image tag (default: 25.11.1-ubuntu24.04)
  --region <region>         AWS region (default: AWS CLI configured or us-west-2)
  --local-build             Build image locally instead of CodeBuild
  --skip-build              Skip image build (use existing image in ECR)
  --help                    Show this help message

Examples:
  # Build via CodeBuild for ml.g5.8xlarge instances
  $0 --instance-type ml.g5.8xlarge --infra cfn

  # Build locally for ml.p5.48xlarge instances (2 nodes)
  $0 --instance-type ml.p5.48xlarge --instance-count 2 --infra tf --local-build

  # Skip build (image already in ECR)
  $0 --instance-type ml.g5.8xlarge --infra cfn --skip-build
EOF
    exit 0
}

###########################
###### Parse Arguments ####
###########################

while [[ $# -gt 0 ]]; do
    case $1 in
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --instance-count)
            INSTANCE_COUNT="$2"
            shift 2
            ;;
        --infra)
            INFRA="$2"
            shift 2
            ;;
        --repo-name)
            REPO_NAME="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --local-build)
            LOCAL_BUILD=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

###########################
### Validate Arguments ####
###########################

if [[ -z "${INSTANCE_TYPE}" ]]; then
    echo "Error: --instance-type is required (e.g. ml.g5.8xlarge)"
    exit 1
fi

if [[ -z "${INFRA}" ]]; then
    echo "Error: --infra is required (cfn or tf)"
    exit 1
fi

if [[ "${INFRA}" != "cfn" && "${INFRA}" != "tf" ]]; then
    echo "Error: --infra must be 'cfn' or 'tf' (got: ${INFRA})"
    exit 1
fi

# Resolve Helm profile variables (validates type via EC2 API, sets
# GPU_COUNT, EFA_COUNT, GPU_GRES, REPLICAS, etc.)
resolve_helm_profile "${INSTANCE_TYPE}" "${INSTANCE_COUNT}"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_REPOSITORY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}"

echo ""
echo "=========================================="
echo "  Slurmd DLC Setup"
echo "=========================================="
echo ""
echo "  Instance type: ${INSTANCE_TYPE}"
echo "  Instance count: ${INSTANCE_COUNT}"
echo "  Infrastructure: ${INFRA}"
echo "  Region: ${AWS_REGION}"
echo "  Account: ${AWS_ACCOUNT_ID}"
echo "  Image: ${IMAGE_REPOSITORY}:${IMAGE_TAG}"
echo "  GPU count: ${GPU_COUNT}"
echo "  EFA count: ${EFA_COUNT}"
echo "  GRES: ${GPU_GRES}"
echo "  Replicas: ${REPLICAS}"
echo ""

###########################
## Image Build ############
###########################

if [[ "${SKIP_BUILD}" == "true" ]]; then
    echo "Skipping image build..."
    echo ""

    # Verify image exists in ECR
    if ! aws ecr describe-images \
        --repository-name "${REPO_NAME}" \
        --image-ids imageTag="${IMAGE_TAG}" \
        --region "${AWS_REGION}" &>/dev/null; then
        echo "Error: Image ${IMAGE_REPOSITORY}:${IMAGE_TAG} not found in ECR."
        echo "  Run without --skip-build to build and push the image first."
        exit 1
    fi

    echo "  Found existing image: ${IMAGE_REPOSITORY}:${IMAGE_TAG}"

elif [[ "${LOCAL_BUILD}" == "true" ]]; then
    echo "Building image locally..."
    echo ""

    # Authenticate to DLC ECR registry
    echo "  Authenticating to DLC ECR registry..."
    aws ecr get-login-password --region us-east-1 | \
        docker login --username AWS \
        --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

    # Build the image (platform-aware)
    if [[ "${OSTYPE}" == "darwin"* ]]; then
        echo "  Building on macOS (linux/amd64)..."
        docker buildx build --platform linux/amd64 \
            -t "${REPO_NAME}:${IMAGE_TAG}" \
            -f dlc-slurmd.Dockerfile .
    else
        echo "  Building on Linux..."
        docker build \
            -t "${REPO_NAME}:${IMAGE_TAG}" \
            -f dlc-slurmd.Dockerfile .
    fi

    # Create ECR repo if needed
    echo "  Creating ECR repository (if not exists)..."
    aws ecr create-repository \
        --no-cli-pager \
        --repository-name "${REPO_NAME}" \
        --region "${AWS_REGION}" 2>/dev/null || echo "  Repository already exists."

    # Authenticate to project ECR
    echo "  Authenticating to project ECR..."
    aws ecr get-login-password --region "${AWS_REGION}" | \
        docker login --username AWS \
        --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

    # Tag and push
    echo "  Tagging and pushing image..."
    docker tag "${REPO_NAME}:${IMAGE_TAG}" "${IMAGE_REPOSITORY}:${IMAGE_TAG}"
    docker push "${IMAGE_REPOSITORY}:${IMAGE_TAG}"

    echo "  Image pushed: ${IMAGE_REPOSITORY}:${IMAGE_TAG}"

else
    echo "Building image via CodeBuild..."
    echo ""

    # Determine S3 bucket for build context
    # Use the S3 bucket from the HyperPod stack outputs if available
    if [[ -f "${SCRIPT_DIR}/env_vars.sh" ]]; then
        source "${SCRIPT_DIR}/env_vars.sh"
    fi

    # Create a temporary S3 bucket for build context if needed
    S3_BUCKET="${S3_BUCKET:-${REPO_NAME}-codebuild-${AWS_ACCOUNT_ID}-${AWS_REGION}}"

    aws s3 mb "s3://${S3_BUCKET}" --region "${AWS_REGION}" 2>/dev/null || true

    # Package and upload build context
    echo "  Packaging build context..."
    local_tmp=$(mktemp -d)
    cp "${SCRIPT_DIR}/dlc-slurmd.Dockerfile" "${local_tmp}/"
    cp "${SCRIPT_DIR}/buildspec.yml" "${local_tmp}/"
    (cd "${local_tmp}" && zip -r build-context.zip .)

    echo "  Uploading to s3://${S3_BUCKET}/codebuild/slurmd-build-context.zip..."
    aws s3 cp "${local_tmp}/build-context.zip" \
        "s3://${S3_BUCKET}/codebuild/slurmd-build-context.zip" \
        --region "${AWS_REGION}"
    rm -rf "${local_tmp}"

    # Deploy CodeBuild stack if not present
    if [[ "${INFRA}" == "cfn" ]]; then
        if ! aws cloudformation describe-stacks \
            --stack-name "${CODEBUILD_STACK_NAME}" \
            --region "${AWS_REGION}" &>/dev/null; then

            # Check if ECR repository already exists
            create_ecr="true"
            if aws ecr describe-repositories \
                --repository-names "${REPO_NAME}" \
                --region "${AWS_REGION}" &>/dev/null; then
                create_ecr="false"
                echo "  ECR repository '${REPO_NAME}' already exists (skipping creation)."
            fi

            echo "  Deploying CodeBuild stack (CloudFormation)..."
            aws cloudformation create-stack \
                --region "${AWS_REGION}" \
                --stack-name "${CODEBUILD_STACK_NAME}" \
                --template-body "file://${SCRIPT_DIR}/codebuild-stack.yaml" \
                --parameters \
                    "ParameterKey=RepositoryName,ParameterValue=${REPO_NAME}" \
                    "ParameterKey=ImageTag,ParameterValue=${IMAGE_TAG}" \
                    "ParameterKey=SourceS3Bucket,ParameterValue=${S3_BUCKET}" \
                    "ParameterKey=CreateECRRepository,ParameterValue=${create_ecr}" \
                --capabilities CAPABILITY_NAMED_IAM

            echo "  Waiting for CodeBuild stack..."
            aws cloudformation wait stack-create-complete \
                --stack-name "${CODEBUILD_STACK_NAME}" \
                --region "${AWS_REGION}"
            echo "  CodeBuild stack deployed."
        else
            echo "  CodeBuild stack already exists."
        fi

        CODEBUILD_PROJECT=$(aws cloudformation describe-stacks \
            --stack-name "${CODEBUILD_STACK_NAME}" \
            --region "${AWS_REGION}" \
            --query "Stacks[0].Outputs[?OutputKey=='CodeBuildProjectName'].OutputValue" \
            --output text)
    else
        # Terraform path: init and apply codebuild.tf
        echo "  Deploying CodeBuild resources (Terraform)..."
        cb_dir="${SCRIPT_DIR}/codebuild-tf"
        mkdir -p "${cb_dir}"
        cp "${SCRIPT_DIR}/codebuild.tf" "${cb_dir}/main.tf"

        # Check if ECR repository already exists
        create_ecr="true"
        if aws ecr describe-repositories \
            --repository-names "${REPO_NAME}" \
            --region "${AWS_REGION}" &>/dev/null; then
            create_ecr="false"
            echo "  ECR repository '${REPO_NAME}' already exists (skipping creation)."
        fi

        terraform -chdir="${cb_dir}" init -input=false
        terraform -chdir="${cb_dir}" apply -auto-approve \
            -var="source_s3_bucket=${S3_BUCKET}" \
            -var="repository_name=${REPO_NAME}" \
            -var="image_tag=${IMAGE_TAG}" \
            -var="create_ecr_repository=${create_ecr}"

        CODEBUILD_PROJECT=$(terraform -chdir="${cb_dir}" output -raw codebuild_project_name)
    fi

    # Trigger build
    echo "  Starting CodeBuild build: ${CODEBUILD_PROJECT}..."
    BUILD_ID=$(aws codebuild start-build \
        --project-name "${CODEBUILD_PROJECT}" \
        --region "${AWS_REGION}" \
        --environment-variables-override \
            "name=IMAGE_TAG,value=${IMAGE_TAG},type=PLAINTEXT" \
        --query 'build.id' --output text)

    echo "  Build ID: ${BUILD_ID}"
    echo "  Waiting for build to complete..."

    # Poll build status
    while true; do
        BUILD_STATUS=$(aws codebuild batch-get-builds \
            --ids "${BUILD_ID}" \
            --region "${AWS_REGION}" \
            --query 'builds[0].buildStatus' --output text)

        case "${BUILD_STATUS}" in
            SUCCEEDED)
                echo "  Build completed successfully."
                break
                ;;
            FAILED|FAULT|STOPPED|TIMED_OUT)
                echo "Error: Build ${BUILD_STATUS}."
                echo "  View logs: aws codebuild batch-get-builds --ids ${BUILD_ID}"
                exit 1
                ;;
            *)
                echo "  Build status: ${BUILD_STATUS}..."
                sleep 15
                ;;
        esac
    done
fi

###########################
## SSH Key Generation #####
###########################

echo ""
echo "Checking SSH keys..."

if [[ ! -f ~/.ssh/id_ed25519_slurm ]]; then
    echo "  Generating SSH key pair for Slurm login access..."
    ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_slurm -C "slurm-login" -N ""
else
    echo "  SSH key already exists: ~/.ssh/id_ed25519_slurm"
fi

SSH_KEY="$(cat ~/.ssh/id_ed25519_slurm.pub)"
echo "  Public key: ${SSH_KEY}"

###########################
## Query FSx Filesystem ###
###########################

echo ""
echo "Querying FSx for Lustre filesystem..."

# Extract ResourceNamePrefix from the appropriate config file.
if [[ "${INFRA}" == "cfn" ]]; then
    RESOURCE_NAME_PREFIX=$(jq -r \
        '.[] | select(.ParameterKey == "ResourceNamePrefix") | .ParameterValue' \
        "${SCRIPT_DIR}/params.json")
else
    RESOURCE_NAME_PREFIX=$(grep '^resource_name_prefix' "${SCRIPT_DIR}/custom.tfvars" \
        | sed 's/.*=[[:space:]]*"\(.*\)"/\1/')
fi
if [[ "${INFRA}" == "cfn" ]]; then
    FSX_TAG_NAME="${RESOURCE_NAME_PREFIX}-fsx-lustre"
else
    FSX_TAG_NAME="${RESOURCE_NAME_PREFIX}-fsx"
fi

echo "  Looking for FSx filesystem with Name tag: ${FSX_TAG_NAME}"

FSX_DESCRIBE=$(aws fsx describe-file-systems \
    --region "${AWS_REGION}" \
    --query "FileSystems[?Tags[?Key=='Name'&&Value=='${FSX_TAG_NAME}']]" \
    --output json)

FSX_ID=$(echo "${FSX_DESCRIBE}" | jq -r '.[0].FileSystemId // empty')
if [[ -z "${FSX_ID}" ]]; then
    echo "Error: No FSx filesystem found with tag Name=${FSX_TAG_NAME}"
    echo "  Verify the filesystem exists in region ${AWS_REGION}."
    exit 1
fi

FSX_DNS="${FSX_ID}.fsx.${AWS_REGION}.amazonaws.com"
FSX_MOUNT=$(echo "${FSX_DESCRIBE}" | jq -r '.[0].LustreConfiguration.MountName // empty')
if [[ -z "${FSX_MOUNT}" ]]; then
    echo "Error: Could not determine FSx mount name for ${FSX_ID}."
    exit 1
fi

echo "  FSx ID: ${FSX_ID}"
echo "  FSx DNS: ${FSX_DNS}"
echo "  FSx mount: ${FSX_MOUNT}"

###########################
## Generate Lustre PVC ####
###########################

echo ""
echo "Generating lustre-pvc-slurm.yaml from template..."

sed -e "s|\${fsx_id}|${FSX_ID}|g" \
    -e "s|\${fsx_dns}|${FSX_DNS}|g" \
    -e "s|\${fsx_mount}|${FSX_MOUNT}|g" \
    "${SCRIPT_DIR}/lustre-pvc-slurm.yaml.template" > "${SCRIPT_DIR}/lustre-pvc-slurm.yaml"

echo "  Written to: ${SCRIPT_DIR}/lustre-pvc-slurm.yaml"

###########################
## Generate Values File ###
###########################

echo ""
echo "Generating slurm-values.yaml from template..."

sed -e "s|\${image_repository}|${IMAGE_REPOSITORY}|g" \
    -e "s|\${image_tag}|${IMAGE_TAG}|g" \
    -e "s|\${ssh_key}|${SSH_KEY}|g" \
    -e "s|\${mgmt_instance_type}|${MGMT_INSTANCE_TYPE}|g" \
    -e "s|\${accel_instance_type}|${HELM_ACCEL_INSTANCE_TYPE}|g" \
    -e "s|\${gpu_count}|${GPU_COUNT}|g" \
    -e "s|\${efa_count}|${EFA_COUNT}|g" \
    -e "s|\${gpu_gres}|${GPU_GRES}|g" \
    -e "s|\${replicas}|${REPLICAS}|g" \
    -e "s|\${pvc_name}|${PVC_NAME}|g" \
    "${SCRIPT_DIR}/slurm-values.yaml.template" > "${SCRIPT_DIR}/slurm-values.yaml"

echo "  Written to: ${SCRIPT_DIR}/slurm-values.yaml"

echo ""
echo "=========================================="
echo "  Setup Complete"
echo "=========================================="
echo ""
echo "  Image: ${IMAGE_REPOSITORY}:${IMAGE_TAG}"
echo "  Values: ${SCRIPT_DIR}/slurm-values.yaml"
echo "  SSH key: ~/.ssh/id_ed25519_slurm"
echo ""
echo "Next steps:"
echo "  1. Review slurm-values.yaml"
echo "  2. Run install.sh to deploy the Slurm cluster"
echo ""
