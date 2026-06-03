#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# : "${STACK_ID:-hyperpod-eks-full-stack}"

# Clear previously set env_vars 
> env_vars 

# Define AWS Region
if [ -z ${AWS_REGION} ]; then
    echo "[WARNING] AWS_REGION environment variable is not set, automatically set depending on aws cli default region."
    export AWS_REGION=$(aws configure get region)
fi
echo "export AWS_REGION=${AWS_REGION}" >> env_vars
echo "[INFO] AWS_REGION = ${AWS_REGION}"

# Retrieve EKS CLUSTER Name if not already defined
if [[ -z "${EKS_CLUSTER_NAME}" ]]; then
    # Only retrieve from CloudFormation if not already set
    export EKS_CLUSTER_NAME=`aws cloudformation describe-stacks \
        --stack-name $STACK_ID \
        --query 'Stacks[0].Outputs[?OutputKey==\`OutputEKSClusterName\`].OutputValue' \
        --region ${AWS_REGION} \
        --output text`

    if [[ ! -z $EKS_CLUSTER_NAME ]]; then
        echo "export EKS_CLUSTER_NAME=${EKS_CLUSTER_NAME}" >> env_vars
        echo "[INFO] EKS_CLUSTER_NAME = ${EKS_CLUSTER_NAME}"
    else
        echo "[ERROR] failed to retrieve EKS_CLUSTER_NAME"
        return 1
    fi
else
    echo "[INFO] Using existing EKS_CLUSTER_NAME = ${EKS_CLUSTER_NAME}"
    echo "export EKS_CLUSTER_NAME=${EKS_CLUSTER_NAME}" >> env_vars
fi

# Retrieve EKS CLUSTER ARN
# Check if EKS_CLUSTER_ARN is already set and not empty
if [[ -z "${EKS_CLUSTER_ARN}" ]]; then
    # First attempt: retrieve from CloudFormation
    export EKS_CLUSTER_ARN=`aws cloudformation describe-stacks \
        --stack-name $STACK_ID \
        --query 'Stacks[0].Outputs[?OutputKey==\`OutputEKSClusterArn\`].OutputValue' \
        --region ${AWS_REGION} \
        --output text`

    # Second attempt: verify cluster exists and get ARN
    if [[ -z "${EKS_CLUSTER_ARN}" && ! -z "${EKS_CLUSTER_NAME}" ]]; then
        # Verify cluster exists
        if aws eks describe-cluster --name ${EKS_CLUSTER_NAME} --region ${AWS_REGION} &>/dev/null; then
            export EKS_CLUSTER_ARN=`aws eks describe-cluster \
                --name ${EKS_CLUSTER_NAME} \
                --query 'cluster.arn' \
                --region ${AWS_REGION} \
                --output text`
            echo "[INFO] Retrieved EKS_CLUSTER_ARN from existing cluster"
        else
            echo "[ERROR] EKS cluster ${EKS_CLUSTER_NAME} does not exist in region ${AWS_REGION}"
            return 1
        fi
    fi

    if [[ ! -z "${EKS_CLUSTER_ARN}" ]]; then
        echo "export EKS_CLUSTER_ARN=${EKS_CLUSTER_ARN}" >> env_vars
        echo "[INFO] EKS_CLUSTER_ARN = ${EKS_CLUSTER_ARN}"
    else
        echo "[ERROR] failed to retrieve EKS_CLUSTER_ARN"
        return 1
    fi
else
    echo "[INFO] Using existing EKS_CLUSTER_ARN = ${EKS_CLUSTER_ARN}"
fi

# Check if S3_BUCKET_NAME is already set and not empty
if [[ -z "${S3_BUCKET_NAME}" ]]; then
    # Retrieve S3 Bucket Name 
    export S3_BUCKET_NAME=`aws cloudformation describe-stacks \
        --stack-name $STACK_ID \
        --query 'Stacks[0].Outputs[?OutputKey==\`OutputS3BucketName\`].OutputValue' \
        --region ${AWS_REGION} \
        --output text`

    if [[ ! -z $S3_BUCKET_NAME ]]; then
        echo "export S3_BUCKET_NAME=${S3_BUCKET_NAME}" >> env_vars
        echo "[INFO] S3_BUCKET_NAME = ${S3_BUCKET_NAME}"
    else
        echo "[ERROR] failed to retrieve S3_BUCKET_NAME"
        return 1
    fi
else
    echo "[INFO] Using existing S3_BUCKET_NAME = ${S3_BUCKET_NAME}"
    echo "export S3_BUCKET_NAME=${S3_BUCKET_NAME}" >> env_vars
fi

# Check if EXECUTION_ROLE is already set and not empty
if [[ -z "${EXECUTION_ROLE}" ]]; then
    # Retrieve SageMaker Execution Role 
    export EXECUTION_ROLE=`aws cloudformation describe-stacks \
        --stack-name $STACK_ID \
        --query 'Stacks[0].Outputs[?OutputKey==\`OutputSageMakerIAMRoleArn\`].OutputValue' \
        --region ${AWS_REGION} \
        --output text`

    if [[ ! -z $EXECUTION_ROLE ]]; then
        echo "export EXECUTION_ROLE=${EXECUTION_ROLE}" >> env_vars
        echo "[INFO] EXECUTION_ROLE = ${EXECUTION_ROLE}"
    else
        echo "[ERROR] failed to retrieve EXECUTION_ROLE"
        return 1
    fi
else
    echo "[INFO] Using existing EXECUTION_ROLE = ${EXECUTION_ROLE}"
    echo "export EXECUTION_ROLE=${EXECUTION_ROLE}" >> env_vars
fi

# Check if VPC_ID is already set and not empty
if [[ -z "${VPC_ID}" ]]; then
    # Only retrieve from CloudFormation if not already set
    export VPC_ID=`aws cloudformation describe-stacks \
        --stack-name $STACK_ID \
        --query 'Stacks[0].Outputs[?OutputKey==\`OutputVpcId\`].OutputValue' \
        --region ${AWS_REGION} \
        --output text`

    if [[ ! -z $VPC_ID ]]; then
        echo "export VPC_ID=${VPC_ID}" >> env_vars
        echo "[INFO] VPC_ID = ${VPC_ID}"
    else
        echo "[ERROR] failed to retrieve VPC_ID"
        return 1
    fi
else
    echo "[INFO] Using existing VPC_ID = ${VPC_ID}"
    echo "export VPC_ID=${VPC_ID}" >> env_vars
fi

# Check if PRIVATE_SUBNET_ID is already set and not empty
if [[ -z "${PRIVATE_SUBNET_ID}" ]]; then
    # Only retrieve from CloudFormation if not already set
    export PRIVATE_SUBNET_ID=`aws cloudformation describe-stacks \
        --stack-name $STACK_ID \
        --query 'Stacks[0].Outputs[?OutputKey==\`OutputPrivateSubnetIds\`].OutputValue' \
        --region ${AWS_REGION} \
        --output text`

    if [[ ! -z $PRIVATE_SUBNET_ID ]]; then
        echo "export PRIVATE_SUBNET_ID=${PRIVATE_SUBNET_ID}" >> env_vars
        echo "[INFO] PRIVATE_SUBNET_ID = ${PRIVATE_SUBNET_ID}"
    else
        echo "[ERROR] failed to retrieve PRIVATE_SUBNET_ID"
        return 1
    fi
else
    echo "[INFO] Using existing PRIVATE_SUBNET_ID = ${PRIVATE_SUBNET_ID}"
    echo "export PRIVATE_SUBNET_ID=${PRIVATE_SUBNET_ID}" >> env_vars
fi

# Check if SECURITY_GROUP_ID is already set and not empty
if [[ -z "${SECURITY_GROUP_ID}" ]]; then
    # Only retrieve from CloudFormation if not already set
    export SECURITY_GROUP_ID=`aws cloudformation describe-stacks \
        --stack-name $STACK_ID \
        --query 'Stacks[0].Outputs[?OutputKey==\`OutputSecurityGroupId\`].OutputValue' \
        --region ${AWS_REGION} \
        --output text`

    if [[ ! -z $SECURITY_GROUP_ID ]]; then
        echo "export SECURITY_GROUP_ID=${SECURITY_GROUP_ID}" >> env_vars
        echo "[INFO] SECURITY_GROUP_ID = ${SECURITY_GROUP_ID}"
    else
        echo "[ERROR] failed to retrieve SECURITY_GROUP_ID"
        return 1
    fi
else
    echo "[INFO] Using existing SECURITY_GROUP_ID = ${SECURITY_GROUP_ID}"
    echo "export SECURITY_GROUP_ID=${SECURITY_GROUP_ID}" >> env_vars
fi


# Define accelerated compute instance type.
if [ -z ${ACCEL_INSTANCE_TYPE} ]; then
    echo "[WARNING] ACCEL_INSTANCE_TYPE environment variable is not set, automatically set to ml.g5.12xlarge."
    export ACCEL_INSTANCE_TYPE=ml.g5.12xlarge
fi
echo "export ACCEL_INSTANCE_TYPE=${ACCEL_INSTANCE_TYPE}" >> env_vars
echo "[INFO] ACCEL_INSTANCE_TYPE = ${ACCEL_INSTANCE_TYPE}"

# Set number of accelerated compute nodes to deploy 
if [ -z ${ACCEL_INSTANCE_COUNT} ]; then
    echo "[WARNING] ACCEL_INSTANCE_COUNT environment variable is not set, automatically set to 1."
    export ACCEL_INSTANCE_COUNT=1
fi
echo "export ACCEL_INSTANCE_COUNT=${ACCEL_INSTANCE_COUNT}" >> env_vars
echo "[INFO] ACCEL_INSTANCE_COUNT = ${ACCEL_INSTANCE_COUNT}"

# Set the EBS Volume size for the accelerated compute nodes 
if [ -z ${ACCEL_VOLUME_SIZE} ]; then
    echo "[WARNING] ACCEL_VOLUME_SIZE environment variable is not set, automatically set to 500."
    export ACCEL_VOLUME_SIZE=500
fi
echo "export ACCEL_VOLUME_SIZE=${ACCEL_VOLUME_SIZE}" >> env_vars
echo "[INFO] ACCEL_VOLUME_SIZE = ${ACCEL_VOLUME_SIZE}"

# Define general purpose compute instance type.
if [ -z ${GEN_INSTANCE_TYPE} ]; then
    echo "[WARNING] GEN_INSTANCE_TYPE environment variable is not set, automatically set to ml.m5.2xlarge."
    export GEN_INSTANCE_TYPE=ml.m5.2xlarge
fi
echo "export GEN_INSTANCE_TYPE=${GEN_INSTANCE_TYPE}" >> env_vars
echo "[INFO] GEN_INSTANCE_TYPE = ${GEN_INSTANCE_TYPE}"

# Set the number of general purpose nodes to deploy
if [ -z ${GEN_INSTANCE_COUNT} ]; then
    echo "[WARNING] GEN_INSTANCE_COUNT environment variable is not set, automatically set to 1."
    export GEN_INSTANCE_COUNT=1
fi
echo "export GEN_INSTANCE_COUNT=${GEN_INSTANCE_COUNT}" >> env_vars
echo "[INFO] GEN_INSTANCE_COUNT = ${GEN_INSTANCE_COUNT}"

# Set the EBS Volume size for the general purpose compute nodes 
if [ -z ${GEN_VOLUME_SIZE} ]; then
    echo "[WARNING] GEN_VOLUME_SIZE environment variable is not set, automatically set to 500."
    export GEN_VOLUME_SIZE=500
fi
echo "export GEN_VOLUME_SIZE=${GEN_VOLUME_SIZE}" >> env_vars
echo "[INFO] GEN_VOLUME_SIZE = ${GEN_VOLUME_SIZE}"

# Set auto-recovery
if [ -z ${NODE_RECOVERY} ]; then
    echo "[WARNING] NODE_RECOVERY environment variable is not set, set to Automatic."
    export NODE_RECOVERY="Automatic"
fi
echo "export NODE_RECOVERY=${NODE_RECOVERY}" >> env_vars
echo "[INFO] NODE_RECOVERY = ${NODE_RECOVERY}"

# Set network flag for Docker if in SageMaker Code Editor
if [ "${SAGEMAKER_APP_TYPE:-}" = "CodeEditor" ]; then 
    echo "export DOCKER_NETWORK=\"--network sagemaker\"" >> env_vars
fi 

# Get absolute path of env_vars file
ENV_VARS_PATH="$(realpath "$(dirname "$0")/env_vars")"

# Persist the environment variables
add_source_command() {
    local config_file="$1"
    local source_line="[ -f \"${ENV_VARS_PATH}\" ] && source \"${ENV_VARS_PATH}\""
    
    # Only add if the line doesn't exist already
    if ! grep -q "source.*${ENV_VARS_PATH}" "$config_file"; then
        echo "$source_line" >> "$config_file"
        echo "[INFO] Added environment variables to $config_file"
    else
        echo "[INFO] Environment variables already configured in $config_file"
    fi
}

# Check shell config files
if [ -f ~/.bashrc ]; then
    add_source_command ~/.bashrc
fi

if [ -f ~/.zshrc ]; then
    add_source_command ~/.zshrc
fi