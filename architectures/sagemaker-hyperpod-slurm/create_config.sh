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

# : "${STACK_ID_VPC:=sagemaker-hyperpod}"

# Check for JQ
if ! command -v jq &> /dev/null
then
    echo -e "please install jq...\nsudo yum install -y jq or brew install jq"
    exit 1
fi

# Define AWS Region
if [ -z ${AWS_REGION} ]; then
    echo "[WARNING] AWS_REGION environment variable is not set, automatically set depending on aws cli default region."
    export AWS_REGION=$(aws configure get region)
fi
echo "export AWS_REGION=${AWS_REGION}" >> env_vars
echo "[INFO] AWS_REGION = ${AWS_REGION}"

# Define Instances seperated by ','.
if [ -z ${INSTANCES} ]; then
    echo "[WARNING] INSTANCES environment variable is not set, automatically set to g5.12xlarge."
    export INSTANCES=g5.12xlarge
fi
echo "export INSTANCES=${INSTANCES}" >> env_vars
echo "[INFO] INSTANCES = ${INSTANCES}"

# Retrieve VPC ID
export VPC_ID=`aws cloudformation describe-stacks \
    --stack-name $STACK_ID_VPC \
    --query 'Stacks[0].Outputs[?OutputKey==\`VPC\`].OutputValue' \
    --region ${AWS_REGION} \
    --output text`

if [[ ! -z $VPC_ID ]]; then
    echo "export VPC_ID=${VPC_ID}" >> env_vars
    echo "[INFO] VPC_ID = ${VPC_ID}"
else
    echo "[ERROR] failed to retrieve VPC ID"
    return 1
fi

# Grab the subnet id
export SUBNET_ID=`aws cloudformation describe-stacks \
    --stack-name $STACK_ID_VPC \
    --query 'Stacks[0].Outputs[?OutputKey==\`PrimaryPrivateSubnet\`].OutputValue' \
    --region ${AWS_REGION} \
    --output text`

if [[ ! -z $SUBNET_ID ]]; then
    echo "export SUBNET_ID=${SUBNET_ID}" >> env_vars
    echo "[INFO] SUBNET_ID = ${SUBNET_ID}"
else
    echo "[ERROR] failed to retrieve SUBNET ID"
    return 1
fi

# Grab the backup subnet id
export BACKUP_SUBNET=`aws cloudformation describe-stacks \
    --stack-name $STACK_ID_VPC \
    --query 'Stacks[0].Outputs[?OutputKey==\`BackupPrivateSubnet\`].OutputValue' \
    --region ${AWS_REGION} \
    --output text`

if [[ ! -z $BACKUP_SUBNET ]]; then
    echo "export BACKUP_SUBNET=${BACKUP_SUBNET}" >> env_vars
    echo "[INFO] BACKUP_SUBNET = ${BACKUP_SUBNET}"
else
    echo "[ERROR] failed to retrieve BACKUP SUBNET ID"
    return 1
fi


# Grab the subnet id
export PUBLIC_SUBNET_ID=`aws cloudformation describe-stacks \
    --stack-name $STACK_ID_VPC \
    --query 'Stacks[0].Outputs[?OutputKey==\`PublicSubnet\`].OutputValue' \
    --region ${AWS_REGION} \
    --output text`

if [[ ! -z $PUBLIC_SUBNET_ID ]]; then
    echo "export PUBLIC_SUBNET_ID=${PUBLIC_SUBNET_ID}" >> env_vars
    echo "[INFO] PUBLIC_SUBNET_ID = ${PUBLIC_SUBNET_ID}"
else
    echo "[ERROR] failed to retrieve Public SUBNET ID"
    return 1
fi

# Get FSx Filesystem id from CloudFormation
export FSX_ID=`aws cloudformation describe-stacks \
    --stack-name $STACK_ID_VPC \
    --query 'Stacks[0].Outputs[?OutputKey==\`FSxLustreFilesystemId\`].OutputValue' \
    --region ${AWS_REGION} \
    --output text`

if [[ ! -z $FSX_ID ]]; then
    echo "export FSX_ID=${FSX_ID}" >> env_vars
    echo "[INFO] FSX_ID = ${FSX_ID}"
else
    echo "[ERROR] failed to retrieve FSX ID"
    return 1
fi

# Get FSx Filesystem Mountname from CloudFormation
export FSX_MOUNTNAME=`aws cloudformation describe-stacks \
    --stack-name $STACK_ID_VPC \
    --query 'Stacks[0].Outputs[?OutputKey==\`FSxLustreFilesystemMountname\`].OutputValue' \
    --region ${AWS_REGION} \
    --output text`

if [[ ! -z $FSX_MOUNTNAME ]]; then
    echo "export FSX_MOUNTNAME=${FSX_MOUNTNAME}" >> env_vars
    echo "[INFO] FSX_MOUNTNAME = ${FSX_MOUNTNAME}"
else
    echo "[ERROR] failed to retrieve FSX Mountname"
    return 1
fi

# Get FSx Security Group from CloudFormation
export SECURITY_GROUP=`aws cloudformation describe-stacks \
    --stack-name $STACK_ID_VPC \
    --query 'Stacks[0].Outputs[?OutputKey==\`SecurityGroup\`].OutputValue' \
    --region ${AWS_REGION} \
    --output text`

if [[ ! -z $SECURITY_GROUP ]]; then
    echo "export SECURITY_GROUP=${SECURITY_GROUP}" >> env_vars
    echo "[INFO] SECURITY_GROUP = ${SECURITY_GROUP}"
else
    echo "[ERROR] failed to retrieve FSX Security Group"
    return 1
fi

# Get sagemaker role ARN 
export ROLE=`aws cloudformation describe-stacks \
    --stack-name $STACK_ID_VPC \
    --query 'Stacks[0].Outputs[?OutputKey==\`AmazonSagemakerClusterExecutionRoleArn\`].OutputValue' \
    --region ${AWS_REGION} \
    --output text`

if [[ ! -z $ROLE ]]; then
    echo "export ROLE=${ROLE}" >> env_vars
    echo "[INFO] ROLE = ${ROLE}"
else
    echo "[ERROR] failed to retrieve Role ARN"
    return 1
fi

# Get sagemaker role ROLENAME 
export ROLENAME=$(basename "$ROLE")

if [[ ! -z $ROLENAME ]]; then
    echo "export ROLENAME=${ROLENAME}" >> env_vars
    echo "[INFO] ROLENAME = ${ROLENAME}"
else
    echo "[ERROR] failed to retrieve Role NAME"
    return 1
fi

# Get s3 bucket name
export BUCKET=`aws cloudformation describe-stacks \
    --stack-name $STACK_ID_VPC \
    --query 'Stacks[0].Outputs[?OutputKey==\`AmazonS3BucketName\`].OutputValue' \
    --region ${AWS_REGION} \
    --output text`

if [[ ! -z $BUCKET ]]; then
    echo "export BUCKET=${BUCKET}" >> env_vars
    echo "[INFO] BUCKET = ${BUCKET}"
else
    echo "[ERROR] failed to retrieve Bucket Name"
    return 1
fi


