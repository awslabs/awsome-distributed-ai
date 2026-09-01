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

: "${STACK_ID_VPC:=parallelcluster-prerequisites}"

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

if [ -z ${INSTANCE} ]; then
    echo "[WARNING] INSTANCES environment variable is not set, automatically set to p5.48xlarge."
    export INSTANCE=p5.48xlarge
fi
echo "export INSTANCE=${INSTANCE}" >> env_vars
echo "[INFO] INSTANCE = ${INSTANCE}"

if [ -z ${NUM_INSTANCES} ]; then
    echo "[WARNING] NUM_INSTANCES environment variable is not set, automatically set to 2"
    export NUM_INSTANCES=2
fi
echo "export NUM_INSTANCES=${NUM_INSTANCES}" >> env_vars
echo "[INFO] NUM_INSTANCES = ${NUM_INSTANCES}"

if [ -z ${CAPACITY_RESERVATION_ID} ]; then
    echo "[WARNING] CAPACITY_RESERVATION_ID environment variable is not set, automatically set to 2"
    export CAPACITY_RESERVATION_ID="NA"
fi
echo "export CAPACITY_RESERVATION_ID=${CAPACITY_RESERVATION_ID}" >> env_vars
echo "[INFO] CAPACITY_RESERVATION_ID = ${CAPACITY_RESERVATION_ID}"

if [ -z ${KEY_PAIR_NAME} ]; then
    echo "[WARNING] KEY_PAIR_NAME environment variable is not set, assuming that you will not use it."
    export KEY_PAIR_NAME="REMOVE_THIS_LINE_AND_A_LINE_BEFORE"
fi
echo "export KEY_PAIR_NAME=${KEY_PAIR_NAME}" >> env_vars
echo "[INFO] KEY_PAIR_NAME = ${KEY_PAIR_NAME}"

if [ -z ${AMPREMOTEWRITEURL} ]; then
    echo "[WARNING] AMPREMOTEWRITEURL environment variable is not set, assuming that you will not use it."
    export AMPREMOTEWRITEURL="NA"
fi
echo "export AMPREMOTEWRITEURL=${AMPREMOTEWRITEURL}" >> env_vars
echo "[INFO] AMPREMOTEWRITEURL = ${AMPREMOTEWRITEURL}"

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
export PRIVATE_SUBNET_ID=`aws cloudformation describe-stacks \
    --stack-name $STACK_ID_VPC \
    --query 'Stacks[0].Outputs[?OutputKey==\`PrimaryPrivateSubnet\`].OutputValue' \
    --region ${AWS_REGION} \
    --output text`

if [[ ! -z $PRIVATE_SUBNET_ID ]]; then
    echo "export PRIVATE_SUBNET_ID=${PRIVATE_SUBNET_ID}" >> env_vars
    echo "[INFO] PRIVATE_SUBNET_ID = ${PRIVATE_SUBNET_ID}"
else
    echo "[ERROR] failed to retrieve PRIVATE_SUBNET_ID"
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

# Get FSxO Filesystem Root Volume from CloudFormation
export FSXO_ID=`aws cloudformation describe-stacks \
    --stack-name $STACK_ID_VPC \
    --query 'Stacks[0].Outputs[?OutputKey==\`FSxORootVolumeId\`].OutputValue' \
    --region ${AWS_REGION} \
    --output text`

if [[ ! -z $FSX_MOUNTNAME ]]; then
    echo "export FSXO_ID=${FSXO_ID}" >> env_vars
    echo "[INFO] FSXO_ID = ${FSXO_ID}"
else
    echo "[ERROR] failed to retrieve FSx for OpenZFS Root Volume ID"
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
