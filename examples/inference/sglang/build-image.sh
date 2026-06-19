#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Build and push the shared EFA-enabled SGLang image (Dockerfile.efa) to ECR.
# This image — stock lmsysorg/sglang + the AWS EFA installer — is the base for
# every multi-node example here (Kimi 1P1D, and future inter-node topologies);
# single-node examples use the upstream image directly and don't need it.
# Prints the pushed image URI on the last line.

set -euo pipefail

algorithm_name=sgl-dev-cu13
dockerfilename=Dockerfile.efa

export DOCKER_BUILDKIT=1

region=$(aws configure get region)
account=$(aws sts get-caller-identity --query Account --output text)

aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin "${account}.dkr.ecr.${region}.amazonaws.com"
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin "763104351884.dkr.ecr.${region}.amazonaws.com"

aws ecr describe-repositories --region $region --repository-names "${algorithm_name}" > /dev/null 2>&1 || {
    echo "create repository:" "${algorithm_name}"
    aws ecr create-repository --region $region  --repository-name "${algorithm_name}" > /dev/null
}

docker build --pull -t ${algorithm_name} -f $dockerfilename .

timestamp=$(date +%Y%m%d%H%M%S)
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:${timestamp}"
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}

echo $fullname
