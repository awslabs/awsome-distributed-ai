#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Build and push this recipe's EFA-enabled DeepSeek-V4.1 image (./Dockerfile)
# to ECR. Same flow as ../build-image.sh, but with this recipe's Dockerfile,
# its own ECR repository, and an overridable base image so the mutable
# dev-dsv41 tag can be pinned by digest:
#
#   ./build-image.sh                                            # base = lmsysorg/sglang:dev-dsv41
#   BASE_IMAGE=lmsysorg/sglang@sha256:<digest> ./build-image.sh # pinned base
#
# Prints the resolved base digest and the NIXL / SGLang versions baked into the
# image on stderr, and the pushed image URI on the last line of stdout.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

algorithm_name=sgl-dsv41-efa
dockerfilename=Dockerfile
BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang:dev-dsv41}"

export DOCKER_BUILDKIT=1

region=$(aws configure get region)
account=$(aws sts get-caller-identity --query Account --output text)

aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin "${account}.dkr.ecr.${region}.amazonaws.com"

aws ecr describe-repositories --region $region --repository-names "${algorithm_name}" > /dev/null 2>&1 || {
    echo "create repository:" "${algorithm_name}"
    aws ecr create-repository --region $region  --repository-name "${algorithm_name}" > /dev/null
}

# Pull the base explicitly so its digest can be recorded (dev-* tags move).
docker pull "${BASE_IMAGE}"
echo "base image digest: $(docker inspect --format '{{index .RepoDigests 0}}' "${BASE_IMAGE}")" >&2

docker build --build-arg BASE_IMAGE="${BASE_IMAGE}" -t ${algorithm_name} -f ${dockerfilename} .

# Report what got baked in (importlib.metadata avoids importing torch/CUDA on a
# GPU-less build host).
echo "versions baked into ${algorithm_name}:" >&2
docker run --rm --entrypoint python3 ${algorithm_name} -c \
    'import importlib.metadata as m; print("  nixl", m.version("nixl"), "| sglang", m.version("sglang"))' >&2

timestamp=$(date +%Y%m%d%H%M%S)
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:${timestamp}"
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}

echo $fullname
