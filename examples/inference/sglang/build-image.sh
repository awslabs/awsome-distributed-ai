#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Build and push the shared EFA-enabled SGLang image (Dockerfile.efa) to ECR.
# This image — upstream lmsysorg/sglang + the AWS EFA installer — is the base
# for every multi-node example here; single-node examples use the upstream
# image directly and don't need it.
#
# Overridable through the environment (defaults build the Kimi image):
#   BASE_IMAGE      SGLang base image; pin mutable dev-*/nightly-* tags by
#                   digest (lmsysorg/sglang@sha256:...)
#   EFA_VERSION     AWS EFA installer version
#   ALGORITHM_NAME  ECR repository name
#
#   ./build-image.sh
#   BASE_IMAGE=lmsysorg/sglang@sha256:<digest> EFA_VERSION=1.50.0 \
#     ALGORITHM_NAME=sgl-dsv41-efa ./build-image.sh
#
# Prints the resolved base digest and the NIXL / SGLang versions baked into the
# image on stderr, and the pushed image URI on the last line of stdout.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang:v0.5.12.post1-cu130}"
EFA_VERSION="${EFA_VERSION:-1.47.0}"
algorithm_name="${ALGORITHM_NAME:-sgl-dev-cu13}"
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

# Pull the base explicitly so the digest that actually gets built is recorded
# (dev-* and nightly-* tags move).
docker pull "${BASE_IMAGE}"
echo "base image digest: $(docker inspect --format '{{index .RepoDigests 0}}' "${BASE_IMAGE}")" >&2

docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg EFA_VERSION="${EFA_VERSION}" \
    -t ${algorithm_name} -f ${dockerfilename} .

# Report what got baked in (importlib.metadata avoids importing torch/CUDA on a
# GPU-less build host). nixl-cu13 carries the CUDA 13 backend and is a separate
# distribution from nixl, so both are listed; absent packages print "-".
echo "versions baked into ${algorithm_name}:" >&2
docker run --rm --entrypoint python3 ${algorithm_name} -c '
import importlib.metadata as m
def v(p):
    try:
        return m.version(p)
    except m.PackageNotFoundError:
        return "-"
print("  " + " | ".join(p + " " + v(p) for p in ("nixl", "nixl-cu13", "sglang")))
' >&2

timestamp=$(date +%Y%m%d%H%M%S)
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:${timestamp}"
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}

echo $fullname
