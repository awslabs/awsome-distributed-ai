#!/usr/bin/env bash
set -euo pipefail
REGION="${REGION:-ap-south-1}"
ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
TAG="${TAG:-$(git rev-parse --short=12 HEAD)-2608}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/mnt/fsx/ubuntu/workspace/artifacts/adai-kimi-k2-megatron-ep/image-build-${TAG}}"
REPOSITORY_PREFIX="${REPOSITORY_PREFIX:-adai/kimi-k2-megatron-ep-2608}"
mkdir -p "${ARTIFACT_DIR}"
ROOT="$(git rev-parse --show-toplevel)"
DOCKERFILE="${ROOT}/3.test_cases/megatron/megatron-bridge/Dockerfile"

aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com" >/dev/null

declare -A TARGETS=(
  [nccl-alltoall]=nccl-alltoall-final
  [uccl]=uccl-final
  [deepep-v1-nvshmem]=deepep-v1-nvshmem-final
  [deepep-v2-gin-gda]=deepep-v2-gin-gda-final
  [deepep-v2-pr5-control]=deepep-v2-pr5-control
)

for arm in nccl-alltoall uccl deepep-v1-nvshmem deepep-v2-gin-gda deepep-v2-pr5-control; do
  repository="${REPOSITORY_PREFIX}-${arm}"
  aws ecr describe-repositories --region "${REGION}" --repository-names "${repository}" >/dev/null 2>&1 || \
    aws ecr create-repository --region "${REGION}" --repository-name "${repository}" \
      --image-scanning-configuration scanOnPush=true >/dev/null
  image="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${repository}:${TAG}"
  docker buildx build --progress=plain --load --target "${TARGETS[$arm]}" -f "${DOCKERFILE}" -t "${image}" "${ROOT}" \
    2>&1 | tee "${ARTIFACT_DIR}/${arm}.build.log"
  docker push "${image}" 2>&1 | tee "${ARTIFACT_DIR}/${arm}.push.log"
  digest="$(aws ecr describe-images --region "${REGION}" --repository-name "${repository}" \
    --image-ids imageTag="${TAG}" --query 'imageDetails[0].imageDigest' --output text)"
  [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]
  printf '%s=%s@%s\n' "${arm}" "${image%:*}" "${digest}" | tee -a "${ARTIFACT_DIR}/images.env"
done
sha256sum "${ARTIFACT_DIR}"/*.log "${ARTIFACT_DIR}/images.env" > "${ARTIFACT_DIR}/SHA256SUMS"
