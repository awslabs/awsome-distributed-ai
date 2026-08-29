#!/usr/bin/env bash
set -euo pipefail

: "${LOCAL_ARTIFACT_DIR:?set LOCAL_ARTIFACT_DIR}"
: "${IMG_NCCL_ALLTOALL_PARENT:?set IMG_NCCL_ALLTOALL_PARENT}"
: "${IMG_UCCL_PARENT:?set IMG_UCCL_PARENT}"
: "${IMG_DEEPEP_V1_PARENT:?set IMG_DEEPEP_V1_PARENT}"
: "${IMG_DEEPEP_V2_PARENT:?set IMG_DEEPEP_V2_PARENT}"

REGION="${REGION:-ap-south-1}"
ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
TAG="${TAG:-overlap-schedule-$(git rev-parse --short=8 HEAD)}"
REPOSITORY_PREFIX="${REPOSITORY_PREFIX:-adai/kimi-k2-megatron-ep-2608}"
ROOT="$(git rev-parse --show-toplevel)"
DOCKERFILE="${ROOT}/3.test_cases/megatron/megatron-bridge/Dockerfile.mcore-overlap-schedule"

mkdir -p "${LOCAL_ARTIFACT_DIR}"
aws ecr get-login-password --region "${REGION}" |
  docker login --username AWS --password-stdin \
    "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com" >/dev/null

declare -A parents=(
  [nccl-alltoall]="${IMG_NCCL_ALLTOALL_PARENT}"
  [uccl]="${IMG_UCCL_PARENT}"
  [deepep-v1-nvshmem]="${IMG_DEEPEP_V1_PARENT}"
  [deepep-v2-gin-gda]="${IMG_DEEPEP_V2_PARENT}"
)

build_arm() {
  local arm="$1"
  local parent="${parents[$arm]}"
  local repository="${REPOSITORY_PREFIX}-${arm}"
  local image="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${repository}:${TAG}"
  local arm_dir="${LOCAL_ARTIFACT_DIR}/${arm}"
  mkdir -p "${arm_dir}"
  printf 'arm=%s\nparent=%s\ntag=%s\n' "${arm}" "${parent}" "${TAG}" > "${arm_dir}/build-inputs.txt"
  aws ecr describe-repositories --region "${REGION}" --repository-names "${repository}" >/dev/null
  docker buildx build --progress=plain --push \
    --build-arg "QUALIFIED_PARENT=${parent}" \
    --build-arg "EP_ARM=${arm}" \
    -f "${DOCKERFILE}" -t "${image}" "${ROOT}" \
    > "${arm_dir}/build.log" 2>&1
  local digest
  digest="$(
    aws ecr describe-images --region "${REGION}" --repository-name "${repository}" \
      --image-ids "imageTag=${TAG}" --query 'imageDetails[0].imageDigest' --output text
  )"
  [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]
  printf '%s@%s\n' "${image%:*}" "${digest}" > "${arm_dir}/image.txt"
  sha256sum "${arm_dir}/build-inputs.txt" "${arm_dir}/build.log" "${arm_dir}/image.txt" \
    > "${arm_dir}/SHA256SUMS"
}

pids=()
for arm in nccl-alltoall uccl deepep-v1-nvshmem deepep-v2-gin-gda; do
  build_arm "${arm}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
[[ "${failed}" -eq 0 ]]

find "${LOCAL_ARTIFACT_DIR}" -mindepth 2 -maxdepth 2 -name image.txt -print0 |
  sort -z | xargs -0 cat > "${LOCAL_ARTIFACT_DIR}/images.txt"
sha256sum "${LOCAL_ARTIFACT_DIR}"/*/SHA256SUMS "${LOCAL_ARTIFACT_DIR}/images.txt" \
  > "${LOCAL_ARTIFACT_DIR}/SHA256SUMS"
