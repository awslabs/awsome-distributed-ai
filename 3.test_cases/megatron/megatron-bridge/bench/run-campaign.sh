#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
: "${CTX:?set CTX}"
: "${LOCAL_ARTIFACT_ROOT:?set LOCAL_ARTIFACT_ROOT}"
CAMPAIGN_ID="${CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-kimi-k2-megatron-ep-2608}"
export CAMPAIGN_ID CTX LOCAL_ARTIFACT_ROOT
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE_DIR="$(cd "${SELF_DIR}/.." && pwd)"
OUT="${LOCAL_ARTIFACT_ROOT}/${CAMPAIGN_ID}"
mkdir -p "${OUT}/census" "${OUT}/control" "${OUT}/results"
K=(kubectl --context "${CTX}")
LOCK_NS="${LOCK_NS:-default}"
CLUSTER_LOCK="${CLUSTER_LOCK:-adai-ap-south-1-gpu-campaign-lock}"
HOLDER_ID="kimi-k2-megatron-ep-${CAMPAIGN_ID}"
LOCK_HELD=0

census() {
  local label="$1" dir="${OUT}/census/${label}"
  mkdir -p "${dir}"
  date -u +%FT%TZ > "${dir}/time.txt"
  aws sts get-caller-identity > "${dir}/aws-caller.json"
  aws ec2 describe-capacity-reservations --region ap-south-1 > "${dir}/capacity-reservations.json"
  aws ec2 describe-instances --region ap-south-1 --filters Name=instance-state-name,Values=running > "${dir}/instances.json"
  "${K[@]}" get nodes -o json > "${dir}/nodes.json"
  "${K[@]}" get pods -A -o json > "${dir}/pods.json"
  "${K[@]}" get namespaces -o json > "${dir}/namespaces.json"
  "${K[@]}" -n "${LOCK_NS}" get lease "${CLUSTER_LOCK}" -o json > "${dir}/cluster-lock.json" 2>&1 || true
  find "${dir}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${dir}/SHA256SUMS"
}

lease_active_other() {
  local lease holder renew duration expiry now
  lease="$("${K[@]}" -n "${LOCK_NS}" get lease "${CLUSTER_LOCK}" -o json 2>/dev/null || true)"
  [[ -n "${lease}" ]] || return 1
  holder="$(jq -r '.spec.holderIdentity // ""' <<<"${lease}")"
  [[ -n "${holder}" && "${holder}" != "${HOLDER_ID}" ]] || return 1
  renew="$(jq -r '.spec.renewTime // .spec.acquireTime // "1970-01-01T00:00:00Z"' <<<"${lease}")"
  duration="$(jq -r '.spec.leaseDurationSeconds // 0' <<<"${lease}")"
  expiry=$(( $(date -u -d "${renew}" +%s) + duration ))
  now="$(date -u +%s)"
  (( expiry > now ))
}

acquire_lock() {
  if lease_active_other; then
    "${K[@]}" -n "${LOCK_NS}" get lease "${CLUSTER_LOCK}" -o json > "${OUT}/control/lock-conflict.json"
    return 1
  fi
  local now current
  now="$(date -u +%FT%TZ)"
  if current="$("${K[@]}" -n "${LOCK_NS}" get lease "${CLUSTER_LOCK}" -o json 2>/dev/null)"; then
    jq --arg holder "${HOLDER_ID}" --arg now "${now}" \
      '.spec.holderIdentity=$holder | .spec.acquireTime=$now | .spec.renewTime=$now | .spec.leaseDurationSeconds=7200' \
      <<<"${current}" | "${K[@]}" replace -f - >/dev/null
  else
    "${K[@]}" -n "${LOCK_NS}" create lease "${CLUSTER_LOCK}" --duration=7200s --holder-identity="${HOLDER_ID}" >/dev/null
  fi
  LOCK_HELD=1
}

renew_lock() {
  [[ "${LOCK_HELD}" -eq 1 ]] || return 0
  local current holder now
  current="$("${K[@]}" -n "${LOCK_NS}" get lease "${CLUSTER_LOCK}" -o json)"
  holder="$(jq -r '.spec.holderIdentity' <<<"${current}")"
  [[ "${holder}" = "${HOLDER_ID}" ]] || { echo "lost cluster lock to ${holder}" >&2; exit 20; }
  now="$(date -u +%FT%TZ)"
  jq --arg now "${now}" '.spec.renewTime=$now' <<<"${current}" | "${K[@]}" replace -f - >/dev/null
}

release_lock() {
  [[ "${LOCK_HELD}" -eq 1 ]] || return 0
  local current holder
  current="$("${K[@]}" -n "${LOCK_NS}" get lease "${CLUSTER_LOCK}" -o json 2>/dev/null || true)"
  holder="$(jq -r '.spec.holderIdentity // ""' <<<"${current:-{}}")"
  if [[ "${holder}" = "${HOLDER_ID}" ]]; then
    jq '.spec.holderIdentity="" | .spec.leaseDurationSeconds=1' <<<"${current}" | "${K[@]}" replace -f - >/dev/null
  fi
  LOCK_HELD=0
}

teardown() {
  local ns="adai-kimi-k2-megatron-ep-${CAMPAIGN_ID,,}"
  "${K[@]}" delete namespace "${ns}" --ignore-not-found --wait=true --timeout=300s >/dev/null 2>&1 || true
  release_lock
}
trap teardown EXIT

free_nodes() {
  local instance_type="$1"
  "${K[@]}" get nodes -l "node.kubernetes.io/instance-type=${instance_type}" -o json | jq -r --slurpfile pods <("${K[@]}" get pods -A -o json) '
    .items[] as $node |
    ($pods[0].items | map(select(.spec.nodeName == $node.metadata.name and (.status.phase == "Running" or .status.phase == "Pending")) | [.spec.containers[]?.resources.requests["nvidia.com/gpu"] // "0" | tonumber] | add // 0) | add // 0) as $used |
    select($used == 0) |
    select(any($node.status.conditions[]; .type == "Ready" and .status == "True")) |
    $node.metadata.name' | sort
}

census before
mapfile -t B300_FREE < <(free_nodes p6-b300.48xlarge)
if [[ "${#B300_FREE[@]}" -lt 32 ]]; then
  printf 'NOT_RUN_INSUFFICIENT_CAPACITY timestamp_utc=%s required_nodes=32 available_free_b300_nodes=%d\n' \
    "$(date -u +%FT%TZ)" "${#B300_FREE[@]}" > "${OUT}/results/headline-256gpu.STATUS"
fi

if ! acquire_lock; then
  printf 'NOT_RUN_CONCURRENT_CAMPAIGN timestamp_utc=%s lock=%s\n' "$(date -u +%FT%TZ)" "${CLUSTER_LOCK}" > "${OUT}/results/qualification.STATUS"
  census blocked-by-lock
  exit 0
fi

# Re-evaluate occupancy after the atomic lease claim. A census taken before the
# claim is provenance, not permission to use a node whose state has changed.
mapfile -t B300_FREE < <(free_nodes p6-b300.48xlarge)

if [[ "${#B300_FREE[@]}" -ge 32 ]]; then
  SELECTED=("${B300_FREE[@]:0:32}")
  export INSTANCE_TYPE=p6-b300.48xlarge EFA_PER_NODE=16
  NNODES=32
  PROFILE=headline
else
  mapfile -t B200_FREE < <(free_nodes p6-b200.48xlarge)
  QUAL_NODES="${QUAL_NODES:-2}"
  if [[ "${ALLOW_FALLBACK_B200:-1}" != 1 || "${#B200_FREE[@]}" -lt "${QUAL_NODES}" ]]; then
    printf 'NOT_RUN_INSUFFICIENT_CAPACITY timestamp_utc=%s requested_fallback_nodes=%d available_free_b200_nodes=%d\n' \
      "$(date -u +%FT%TZ)" "${QUAL_NODES}" "${#B200_FREE[@]}" > "${OUT}/results/qualification.STATUS"
    census insufficient-fallback
    exit 0
  fi
  SELECTED=("${B200_FREE[@]:0:${QUAL_NODES}}")
  export INSTANCE_TYPE=p6-b200.48xlarge EFA_PER_NODE=8
  NNODES="${QUAL_NODES}"
  PROFILE=qualification
fi
NODE_NAMES="$(IFS=,; echo "${SELECTED[*]}")"
export NODE_NAMES
printf '%s\n' "${SELECTED[@]}" > "${OUT}/control/selected-nodes.txt"

if [[ "${PROFILE}" = headline ]]; then
  export TENSOR_PARALLEL=8 PIPELINE_PARALLEL=8 EXPERT_PARALLEL=32 TRAIN_ITERS=40 GLOBAL_BATCH=256
  CELLS=(throughput-no-overlap throughput-overlap small-message small-message-overlap)
  REPEATS=(1 2 3)
else
  export TENSOR_PARALLEL=8 PIPELINE_PARALLEL=2 EXPERT_PARALLEL=8 TRAIN_ITERS=8 GLOBAL_BATCH=16
  CELLS=(qualification-no-overlap qualification-overlap)
  REPEATS=(1)
fi

for repeat in "${REPEATS[@]}"; do
  mapfile -t ORDER < <(python3 - "${repeat}" <<'PY'
import random, sys
arms = ["nccl-alltoall", "uccl", "deepep-v1-nvshmem", "deepep-v2-gin-gda"]
random.Random(1234 + int(sys.argv[1])).shuffle(arms)
print(*arms, sep="\n")
PY
  )
  printf '%s\n' "${ORDER[@]}" > "${OUT}/control/arm-order-repeat-${repeat}.txt"
  for cell in "${CELLS[@]}"; do
    if [[ "${cell}" == *overlap && "${cell}" != *no-overlap ]]; then overlap=on; else overlap=off; fi
    if [[ "${cell}" == small-message* || "${PROFILE}" = qualification ]]; then mb=1; else mb=4; fi
    for arm in "${ORDER[@]}"; do
      renew_lock
      CELL="${cell}" REPEAT="${repeat}" MICRO_BATCH="${mb}" MOE_A2A_OVERLAP="${overlap}" \
        "${CASE_DIR}/run-ab-rawpods.sh" "${arm}" "${NNODES}"
      ns="adai-kimi-k2-megatron-ep-${CAMPAIGN_ID,,}"
      "${K[@]}" -n "${ns}" delete pods,service -l adai-campaign="${CAMPAIGN_ID}" --wait=true --timeout=300s >/dev/null
    done
  done
done

python3 "${SELF_DIR}/parse-runs.py" "${OUT}/26.08/kimi-k2" --warmup 8 --output "${OUT}/results/index.json"
census after
teardown
trap - EXIT
find "${OUT}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${OUT}/SHA256SUMS"
