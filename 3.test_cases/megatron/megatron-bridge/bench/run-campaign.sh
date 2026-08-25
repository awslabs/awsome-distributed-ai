#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
: "${CTX:?set CTX}"
: "${LOCAL_ARTIFACT_ROOT:?set LOCAL_ARTIFACT_ROOT}"
CAMPAIGN_ID="${CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-kimi-k2-megatron-ep-2608}"
namespace_suffix="$(printf '%s' "${CAMPAIGN_ID}" | sha256sum | cut -c1-16)"
CAMPAIGN_NAMESPACE="${CAMPAIGN_NAMESPACE:-adai-kimi-k2-megatron-ep-${namespace_suffix}}"
case "${CAMPAIGN_NAMESPACE}" in adai-kimi-k2-megatron-ep-*) ;; *) echo "refusing non-campaign namespace ${CAMPAIGN_NAMESPACE}" >&2; exit 2;; esac
[[ "${#CAMPAIGN_NAMESPACE}" -le 63 ]] || { echo "campaign namespace exceeds 63 characters: ${CAMPAIGN_NAMESPACE}" >&2; exit 2; }
export CAMPAIGN_ID CAMPAIGN_NAMESPACE CTX LOCAL_ARTIFACT_ROOT
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE_DIR="$(cd "${SELF_DIR}/.." && pwd)"
RUN_ENTRYPOINT_LOCAL_FILE="${RUN_ENTRYPOINT_LOCAL_FILE:-${CASE_DIR}/kimi-k2/benchmarks/bench_kimi_k2_pretrain.py}"
export RUN_ENTRYPOINT_LOCAL_FILE
OUT="${LOCAL_ARTIFACT_ROOT}/${CAMPAIGN_ID}"
mkdir -p "${OUT}/census" "${OUT}/control" "${OUT}/results"
K=(kubectl --context "${CTX}")
LOCK_NS="${LOCK_NS:-default}"
CLUSTER_LOCK="${CLUSTER_LOCK:-adai-ap-south-1-gpu-campaign-lock}"
HOLDER_ID="kimi-k2-megatron-ep-${CAMPAIGN_ID}"
LOCK_DURATION_SECONDS="${LOCK_DURATION_SECONDS:-57600}"
LOCK_HELD=0
VLLM_NAMESPACE="${VLLM_NAMESPACE:-dsv3-ep-backend-comparison-20260823}"
DEFAULT_VLLM_PROTECTED_NODES="ip-10-6-100-229.ap-south-1.compute.internal,ip-10-6-105-109.ap-south-1.compute.internal,ip-10-6-105-110.ap-south-1.compute.internal,ip-10-6-106-109.ap-south-1.compute.internal"
mapfile -t DISCOVERED_PROTECTED_NODES < <(
  "${K[@]}" -n "${VLLM_NAMESPACE}" get pods -o json 2>/dev/null |
    jq -r '.items[].spec.nodeName // empty' | sort -u
)
IFS=',' read -r -a EXPLICIT_PROTECTED_NODES <<<"${PROTECTED_NODES:-${DEFAULT_VLLM_PROTECTED_NODES}}"
PROTECTED_NODE_SET=("${DISCOVERED_PROTECTED_NODES[@]}" "${EXPLICIT_PROTECTED_NODES[@]}")
mapfile -t PROTECTED_NODE_SET < <(printf '%s\n' "${PROTECTED_NODE_SET[@]}" | sed '/^$/d' | sort -u)
PROTECTED_NODES="$(IFS=,; echo "${PROTECTED_NODE_SET[*]}")"
export PROTECTED_NODES
printf '%s\n' "${PROTECTED_NODE_SET[@]}" > "${OUT}/control/protected-nodes.txt"

census() {
  local label="$1"
  local dir="${OUT}/census/${label}"
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
  now="$(date -u +%Y-%m-%dT%H:%M:%S.000000Z)"
  if current="$("${K[@]}" -n "${LOCK_NS}" get lease "${CLUSTER_LOCK}" -o json 2>/dev/null)"; then
    jq --arg holder "${HOLDER_ID}" --arg now "${now}" \
      --argjson duration "${LOCK_DURATION_SECONDS}" \
      '.spec.holderIdentity=$holder | .spec.acquireTime=$now | .spec.renewTime=$now | .spec.leaseDurationSeconds=$duration' \
      <<<"${current}" | "${K[@]}" replace -f - >/dev/null
  else
    jq -n --arg namespace "${LOCK_NS}" --arg name "${CLUSTER_LOCK}" \
      --arg holder "${HOLDER_ID}" --arg now "${now}" \
      --argjson duration "${LOCK_DURATION_SECONDS}" \
      '{apiVersion:"coordination.k8s.io/v1",kind:"Lease",metadata:{namespace:$namespace,name:$name},spec:{holderIdentity:$holder,acquireTime:$now,renewTime:$now,leaseDurationSeconds:$duration}}' |
      "${K[@]}" create -f - >/dev/null
  fi
  LOCK_HELD=1
}

renew_lock() {
  [[ "${LOCK_HELD}" -eq 1 ]] || return 0
  local current holder now
  current="$("${K[@]}" -n "${LOCK_NS}" get lease "${CLUSTER_LOCK}" -o json)"
  holder="$(jq -r '.spec.holderIdentity' <<<"${current}")"
  [[ "${holder}" = "${HOLDER_ID}" ]] || { echo "lost cluster lock to ${holder}" >&2; exit 20; }
  now="$(date -u +%Y-%m-%dT%H:%M:%S.000000Z)"
  jq --arg now "${now}" '.spec.renewTime=$now' <<<"${current}" | "${K[@]}" replace -f - >/dev/null
}

release_lock() {
  [[ "${LOCK_HELD}" -eq 1 ]] || return 0
  local current holder
  current="$("${K[@]}" -n "${LOCK_NS}" get lease "${CLUSTER_LOCK}" -o json 2>/dev/null || true)"
  holder=""
  if [[ -n "${current}" ]]; then
    holder="$(jq -r '.spec.holderIdentity // ""' <<<"${current}")"
  fi
  if [[ "${holder}" = "${HOLDER_ID}" ]]; then
    jq '.spec.holderIdentity="" | .spec.leaseDurationSeconds=1' <<<"${current}" | "${K[@]}" replace -f - >/dev/null
  fi
  LOCK_HELD=0
}

teardown() {
  local ns="${CAMPAIGN_NAMESPACE}"
  local owner
  owner="$("${K[@]}" get namespace "${ns}" -o jsonpath='{.metadata.labels.adai-campaign}' 2>/dev/null || true)"
  if [[ "${owner}" = "${CAMPAIGN_ID}" ]]; then
    "${K[@]}" delete namespace "${ns}" --wait=true --timeout=300s >/dev/null 2>&1 || true
  elif [[ -n "${owner}" ]]; then
    printf 'REFUSED_FOREIGN_NAMESPACE namespace=%s owner=%s\n' "${ns}" "${owner}" > "${OUT}/control/teardown-refusal.STATUS"
  fi
  release_lock
}
trap teardown EXIT

free_nodes() {
  local instance_type="$1"
  local efa_per_node="$2"
  "${K[@]}" get nodes -l "node.kubernetes.io/instance-type=${instance_type}" -o json | jq -r --argjson efa_per_node "${efa_per_node}" --slurpfile pods <("${K[@]}" get pods -A -o json) '
    .items[] as $node |
    ($pods[0].items | map(select(.spec.nodeName == $node.metadata.name and (.status.phase == "Running" or .status.phase == "Pending")) | [.spec.containers[]?.resources.requests["nvidia.com/gpu"] // "0" | tonumber] | add // 0) | add // 0) as $used |
    select($used == 0) |
    select(($node.status.allocatable["nvidia.com/gpu"] // "0" | tonumber) >= 8) |
    select(($node.status.allocatable["vpc.amazonaws.com/efa"] // "0" | tonumber) >= $efa_per_node) |
    select(any($node.status.conditions[]; .type == "Ready" and .status == "True")) |
    $node.metadata.name' | sort | while IFS= read -r candidate; do
      protected=0
      for node in "${PROTECTED_NODE_SET[@]}"; do
        if [[ "${candidate}" = "${node}" ]]; then
          protected=1
          break
        fi
      done
      [[ "${protected}" -eq 1 ]] || printf '%s\n' "${candidate}"
    done
}

census before
mapfile -t B300_FREE < <(free_nodes p6-b300.48xlarge 16)
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
mapfile -t B300_FREE < <(free_nodes p6-b300.48xlarge 16)

if [[ "${#B300_FREE[@]}" -ge 32 ]]; then
  SELECTED=("${B300_FREE[@]:0:32}")
  export INSTANCE_TYPE=p6-b300.48xlarge EFA_PER_NODE=16
  NNODES=32
  PROFILE=headline
else
  mapfile -t B200_FREE < <(free_nodes p6-b200.48xlarge 8)
  if [[ "${FULL_B200_FALLBACK:-0}" = 1 ]]; then
    requested_b200_nodes=32
  else
    requested_b200_nodes="${QUAL_NODES:-2}"
  fi
  if [[ "${ALLOW_FALLBACK_B200:-1}" != 1 || "${#B200_FREE[@]}" -lt "${requested_b200_nodes}" ]]; then
    printf 'NOT_RUN_INSUFFICIENT_CAPACITY timestamp_utc=%s requested_fallback_nodes=%d available_free_b200_nodes=%d\n' \
      "$(date -u +%FT%TZ)" "${requested_b200_nodes}" "${#B200_FREE[@]}" > "${OUT}/results/qualification.STATUS"
    census insufficient-fallback
    exit 0
  fi
  SELECTED=("${B200_FREE[@]:0:${requested_b200_nodes}}")
  export INSTANCE_TYPE=p6-b200.48xlarge EFA_PER_NODE=8
  NNODES="${requested_b200_nodes}"
  if [[ "${FULL_B200_FALLBACK:-0}" = 1 ]]; then
    PROFILE=b200-256gpu
  else
    PROFILE=qualification
  fi
fi

CAPACITY_BLOCK_END="$(
  aws ec2 describe-capacity-reservations --region ap-south-1 |
    jq -r --arg instance_type "${INSTANCE_TYPE}" \
      '[.CapacityReservations[] | select(.State == "active" and .InstanceType == $instance_type) | .EndDate] | sort | .[0] // empty'
)"
if [[ -z "${CAPACITY_BLOCK_END}" ]]; then
  printf 'NOT_RUN_NO_CAPACITY_BLOCK timestamp_utc=%s instance_type=%s\n' \
    "$(date -u +%FT%TZ)" "${INSTANCE_TYPE}" > "${OUT}/results/${PROFILE}.STATUS"
  exit 0
fi
CAPACITY_BLOCK_END_EPOCH="$(date -u -d "${CAPACITY_BLOCK_END}" +%s)"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-7200}"
RUN_CLEANUP_BUFFER_SECONDS="${RUN_CLEANUP_BUFFER_SECONDS:-600}"
RUN_WINDOW_SECONDS=$((RUN_TIMEOUT_SECONDS + RUN_CLEANUP_BUFFER_SECONDS))
RUN_START_MIN_REMAINING_SECONDS="${RUN_START_MIN_REMAINING_SECONDS_OVERRIDE:-${RUN_WINDOW_SECONDS}}"
export RUN_TIMEOUT_SECONDS
if [[ "${PROFILE}" = qualification ]]; then initial_runs=1; else initial_runs=4; fi
INITIAL_MATRIX_EXPECTED_SECONDS="${INITIAL_MATRIX_EXPECTED_SECONDS_OVERRIDE:-$((initial_runs * RUN_WINDOW_SECONDS))}"
[[ "${RUN_START_MIN_REMAINING_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "RUN_START_MIN_REMAINING_SECONDS_OVERRIDE must be a positive integer" >&2
  exit 2
}
[[ "${INITIAL_MATRIX_EXPECTED_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "INITIAL_MATRIX_EXPECTED_SECONDS_OVERRIDE must be a positive integer" >&2
  exit 2
}
cat > "${OUT}/control/reclaim-window.txt" <<EOF
capacity_block_end=${CAPACITY_BLOCK_END}
run_timeout_seconds=${RUN_TIMEOUT_SECONDS}
run_cleanup_buffer_seconds=${RUN_CLEANUP_BUFFER_SECONDS}
run_start_min_remaining_seconds=${RUN_START_MIN_REMAINING_SECONDS}
initial_runs=${initial_runs}
initial_matrix_expected_seconds=${INITIAL_MATRIX_EXPECTED_SECONDS}
EOF
remaining_seconds=$((CAPACITY_BLOCK_END_EPOCH - $(date -u +%s)))
if (( remaining_seconds < INITIAL_MATRIX_EXPECTED_SECONDS )); then
  printf 'NOT_RUN_RECLAIM_WINDOW timestamp_utc=%s remaining_seconds=%d required_seconds=%d capacity_block_end=%s\n' \
    "$(date -u +%FT%TZ)" "${remaining_seconds}" "${INITIAL_MATRIX_EXPECTED_SECONDS}" "${CAPACITY_BLOCK_END}" \
    > "${OUT}/results/${PROFILE}.STATUS"
  exit 0
fi

run_window_available() {
  (( CAPACITY_BLOCK_END_EPOCH - $(date -u +%s) >= RUN_START_MIN_REMAINING_SECONDS ))
}

NODE_NAMES="$(IFS=,; echo "${SELECTED[*]}")"
export NODE_NAMES
printf '%s\n' "${SELECTED[@]}" > "${OUT}/control/selected-nodes.txt"

if [[ "${PROFILE}" != qualification ]]; then
  export TENSOR_PARALLEL=8 PIPELINE_PARALLEL=8 EXPERT_PARALLEL=32 TRAIN_ITERS="${TRAIN_ITERS_OVERRIDE:-40}" GLOBAL_BATCH="${GLOBAL_BATCH_OVERRIDE:-256}"
  CELLS=(throughput-no-overlap throughput-overlap small-message small-message-overlap)
  REPEATS=(1 2 3)
else
  export TENSOR_PARALLEL=8 PIPELINE_PARALLEL=2 EXPERT_PARALLEL=8 TRAIN_ITERS=8 GLOBAL_BATCH=16
  CELLS=(qualification-no-overlap qualification-overlap)
  REPEATS=(1)
fi
if [[ -n "${CELLS_OVERRIDE:-}" ]]; then read -r -a CELLS <<<"${CELLS_OVERRIDE}"; fi
if [[ -n "${REPEATS_OVERRIDE:-}" ]]; then read -r -a REPEATS <<<"${REPEATS_OVERRIDE}"; fi

STOP_CAMPAIGN=0
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
      if ! run_window_available; then
        printf 'NOT_RUN_RECLAIM_WINDOW timestamp_utc=%s cell=%s repeat=%s next_arm=%s capacity_block_end=%s\n' \
          "$(date -u +%FT%TZ)" "${cell}" "${repeat}" "${arm}" "${CAPACITY_BLOCK_END}" \
          > "${OUT}/results/remaining-matrix.STATUS"
        STOP_CAMPAIGN=1
        break
      fi
      renew_lock
      CELL="${cell}" REPEAT="${repeat}" MICRO_BATCH="${mb}" MOE_A2A_OVERLAP="${overlap}" \
        "${CASE_DIR}/run-ab-rawpods.sh" "${arm}" "${NNODES}"
      ns="${CAMPAIGN_NAMESPACE}"
      "${K[@]}" -n "${ns}" delete pods,service -l adai-campaign="${CAMPAIGN_ID}" --wait=true --timeout=300s >/dev/null
    done
    if [[ "${STOP_CAMPAIGN}" -eq 1 ]]; then break; fi
  done
  if [[ "${STOP_CAMPAIGN}" -eq 1 ]]; then break; fi
done

python3 "${SELF_DIR}/parse-runs.py" "${OUT}/26.08/kimi-k2" --warmup 8 --output "${OUT}/results/index.json"
census after
teardown
trap - EXIT
find "${OUT}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${OUT}/SHA256SUMS"
