#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

EP_ARM="${1:?usage: run-ab-rawpods.sh EP_ARM NNODES}"
NNODES="${2:?usage: run-ab-rawpods.sh EP_ARM NNODES}"
case "${EP_ARM}" in
  nccl-alltoall|uccl|deepep-v1-nvshmem|deepep-v2-gin-gda) ;;
  *) echo "unknown EP_ARM: ${EP_ARM}" >&2; exit 2 ;;
esac

: "${CTX:?set CTX}"
: "${CAMPAIGN_ID:?set CAMPAIGN_ID}"
: "${NODE_NAMES:?set comma-separated NODE_NAMES}"
: "${LOCAL_ARTIFACT_ROOT:?set LOCAL_ARTIFACT_ROOT to durable controller storage}"
NS="${NS:-${CAMPAIGN_NAMESPACE:-adai-kimi-k2-megatron-ep-${CAMPAIGN_ID,,}}}"
case "${NS}" in adai-kimi-k2-megatron-ep-*) ;; *) echo "refusing non-campaign namespace ${NS}" >&2; exit 2;; esac
[[ "${#NS}" -le 63 ]] || { echo "campaign namespace exceeds 63 characters: ${NS}" >&2; exit 2; }

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
ARMS_FILE="${ARMS_FILE:-${SELF_DIR}/bench/arms.yaml}"
read -r IMAGE_ENV EXPECTED_DISPATCHER EXPECTED_BACKEND < <(
  python3 - "${ARMS_FILE}" "${EP_ARM}" <<'PY'
import sys, yaml
entry = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["arms"][sys.argv[2]]
print(entry["image_env"], entry["dispatcher"], entry["backend"] or "none")
PY
)
IMAGE="${!IMAGE_ENV:-}"
: "${IMAGE:?set ${IMAGE_ENV} to an immutable image URI}"
[[ "${IMAGE}" == *@sha256:* ]] || { echo "image must be resolved by digest: ${IMAGE}" >&2; exit 2; }
HARVEST_IMAGE="${HARVEST_IMAGE:-${IMG_NCCL_ALLTOALL:-${IMAGE}}}"
[[ "${HARVEST_IMAGE}" == *@sha256:* ]] || { echo "harvest image must be resolved by digest: ${HARVEST_IMAGE}" >&2; exit 2; }

RUN_ENTRYPOINT="${RUN_ENTRYPOINT:-/opt/benchmark/case/kimi-k2/benchmarks/bench_kimi_k2_pretrain.py}"
RUN_ENTRYPOINT_ARGS_JSON="${RUN_ENTRYPOINT_ARGS_JSON:-[]}"
RUN_ENTRYPOINT_LOCAL_FILE="${RUN_ENTRYPOINT_LOCAL_FILE:-}"
RUN_BENCHMARK_LOCAL_FILE="${RUN_BENCHMARK_LOCAL_FILE:-}"
RUN_INPUT_JSON_B64="${RUN_INPUT_JSON_B64:-}"
python3 - "${RUN_ENTRYPOINT_ARGS_JSON}" <<'PY'
import json, sys
args = json.loads(sys.argv[1])
assert isinstance(args, list) and all(isinstance(value, str) and "\n" not in value for value in args), args
PY
RUN_ENTRYPOINT_ARGS_B64="$(printf '%s' "${RUN_ENTRYPOINT_ARGS_JSON}" | base64 -w0)"
RUN_ENTRYPOINT_SOURCE_B64=""
RUN_ENTRYPOINT_SOURCE_SHA256=""
if [[ -n "${RUN_ENTRYPOINT_LOCAL_FILE}" ]]; then
  [[ -f "${RUN_ENTRYPOINT_LOCAL_FILE}" ]] || { echo "missing local entrypoint source: ${RUN_ENTRYPOINT_LOCAL_FILE}" >&2; exit 2; }
  RUN_ENTRYPOINT_SOURCE_B64="$(base64 -w0 < "${RUN_ENTRYPOINT_LOCAL_FILE}")"
  (( ${#RUN_ENTRYPOINT_SOURCE_B64} < 700000 )) || { echo "local entrypoint source is too large for a pod environment" >&2; exit 2; }
  RUN_ENTRYPOINT_SOURCE_SHA256="$(sha256sum "${RUN_ENTRYPOINT_LOCAL_FILE}" | awk '{print $1}')"
  RUN_ENTRYPOINT=/run-artifacts/run-entrypoint.py
else
  [[ "${RUN_ENTRYPOINT}" =~ ^/opt/benchmark/case/[A-Za-z0-9_./-]+$ ]] || { echo "invalid image entrypoint: ${RUN_ENTRYPOINT}" >&2; exit 2; }
fi
RUN_BENCHMARK_SOURCE_B64=""
RUN_BENCHMARK_SOURCE_SHA256=""
if [[ -n "${RUN_BENCHMARK_LOCAL_FILE}" ]]; then
  [[ -f "${RUN_BENCHMARK_LOCAL_FILE}" ]] || { echo "missing local benchmark source: ${RUN_BENCHMARK_LOCAL_FILE}" >&2; exit 2; }
  RUN_BENCHMARK_SOURCE_B64="$(base64 -w0 < "${RUN_BENCHMARK_LOCAL_FILE}")"
  (( ${#RUN_BENCHMARK_SOURCE_B64} < 700000 )) || { echo "local benchmark source is too large for a pod environment" >&2; exit 2; }
  RUN_BENCHMARK_SOURCE_SHA256="$(sha256sum "${RUN_BENCHMARK_LOCAL_FILE}" | awk '{print $1}')"
fi
if [[ -n "${RUN_INPUT_JSON_B64}" ]]; then
  printf '%s' "${RUN_INPUT_JSON_B64}" | base64 -d >/dev/null
fi

IFS=',' read -r -a NODES <<<"${NODE_NAMES}"
[[ "${#NODES[@]}" -eq "${NNODES}" ]] || { echo "NODE_NAMES has ${#NODES[@]} nodes, expected ${NNODES}" >&2; exit 2; }
IFS=',' read -r -a PROTECTED <<<"${PROTECTED_NODES:-}"
for node in "${NODES[@]}"; do
  for protected in "${PROTECTED[@]}"; do
    [[ -z "${protected}" || "${node}" != "${protected}" ]] || { echo "refusing protected node ${node}" >&2; exit 3; }
  done
done

K=(kubectl --context "${CTX}")
INSTANCE_TYPE="${INSTANCE_TYPE:-p6-b300.48xlarge}"
EFA_PER_NODE="${EFA_PER_NODE:-16}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
[[ "${GPUS_PER_NODE}" =~ ^[1-8]$ ]] || { echo "GPUS_PER_NODE must be an integer from 1 through 8" >&2; exit 2; }
WORLD=$((NNODES * GPUS_PER_NODE))
nodes_json="$("${K[@]}" get nodes -o json)"
live_pods_json="$("${K[@]}" get pods -A -o json)"
for node in "${NODES[@]}"; do
  node_json="$(jq -c --arg n "${node}" '[.items[] | select(.metadata.name==$n)] | if length == 1 then .[0] else error("expected one named node") end' <<<"${nodes_json}")"
  actual="$(jq -r '.metadata.labels["node.kubernetes.io/instance-type"] // ""' <<<"${node_json}")"
  [[ "${actual}" = "${INSTANCE_TYPE}" ]] || { echo "node ${node} is ${actual}, expected ${INSTANCE_TYPE}" >&2; exit 3; }
  ready="$(jq -r '[.status.conditions[] | select(.type=="Ready") | .status][0] // ""' <<<"${node_json}")"
  [[ "${ready}" = True ]] || { echo "node ${node} is not Ready" >&2; exit 3; }
  allocatable_gpu="$(jq -r '.status.allocatable["nvidia.com/gpu"] // "0" | tonumber' <<<"${node_json}")"
  allocatable_efa="$(jq -r '.status.allocatable["vpc.amazonaws.com/efa"] // "0" | tonumber' <<<"${node_json}")"
  [[ "${allocatable_gpu}" -ge "${GPUS_PER_NODE}" ]] || { echo "node ${node} exposes ${allocatable_gpu} GPUs, expected at least ${GPUS_PER_NODE}" >&2; exit 3; }
  [[ "${allocatable_efa}" -ge "${EFA_PER_NODE}" ]] || { echo "node ${node} exposes ${allocatable_efa} EFA devices, expected at least ${EFA_PER_NODE}" >&2; exit 3; }
  occupied="$(jq --arg n "${node}" '[.items[] | select(.spec.nodeName==$n and (.status.phase=="Running" or .status.phase=="Pending")) | .spec.containers[]?.resources.requests["nvidia.com/gpu"] // "0" | tonumber] | add // 0' <<<"${live_pods_json}")"
  [[ "${occupied}" -eq 0 ]] || { echo "node ${node} already has ${occupied} requested GPUs" >&2; exit 3; }
done

CELL="${CELL:-qualification}"
REPEAT="${REPEAT:-1}"
RUN_KIND="${RUN_KIND:-training}"
RUN_KEY="${CELL}/repeat-${REPEAT}/${EP_ARM}"
LOCAL_RUN_DIR="${LOCAL_ARTIFACT_ROOT}/${CAMPAIGN_ID}/26.08/kimi-k2/${RUN_KEY}"
mkdir -p "${LOCAL_RUN_DIR}/pod-logs" "${LOCAL_RUN_DIR}/snapshots" "${LOCAL_RUN_DIR}/manifests"
[[ ! -e "${LOCAL_RUN_DIR}/STATUS" ]] || { echo "refusing completed run ${LOCAL_RUN_DIR}" >&2; exit 4; }
"${K[@]}" get nodes "${NODES[@]}" -o json > "${LOCAL_RUN_DIR}/manifests/nodes-before.json"
"${K[@]}" get pods -A -o json > "${LOCAL_RUN_DIR}/manifests/pods-before.json"
cp "${ARMS_FILE}" "${LOCAL_RUN_DIR}/manifests/arms.yaml"
sha256sum "${SELF_DIR}/kimi-k2/benchmarks/bench_kimi_k2_pretrain.py" > "${LOCAL_RUN_DIR}/manifests/model-config.sha256"
if [[ -n "${RUN_ENTRYPOINT_LOCAL_FILE}" ]]; then
  cp "${RUN_ENTRYPOINT_LOCAL_FILE}" "${LOCAL_RUN_DIR}/manifests/run-entrypoint.py"
fi
if [[ -n "${RUN_BENCHMARK_LOCAL_FILE}" ]]; then
  cp "${RUN_BENCHMARK_LOCAL_FILE}" "${LOCAL_RUN_DIR}/manifests/benchmark-entrypoint.py"
fi

if "${K[@]}" get namespace "${NS}" >/dev/null 2>&1; then
  owner="$("${K[@]}" get namespace "${NS}" -o jsonpath='{.metadata.labels.adai-campaign}')"
  [[ "${owner}" = "${CAMPAIGN_ID}" ]] || { echo "namespace ${NS} is not owned by ${CAMPAIGN_ID}" >&2; exit 5; }
else
  "${K[@]}" create namespace "${NS}"
  "${K[@]}" label namespace "${NS}" adai-campaign="${CAMPAIGN_ID}" adai-owner=kimi-k2-megatron-ep
fi
KN=("${K[@]}" -n "${NS}")

safe_arm="${EP_ARM//[^a-z0-9-]/-}"
JOB="mk2-${CELL//[^a-z0-9-]/-}-r${REPEAT}-${safe_arm}"
# Leave room for the largest 32-node rank suffix ("-31") so both the Pod
# metadata name and explicit hostname remain valid DNS labels.
JOB="${JOB:0:60}"
PORT=23456
TP="${TENSOR_PARALLEL:-8}"
PP="${PIPELINE_PARALLEL:-8}"
EP="${EXPERT_PARALLEL:-32}"
TRAIN_ITERS="${TRAIN_ITERS:-40}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
MICRO_BATCH="${MICRO_BATCH:-4}"
SEQ_LEN="${SEQ_LEN:-4096}"
MOE_A2A_OVERLAP="${MOE_A2A_OVERLAP:-off}"
PRESERVE_ROUTE_TRACE_RAW="${PRESERVE_ROUTE_TRACE_RAW:-0}"
HARVEST_PARALLELISM="${HARVEST_PARALLELISM:-4}"
[[ "${PRESERVE_ROUTE_TRACE_RAW}" =~ ^[01]$ ]] || { echo "PRESERVE_ROUTE_TRACE_RAW must be 0 or 1" >&2; exit 2; }
[[ "${HARVEST_PARALLELISM}" =~ ^[1-9][0-9]*$ && "${HARVEST_PARALLELISM}" -le 16 ]] || { echo "HARVEST_PARALLELISM must be an integer from 1 through 16" >&2; exit 2; }

cat > "${LOCAL_RUN_DIR}/environment.txt" <<EOF
campaign_id=${CAMPAIGN_ID}
cell=${CELL}
repeat=${REPEAT}
ep_arm=${EP_ARM}
image=${IMAGE}
harvest_image=${HARVEST_IMAGE}
nodes=${NODE_NAMES}
world_size=${WORLD}
tp=${TP}
pp=${PP}
ep=${EP}
etp=1
train_iterations=${TRAIN_ITERS}
global_batch_samples=${GLOBAL_BATCH}
micro_batch_samples=${MICRO_BATCH}
sequence_length_tokens=${SEQ_LEN}
ep_overlap=${MOE_A2A_OVERLAP}
route_trace_raw_preserved=${PRESERVE_ROUTE_TRACE_RAW}
harvest_parallelism=${HARVEST_PARALLELISM}
expected_dispatcher=${EXPECTED_DISPATCHER}
expected_backend=${EXPECTED_BACKEND}
run_kind=${RUN_KIND}
run_entrypoint=${RUN_ENTRYPOINT}
run_entrypoint_args_json=${RUN_ENTRYPOINT_ARGS_JSON}
run_entrypoint_source_sha256=${RUN_ENTRYPOINT_SOURCE_SHA256}
benchmark_entrypoint_source_sha256=${RUN_BENCHMARK_SOURCE_SHA256}
EOF

cat <<EOF | "${KN[@]}" apply -f -
apiVersion: v1
kind: Service
metadata: {name: ${JOB}, labels: {adai-campaign: "${CAMPAIGN_ID}", adai-owner: kimi-k2-megatron-ep, app: ${JOB}}}
spec:
  clusterIP: None
  selector: {app: ${JOB}}
  ports: [{name: rendezvous, port: ${PORT}}]
EOF

snapshot() {
  while "${K[@]}" get namespace "${NS}" >/dev/null 2>&1; do
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    "${KN[@]}" get pods -l app="${JOB}" -o json > "${LOCAL_RUN_DIR}/snapshots/pods-${stamp}.json" 2>/dev/null || true
    for rank in $(seq 0 $((NNODES - 1))); do
      "${KN[@]}" logs "${JOB}-${rank}" -c trainer --timestamps > "${LOCAL_RUN_DIR}/snapshots/${stamp}-node-rank-${rank}.log" 2>&1 || true
    done
    sleep 30
  done
}
snapshot &
SNAPSHOT_PID=$!
# shellcheck disable=SC2317  # Invoked indirectly by the EXIT trap.
cleanup_snapshot() {
  pkill -TERM -P "${SNAPSHOT_PID}" >/dev/null 2>&1 || true
  kill "${SNAPSHOT_PID}" >/dev/null 2>&1 || true
  wait "${SNAPSHOT_PID}" >/dev/null 2>&1 || true
}
trap cleanup_snapshot EXIT

EXTRA_ENV=""
case "${EP_ARM}" in
  uccl)
    EXTRA_ENV='        - {name: PER_EXPERT_BATCHING, value: "1"}
        - {name: UCCL_SOCKET_IFNAME, value: "eth0"}' ;;
  deepep-v1-nvshmem)
    EXTRA_ENV='        - {name: NVSHMEM_REMOTE_TRANSPORT, value: "libfabric"}
        - {name: NVSHMEM_LIBFABRIC_PROVIDER, value: "efa"}' ;;
  deepep-v2-gin-gda)
    EXTRA_ENV='        - {name: NCCL_GIN_TYPE, value: "5"}
        - {name: NCCL_SYM_GIN_KERNELS_ENABLE, value: "0"}' ;;
esac

for rank in $(seq 0 $((NNODES - 1))); do
  node="${NODES[$rank]}"
  profile_prefix=""
  if [[ "${NSYS_PROFILE:-0}" = 1 && "${rank}" -eq 0 ]]; then
    profile_prefix='nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --output=/run-artifacts/nsys-node-rank-0 '
  fi
  gdr_mount=""
  gdr_volume=""
  privileged=""
  if [[ "${EP_ARM}" = deepep-v1-nvshmem || "${EP_ARM}" = deepep-v2-gin-gda ]]; then
    gdr_mount='- {name: gdrdrv, mountPath: /dev/gdrdrv}'
    gdr_volume='- {name: gdrdrv, hostPath: {path: /dev/gdrdrv, type: CharDevice}}'
    privileged='securityContext: {privileged: true}'
  fi
  cat <<EOF | "${KN[@]}" apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${JOB}-${rank}
  labels: {app: ${JOB}, adai-campaign: "${CAMPAIGN_ID}", adai-owner: kimi-k2-megatron-ep, rank: "${rank}"}
spec:
  restartPolicy: Never
  nodeName: ${node}
  hostname: ${JOB}-${rank}
  subdomain: ${JOB}
  tolerations: [{operator: Exists}]
  containers:
    - name: trainer
      image: ${IMAGE}
      imagePullPolicy: IfNotPresent
      command: [bash, -lc]
      args:
        - >-
          set -o pipefail;
          python3 -c 'import json,os; p=json.load(open("/opt/benchmark/backend.json")); assert p["ep_arm"]==os.environ["EP_ARM"],p';
          cp /opt/benchmark/backend.json /run-artifacts/backend.json;
          cp /opt/benchmark/common-build-manifest.json /run-artifacts/common-build-manifest.json;
          cp /opt/benchmark/image-verification.json /run-artifacts/image-verification.json;
          if [[ -n "\${RUN_ENTRYPOINT_SOURCE_B64}" ]]; then printf '%s' "\${RUN_ENTRYPOINT_SOURCE_B64}" | base64 -d > /run-artifacts/run-entrypoint.py; chmod 0444 /run-artifacts/run-entrypoint.py; fi;
          if [[ -n "\${RUN_BENCHMARK_SOURCE_B64}" ]]; then printf '%s' "\${RUN_BENCHMARK_SOURCE_B64}" | base64 -d > /run-artifacts/benchmark-entrypoint.py; chmod 0444 /run-artifacts/benchmark-entrypoint.py; export KIMI_BENCHMARK_ENTRYPOINT=/run-artifacts/benchmark-entrypoint.py; fi;
          if [[ -n "\${RUN_INPUT_JSON_B64}" ]]; then printf '%s' "\${RUN_INPUT_JSON_B64}" | base64 -d > /run-artifacts/run-input.json; fi;
          mapfile -t run_args < <(python3 -c 'import base64,json,os; [print(value) for value in json.loads(base64.b64decode(os.environ["RUN_ENTRYPOINT_ARGS_B64"]))]');
          python3 /opt/benchmark/case/bench/collect-runtime-manifest.py /run-artifacts/runtime-manifest.json;
          /opt/benchmark/case/bench/collect-node-telemetry.sh /run-artifacts/telemetry & telemetry_pid=\$!;
          ${profile_prefix}torchrun --nnodes=${NNODES} --nproc-per-node=${GPUS_PER_NODE} --node-rank=${rank}
          --master-addr=${JOB}-0.${JOB}.${NS}.svc.cluster.local --master-port=${PORT}
          ${RUN_ENTRYPOINT} "\${run_args[@]}"
          2>&1 | tee /run-artifacts/node-rank-${rank}.log;
          train_rc=\${PIPESTATUS[0]};
          python3 /opt/benchmark/case/bench/summarize-route-trace.py /run-artifacts/router-trace /run-artifacts/route-summary.json || true;
          if [[ "\${PRESERVE_ROUTE_TRACE_RAW}" = 1 ]]; then
            find /run-artifacts/router-trace -type f -name '*.jsonl' -exec gzip -9 -n {} + 2>/dev/null || true;
          else
            find /run-artifacts/router-trace -type f -name '*.jsonl' -delete 2>/dev/null || true;
          fi;
          kill \${telemetry_pid} >/dev/null 2>&1 || true; wait \${telemetry_pid} || true;
          (cd /run-artifacts && find . -type f ! -name custody.sha256 -print0 | sort -z | xargs -0 sha256sum > custody.sha256);
          exit \${train_rc}
      env:
        - {name: EP_ARM, value: "${EP_ARM}"}
        - {name: TENSOR_PARALLEL, value: "${TP}"}
        - {name: PIPELINE_PARALLEL, value: "${PP}"}
        - {name: EXPERT_PARALLEL, value: "${EP}"}
        - {name: TRAIN_ITERS, value: "${TRAIN_ITERS}"}
        - {name: GLOBAL_BATCH, value: "${GLOBAL_BATCH}"}
        - {name: MICRO_BATCH, value: "${MICRO_BATCH}"}
        - {name: SEQ_LEN, value: "${SEQ_LEN}"}
        - {name: MOE_A2A_OVERLAP, value: "${MOE_A2A_OVERLAP}"}
        - {name: MOE_FORCE_BALANCE, value: "on"}
        - {name: PERFORMANCE_SEED, value: "1234"}
        - {name: RUN_KIND, value: "${RUN_KIND}"}
        - {name: RUN_ENTRYPOINT_ARGS_B64, value: "${RUN_ENTRYPOINT_ARGS_B64}"}
        - {name: RUN_ENTRYPOINT_SOURCE_B64, value: "${RUN_ENTRYPOINT_SOURCE_B64}"}
        - {name: RUN_BENCHMARK_SOURCE_B64, value: "${RUN_BENCHMARK_SOURCE_B64}"}
        - {name: RUN_INPUT_JSON_B64, value: "${RUN_INPUT_JSON_B64}"}
        - {name: ROUTER_TRACE_DIR, value: "/run-artifacts/router-trace"}
        - {name: PRESERVE_ROUTE_TRACE_RAW, value: "${PRESERVE_ROUTE_TRACE_RAW}"}
        - {name: FI_PROVIDER, value: "efa"}
        - {name: FI_EFA_USE_DEVICE_RDMA, value: "1"}
        - {name: NCCL_DEBUG, value: "INFO"}
        - {name: NCCL_DEBUG_SUBSYS, value: "INIT,NET"}
${EXTRA_ENV}
      resources:
        requests: {nvidia.com/gpu: ${GPUS_PER_NODE}, vpc.amazonaws.com/efa: ${EFA_PER_NODE}}
        limits: {nvidia.com/gpu: ${GPUS_PER_NODE}, vpc.amazonaws.com/efa: ${EFA_PER_NODE}}
      ${privileged}
      volumeMounts:
        - {name: run-artifacts, mountPath: /run-artifacts}
        - {name: shm, mountPath: /dev/shm}
        ${gdr_mount}
    - name: harvester
      image: ${HARVEST_IMAGE}
      imagePullPolicy: IfNotPresent
      command: [bash, -lc]
      args: ['printf "HARVEST_SIDECAR_READY timestamp_utc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; sleep infinity']
      resources:
        requests: {cpu: 10m, memory: 16Mi}
      volumeMounts:
        - {name: run-artifacts, mountPath: /run-artifacts}
  volumes:
    - {name: run-artifacts, hostPath: {path: /mnt/k8s-disks/0/adai-kimi-k2-megatron-ep/${CAMPAIGN_ID}/${RUN_KEY}, type: DirectoryOrCreate}}
    - {name: shm, emptyDir: {medium: Memory, sizeLimit: 64Gi}}
    ${gdr_volume}
EOF
done

deadline=$((SECONDS + ${RUN_TIMEOUT_SECONDS:-7200}))
terminal=0
while (( SECONDS < deadline )); do
  pods_json="$("${KN[@]}" get pods -l app="${JOB}" -o json)"
  phases="$(jq -r '[.items[].status.phase] | group_by(.) | map({(.[0]): length}) | add // {}' <<<"${pods_json}")"
  trainer_states="$(jq -c '[.items[] | {pod:.metadata.name, state:([.status.containerStatuses[]? | select(.name=="trainer") | if .state.terminated then {terminated:true,exit_code:.state.terminated.exitCode} else {terminated:false} end][0] // {terminated:false})}]' <<<"${pods_json}")"
  echo "${phases}"
  echo "${trainer_states}"
  terminal="$(jq -r '[.[].state | select(.terminated)] | length' <<<"${trainer_states}")"
  [[ "${terminal}" -eq "${NNODES}" ]] && break
  sleep 15
done

# Freeze the snapshot tree before the final harvest and custody hash. Leaving
# the background logger alive here permits a last in-flight kubectl write to
# race with SHA256SUMS generation.
cleanup_snapshot
trap - EXIT

harvest_rank() {
  local rank="$1"
  local copied=0
  local attempt target
  "${KN[@]}" logs "${JOB}-${rank}" -c trainer --timestamps > "${LOCAL_RUN_DIR}/pod-logs/node-rank-${rank}.log" 2>&1 || true
  "${KN[@]}" logs "${JOB}-${rank}" -c harvester --timestamps > "${LOCAL_RUN_DIR}/pod-logs/harvester-node-rank-${rank}.log" 2>&1 || true
  for attempt in 1 2 3; do
    target="${LOCAL_RUN_DIR}/node-${rank}"
    if [[ -e "${target}" ]]; then
      mv "${target}" "${target}.incomplete-attempt-$((attempt - 1))"
    fi
    if "${KN[@]}" cp "${JOB}-${rank}:/run-artifacts/." "${target}" -c harvester 2>>"${LOCAL_RUN_DIR}/harvest-errors.log" &&
       [[ -s "${target}/custody.sha256" ]] &&
       (cd "${target}" && sha256sum -c custody.sha256) > "${LOCAL_RUN_DIR}/manifests/node-${rank}-custody-verify-attempt-${attempt}.log" 2>&1; then
      copied=1
      break
    fi
    printf 'HARVEST_RETRY node_rank=%d attempt=%d timestamp_utc=%s\n' "${rank}" "${attempt}" "$(date -u +%FT%TZ)" >> "${LOCAL_RUN_DIR}/harvest-errors.log"
  done
  [[ "${copied}" -eq 1 ]]
}

harvest_failures=0
for ((batch_start = 0; batch_start < NNODES; batch_start += HARVEST_PARALLELISM)); do
  harvest_pids=()
  batch_end=$((batch_start + HARVEST_PARALLELISM))
  if (( batch_end > NNODES )); then
    batch_end="${NNODES}"
  fi
  for ((rank = batch_start; rank < batch_end; rank++)); do
    harvest_rank "${rank}" &
    harvest_pids+=("$!")
  done
  for harvest_pid in "${harvest_pids[@]}"; do
    if ! wait "${harvest_pid}"; then
      harvest_failures=$((harvest_failures + 1))
    fi
  done
done
pods_after="$("${KN[@]}" get pods -l app="${JOB}" -o json)"
printf '%s\n' "${pods_after}" > "${LOCAL_RUN_DIR}/manifests/pods-after.json"
if [[ "${terminal}" -ne "${NNODES}" ]]; then status=TIMEOUT; exit_code=1
elif [[ "${harvest_failures}" -ne 0 ]]; then status=FAIL_HARVEST; exit_code=1
elif jq -e --argjson expected "${NNODES}" '(.items | length) == $expected and all(.items[]; any(.status.containerStatuses[]?; .name == "trainer" and .state.terminated.exitCode == 0))' <<<"${pods_after}" >/dev/null; then status=PASS; exit_code=0
else status=FAIL; exit_code=1
fi
printf '%s finished_at_utc=%s\n' "${status}" "$(date -u +%FT%TZ)" > "${LOCAL_RUN_DIR}/STATUS"
find "${LOCAL_RUN_DIR}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${LOCAL_RUN_DIR}/SHA256SUMS"
exit "${exit_code}"
