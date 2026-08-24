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
NS="${NS:-adai-kimi-k2-megatron-ep-${CAMPAIGN_ID,,}}"
case "${NS}" in adai-kimi-k2-megatron-ep-*) ;; *) echo "refusing non-campaign namespace ${NS}" >&2; exit 2;; esac

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
GPUS_PER_NODE=8
WORLD=$((NNODES * GPUS_PER_NODE))
for node in "${NODES[@]}"; do
  actual="$("${K[@]}" get node "${node}" -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}')"
  [[ "${actual}" = "${INSTANCE_TYPE}" ]] || { echo "node ${node} is ${actual}, expected ${INSTANCE_TYPE}" >&2; exit 3; }
  ready="$("${K[@]}" get node "${node}" -o jsonpath='{range .status.conditions[?(@.type=="Ready")]}{.status}{end}')"
  [[ "${ready}" = True ]] || { echo "node ${node} is not Ready" >&2; exit 3; }
  occupied="$("${K[@]}" get pods -A -o json | jq --arg n "${node}" '[.items[] | select(.spec.nodeName==$n and (.status.phase=="Running" or .status.phase=="Pending")) | .spec.containers[]?.resources.requests["nvidia.com/gpu"] // "0" | tonumber] | add // 0')"
  [[ "${occupied}" -eq 0 ]] || { echo "node ${node} already has ${occupied} requested GPUs" >&2; exit 3; }
done

CELL="${CELL:-qualification}"
REPEAT="${REPEAT:-1}"
RUN_KEY="${CELL}/repeat-${REPEAT}/${EP_ARM}"
LOCAL_RUN_DIR="${LOCAL_ARTIFACT_ROOT}/${CAMPAIGN_ID}/26.08/kimi-k2/${RUN_KEY}"
mkdir -p "${LOCAL_RUN_DIR}/pod-logs" "${LOCAL_RUN_DIR}/snapshots" "${LOCAL_RUN_DIR}/manifests"
[[ ! -e "${LOCAL_RUN_DIR}/STATUS" ]] || { echo "refusing completed run ${LOCAL_RUN_DIR}" >&2; exit 4; }
"${K[@]}" get nodes "${NODES[@]}" -o json > "${LOCAL_RUN_DIR}/manifests/nodes-before.json"
"${K[@]}" get pods -A -o json > "${LOCAL_RUN_DIR}/manifests/pods-before.json"
cp "${ARMS_FILE}" "${LOCAL_RUN_DIR}/manifests/arms.yaml"
sha256sum "${SELF_DIR}/kimi-k2/benchmarks/bench_kimi_k2_pretrain.py" > "${LOCAL_RUN_DIR}/manifests/model-config.sha256"

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
JOB="${JOB:0:62}"
PORT=23456
TP="${TENSOR_PARALLEL:-8}"
PP="${PIPELINE_PARALLEL:-8}"
EP="${EXPERT_PARALLEL:-32}"
TRAIN_ITERS="${TRAIN_ITERS:-40}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
MICRO_BATCH="${MICRO_BATCH:-4}"
SEQ_LEN="${SEQ_LEN:-4096}"
MOE_A2A_OVERLAP="${MOE_A2A_OVERLAP:-off}"

cat > "${LOCAL_RUN_DIR}/environment.txt" <<EOF
campaign_id=${CAMPAIGN_ID}
cell=${CELL}
repeat=${REPEAT}
ep_arm=${EP_ARM}
image=${IMAGE}
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
expected_dispatcher=${EXPECTED_DISPATCHER}
expected_backend=${EXPECTED_BACKEND}
EOF

cat <<EOF | "${KN[@]}" apply -f -
apiVersion: v1
kind: Service
metadata: {name: ${JOB}, labels: {adai-campaign: "${CAMPAIGN_ID}"}}
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
      "${KN[@]}" logs "${JOB}-${rank}" --timestamps > "${LOCAL_RUN_DIR}/snapshots/${stamp}-node-rank-${rank}.log" 2>&1 || true
    done
    sleep 30
  done
}
snapshot &
SNAPSHOT_PID=$!
cleanup_snapshot() { kill "${SNAPSHOT_PID}" >/dev/null 2>&1 || true; wait "${SNAPSHOT_PID}" >/dev/null 2>&1 || true; }
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
    profile_prefix='nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --output=/run-artifacts/nsys-node-rank-0'
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
  labels: {app: ${JOB}, adai-campaign: "${CAMPAIGN_ID}", rank: "${rank}"}
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
          python3 /opt/benchmark/case/bench/collect-runtime-manifest.py /run-artifacts/runtime-manifest.json;
          /opt/benchmark/case/bench/collect-node-telemetry.sh /run-artifacts/telemetry & telemetry_pid=\$!;
          ${profile_prefix} torchrun --nnodes=${NNODES} --nproc-per-node=${GPUS_PER_NODE} --node-rank=${rank}
          --master-addr=${JOB}-0.${JOB}.${NS}.svc.cluster.local --master-port=${PORT}
          /opt/benchmark/case/kimi-k2/benchmarks/bench_kimi_k2_pretrain.py
          2>&1 | tee /run-artifacts/node-rank-${rank}.log;
          train_rc=\${PIPESTATUS[0]};
          python3 /opt/benchmark/case/bench/summarize-route-trace.py /run-artifacts/router-trace /run-artifacts/route-summary.json || true;
          kill \${telemetry_pid} >/dev/null 2>&1 || true; wait \${telemetry_pid} || true; exit \${train_rc}
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
        - {name: ROUTER_TRACE_DIR, value: "/run-artifacts/router-trace"}
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
  volumes:
    - {name: run-artifacts, hostPath: {path: /mnt/k8s-disks/0/adai-kimi-k2-megatron-ep/${CAMPAIGN_ID}/${RUN_KEY}, type: DirectoryOrCreate}}
    - {name: shm, emptyDir: {medium: Memory, sizeLimit: 64Gi}}
    ${gdr_volume}
EOF
done

deadline=$((SECONDS + ${RUN_TIMEOUT_SECONDS:-7200}))
terminal=0
while (( SECONDS < deadline )); do
  phases="$("${KN[@]}" get pods -l app="${JOB}" -o json | jq -r '[.items[].status.phase] | group_by(.) | map({(.[0]): length}) | add // {}')"
  echo "${phases}"
  terminal="$(jq -r '(.Succeeded // 0) + (.Failed // 0)' <<<"${phases}")"
  [[ "${terminal}" -eq "${NNODES}" ]] && break
  sleep 15
done

for rank in $(seq 0 $((NNODES - 1))); do
  "${KN[@]}" logs "${JOB}-${rank}" --timestamps > "${LOCAL_RUN_DIR}/pod-logs/node-rank-${rank}.log" 2>&1 || true
  "${KN[@]}" cp "${JOB}-${rank}:/run-artifacts/." "${LOCAL_RUN_DIR}/node-${rank}" 2>>"${LOCAL_RUN_DIR}/harvest-errors.log" || true
done
"${KN[@]}" get pods -l app="${JOB}" -o json > "${LOCAL_RUN_DIR}/manifests/pods-after.json"
if [[ "${terminal}" -ne "${NNODES}" ]]; then status=TIMEOUT; exit_code=1
elif "${KN[@]}" get pods -l app="${JOB}" -o json | jq -e 'all(.items[]; .status.phase == "Succeeded")' >/dev/null; then status=PASS; exit_code=0
else status=FAIL; exit_code=1
fi
printf '%s finished_at_utc=%s\n' "${status}" "$(date -u +%FT%TZ)" > "${LOCAL_RUN_DIR}/STATUS"
find "${LOCAL_RUN_DIR}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${LOCAL_RUN_DIR}/SHA256SUMS"
exit "${exit_code}"
