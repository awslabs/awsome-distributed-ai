#!/usr/bin/env bash
set -euo pipefail
usage() { echo "usage: $0 PATCHED_IMAGE@sha256 CONTROL_IMAGE@sha256 NODE1,...,NODE32" >&2; exit 2; }
[[ $# -eq 3 ]] || usage
PATCHED_IMAGE="$1"
CONTROL_IMAGE="$2"
NODE_CSV="$3"
[[ "${PATCHED_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]] || usage
[[ "${CONTROL_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]] || usage
: "${CTX:?set CTX}"
: "${CAMPAIGN_ID:?set CAMPAIGN_ID}"
: "${LOCAL_ARTIFACT_ROOT:?set LOCAL_ARTIFACT_ROOT}"
NS="${NS:-adai-kimi-k2-megatron-ep-${CAMPAIGN_ID,,}}"
CLUSTER_LOCK="${CLUSTER_LOCK:-adai-ap-south-1-gpu-campaign-lock}"
CLUSTER_LOCK_NAMESPACE="${CLUSTER_LOCK_NAMESPACE:-default}"
LOCK_HOLDER_ID="${LOCK_HOLDER_ID:-kimi-k2-megatron-ep-${CAMPAIGN_ID}}"
K=(kubectl --context "${CTX}" -n "${NS}")
IFS=, read -r -a NODES <<<"${NODE_CSV}"
[[ "${#NODES[@]}" -eq 32 ]] || usage
[[ "$(printf '%s\n' "${NODES[@]}" | sort -u | wc -l)" -eq 32 ]] || usage
OUT="${LOCAL_ARTIFACT_ROOT}/${CAMPAIGN_ID}/pr5-scale-regression"
mkdir -p "${OUT}"
CLEANUP_ENABLED=0
CURRENT_RESOURCE_NAME=""
cleanup() {
  if [[ "${CLEANUP_ENABLED}" -eq 1 ]]; then
    case "${CURRENT_RESOURCE_NAME}" in pr5-*) ;; "") ;; *) return 1 ;; esac
    if [[ -n "${CURRENT_RESOURCE_NAME}" ]]; then
      "${K[@]}" delete statefulset "${CURRENT_RESOURCE_NAME}" --wait=true --timeout=300s >/dev/null 2>&1 || true
      "${K[@]}" delete service "${CURRENT_RESOURCE_NAME}" --ignore-not-found >/dev/null 2>&1 || true
    fi
    "${K[@]}" delete configmap pr5-scripts --ignore-not-found >/dev/null 2>&1 || true
  fi
  find "${OUT}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${OUT}/SHA256SUMS"
}
trap cleanup EXIT
kubectl --context "${CTX}" get nodes "${NODES[@]}" -o json > "${OUT}/nodes-census.json"

ready=0
for node in "${NODES[@]}"; do
  type="$(kubectl --context "${CTX}" get node "${node}" -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}' 2>/dev/null || true)"
  state="$(kubectl --context "${CTX}" get node "${node}" -o jsonpath='{range .status.conditions[?(@.type=="Ready")]}{.status}{end}' 2>/dev/null || true)"
  used="$(kubectl --context "${CTX}" get pods -A -o json | jq --arg node "${node}" '[.items[] | select(.spec.nodeName==$node and (.status.phase=="Running" or .status.phase=="Pending")) | .spec.containers[]?.resources.requests["nvidia.com/gpu"] // "0" | tonumber] | add // 0')"
  [[ "${type}" = p6-b300.48xlarge && "${state}" = True && "${used}" -eq 0 ]] && ready=$((ready + 1))
done
if [[ "${ready}" -lt 32 ]]; then
  jq -n --arg timestamp "$(date -u +%FT%TZ)" --argjson ready "${ready}" \
    '{status:"NOT_RUN_INSUFFICIENT_CAPACITY",timestamp_utc:$timestamp,ready_free_b300_nodes:$ready,required_b300_nodes:32}' > "${OUT}/result.json"
  exit 0
fi

lock_holder="$(kubectl --context "${CTX}" -n "${CLUSTER_LOCK_NAMESPACE}" get lease "${CLUSTER_LOCK}" -o jsonpath='{.spec.holderIdentity}' 2>/dev/null || true)"
if [[ "${lock_holder}" != "${LOCK_HOLDER_ID}" ]]; then
  jq -n --arg timestamp "$(date -u +%FT%TZ)" --arg holder "${lock_holder}" --arg expected "${LOCK_HOLDER_ID}" \
    '{status:"NOT_RUN_CONCURRENT_CAMPAIGN",timestamp_utc:$timestamp,shared_lease_holder:$holder,required_holder:$expected}' > "${OUT}/result.json"
  exit 0
fi
namespace_owner="$(kubectl --context "${CTX}" get namespace "${NS}" -o jsonpath='{.metadata.labels.adai-campaign}' 2>/dev/null || true)"
if [[ "${namespace_owner}" != "${CAMPAIGN_ID}" ]]; then
  jq -n --arg timestamp "$(date -u +%FT%TZ)" --arg namespace "${NS}" --arg owner "${namespace_owner}" \
    '{status:"NOT_RUN_NAMESPACE_OWNERSHIP",timestamp_utc:$timestamp,namespace:$namespace,observed_owner:$owner}' > "${OUT}/result.json"
  exit 0
fi
CLEANUP_ENABLED=1

CASE_DIR="$(cd "$(dirname "$0")" && pwd)"
"${K[@]}" create configmap pr5-scripts \
  --from-file=pr5-runner.sh="${CASE_DIR}/pr5-runner.sh" \
  --from-file=pr5-gate.py="${CASE_DIR}/pr5-gate.py" \
  --dry-run=client -o yaml | "${K[@]}" apply -f - >/dev/null

run_case() {
  local case_name="$1" image="$2" domains="$3" expectation="$4"
  local case_out="${OUT}/${case_name}" name="pr5-${case_name}" values=""
  CURRENT_RESOURCE_NAME="${name}"
  mkdir -p "${case_out}/logs"
  lock_holder="$(kubectl --context "${CTX}" -n "${CLUSTER_LOCK_NAMESPACE}" get lease "${CLUSTER_LOCK}" -o jsonpath='{.spec.holderIdentity}' 2>/dev/null || true)"
  if [[ "${lock_holder}" != "${LOCK_HOLDER_ID}" ]]; then
    jq -n --arg case "${case_name}" --arg holder "${lock_holder}" --arg expected "${LOCK_HOLDER_ID}" \
      '{case:$case,status:"NOT_RUN_CONCURRENT_CAMPAIGN",shared_lease_holder:$holder,required_holder:$expected}' > "${case_out}/result.json"
    cp "${case_out}/result.json" "${OUT}/result.json"
    exit 0
  fi
  for node in "${NODES[@]:0:${domains}}"; do
    used="$(kubectl --context "${CTX}" get pods -A -o json | jq --arg node "${node}" '[.items[] | select(.spec.nodeName==$node and (.status.phase=="Running" or .status.phase=="Pending")) | .spec.containers[]?.resources.requests["nvidia.com/gpu"] // "0" | tonumber] | add // 0')"
    if [[ "${used}" -ne 0 ]]; then
      jq -n --arg case "${case_name}" --arg node "${node}" --argjson requested_gpus "${used}" \
        '{case:$case,status:"NOT_RUN_NODE_OCCUPANCY_CHANGED",occupied_node:$node,requested_gpus:$requested_gpus}' > "${case_out}/result.json"
      cp "${case_out}/result.json" "${OUT}/result.json"
      exit 0
    fi
    values+=$'\n                      - '"${node}"
  done
  cat <<EOF | "${K[@]}" apply -f - >/dev/null
apiVersion: v1
kind: Service
metadata: {name: ${name}, labels: {adai-campaign: "${CAMPAIGN_ID}"}}
spec: {clusterIP: None, publishNotReadyAddresses: true, selector: {app: ${name}}}
---
apiVersion: apps/v1
kind: StatefulSet
metadata: {name: ${name}, labels: {adai-campaign: "${CAMPAIGN_ID}"}}
spec:
  serviceName: ${name}
  replicas: ${domains}
  podManagementPolicy: Parallel
  selector: {matchLabels: {app: ${name}}}
  template:
    metadata: {labels: {app: ${name}, adai-campaign: "${CAMPAIGN_ID}"}}
    spec:
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
      tolerations: [{operator: Exists}]
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: In
                    values:${values}
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - {labelSelector: {matchLabels: {app: ${name}}}, topologyKey: kubernetes.io/hostname}
      containers:
        - name: gate
          image: ${image}
          command: [/bin/bash, /opt/benchmark/pr5-runner.sh]
          securityContext: {privileged: true}
          env:
            - {name: PR5_DOMAINS, value: "${domains}"}
            - {name: PR5_EXPECT, value: "${expectation}"}
            - {name: PR5_SERVICE, value: "${name}"}
            - {name: PR5_MASTER_ADDR, value: "${name}-0.${name}.${NS}.svc.cluster.local"}
            - name: POD_NAME
              valueFrom: {fieldRef: {fieldPath: metadata.name}}
          resources:
            requests: {nvidia.com/gpu: 8, vpc.amazonaws.com/efa: 16}
            limits: {nvidia.com/gpu: 8, vpc.amazonaws.com/efa: 16}
          volumeMounts:
            - {name: scripts, mountPath: /opt/benchmark/pr5-runner.sh, subPath: pr5-runner.sh}
            - {name: scripts, mountPath: /opt/benchmark/pr5-gate.py, subPath: pr5-gate.py}
            - {name: shm, mountPath: /dev/shm}
            - {name: gdrdrv, mountPath: /dev/gdrdrv}
      volumes:
        - {name: scripts, configMap: {name: pr5-scripts, defaultMode: 0755}}
        - {name: shm, emptyDir: {medium: Memory, sizeLimit: 64Gi}}
        - {name: gdrdrv, hostPath: {path: /dev/gdrdrv, type: CharDevice}}
EOF
  "${K[@]}" get statefulset "${name}" -o yaml > "${case_out}/statefulset.yaml"
  printf '%s\n' "${NODES[@]:0:${domains}}" > "${case_out}/nodes.txt"
  deadline=$((SECONDS + 7200)); passed=0; failed=0
  while (( SECONDS < deadline )); do
    passed=0; failed=0
    for index in $(seq 0 $((domains - 1))); do
      pod="${name}-${index}"
      "${K[@]}" logs "${pod}" --timestamps > "${case_out}/logs/${pod}.log" 2>&1 || true
      if grep -q ADAI_PR5_GATE_PASS "${case_out}/logs/${pod}.log"; then passed=$((passed + 1));
      elif grep -q ADAI_PR5_GATE_FAIL "${case_out}/logs/${pod}.log"; then failed=$((failed + 1)); fi
    done
    [[ "${failed}" -eq 0 && "${passed}" -ne "${domains}" ]] || break
    sleep 30
  done
  if [[ "${failed}" -eq 0 && "${passed}" -eq "${domains}" ]]; then status=PASS; else status=FAIL; fi
  jq -n --arg case "${case_name}" --arg status "${status}" --arg expectation "${expectation}" \
    --argjson domains "${domains}" --argjson passed "${passed}" --argjson failed "${failed}" \
    '{case:$case,status:$status,expectation:$expectation,physical_domains:$domains,ranks:($domains*8),passed_pods:$passed,failed_pods:$failed}' > "${case_out}/result.json"
  if ! "${K[@]}" delete statefulset "${name}" --wait=true --timeout=300s >/dev/null 2>&1; then
    exit 1
  fi
  if ! "${K[@]}" delete service "${name}" --ignore-not-found >/dev/null 2>&1; then
    exit 1
  fi
  CURRENT_RESOURCE_NAME=""
  [[ "${status}" = PASS ]]
}

overall=PASS
run_case control-22-domains "${CONTROL_IMAGE}" 22 pass || overall=FAIL
run_case control-23-domains "${CONTROL_IMAGE}" 23 refuse || overall=FAIL
run_case treatment-23-domains "${PATCHED_IMAGE}" 23 pass || overall=FAIL
run_case treatment-32-domains "${PATCHED_IMAGE}" 32 pass || overall=FAIL
jq -s --arg status "${overall}" '{status:$status,cases:.}' "${OUT}"/*/result.json > "${OUT}/result.json"
[[ "${overall}" = PASS ]]
