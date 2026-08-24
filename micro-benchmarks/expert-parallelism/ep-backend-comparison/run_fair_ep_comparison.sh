#!/usr/bin/env bash
set -euo pipefail

: "${CAMPAIGN_ID:?Set a unique CAMPAIGN_ID}"
: "${FAIR_EP_NODES:?Set 4 comma-separated B200 node names}"
: "${PROTECTED_NODES_CSV:?Set the concurrent campaign protected node names}"
: "${ARTIFACT_ROOT:?Set the durable artifact directory}"
: "${KUBECTL_CONTEXT:=aps1}"
: "${CAMPAIGN_NAMESPACE:=${CAMPAIGN_ID}}"
: "${SHARED_LOCK_NAME:=adai-ap-south-1-gpu-campaign-lock}"
: "${SHARED_LOCK_NAMESPACE:=default}"
: "${LOCK_MODE:=exclusive}"
: "${EXPECTED_LOCK_HOLDER:=}"
: "${LOCK_DURATION_SECONDS:=28800}"
: "${INDEPENDENT_STARTS:=3}"
: "${WARMUP_ITERATIONS:=20}"
: "${MEASURED_ITERATIONS:=100}"
: "${CASE_TIMEOUT_SECONDS:=1800}"
: "${EFA_PER_NODE:=8}"
: "${UCCL_IMAGE:=159553542841.dkr.ecr.ap-south-1.amazonaws.com/adai/dsv3-ep-backend-comparison-uccl@sha256:a703a0d35916bbd51b2d34d968626d9617cbceb328d64b920f21ad159d93c91a}"
: "${DEEPEP_V1_IMAGE:=159553542841.dkr.ecr.ap-south-1.amazonaws.com/adai/dsv3-ep-backend-comparison-deepep-v1-nvshmem@sha256:175a8470e11bc7832024941d1775923d5e8ecae2c6dc06183d5f1653d66b4fac}"
: "${DEEPEP_V2_IMAGE:=159553542841.dkr.ecr.ap-south-1.amazonaws.com/adai/dsv3-ep-backend-comparison-deepep-v2-gin-gda@sha256:4d07367ea290c5d6ec3c02b223ac819feed5240a46fd4a6492421e9c0853dbeb}"

[[ "${CAMPAIGN_ID}" =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]]
[[ "${CAMPAIGN_NAMESPACE}" =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]]
[[ "${LOCK_MODE}" == exclusive || "${LOCK_MODE}" == observe ]]
if [[ "${LOCK_MODE}" == observe && -z "${EXPECTED_LOCK_HOLDER}" ]]; then
    printf 'LOCK_MODE=observe requires EXPECTED_LOCK_HOLDER\n' >&2
    exit 2
fi
[[ "${INDEPENDENT_STARTS}" -eq 3 ]] || {
    printf 'This scored matrix requires exactly 3 independent starts\n' >&2
    exit 2
}

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${ARTIFACT_ROOT}/control" "${ARTIFACT_ROOT}/runs" \
    "${ARTIFACT_ROOT}/summary" "${ARTIFACT_ROOT}/teardown"
K=(kubectl --context "${KUBECTL_CONTEXT}")

IFS=, read -r -a selected_nodes <<<"${FAIR_EP_NODES}"
IFS=, read -r -a protected_nodes <<<"${PROTECTED_NODES_CSV}"
((${#selected_nodes[@]} == 4)) || {
    printf 'FAIR_EP_NODES must contain exactly 4 nodes\n' >&2
    exit 2
}
[[ "$(printf '%s\n' "${selected_nodes[@]}" | sort -u | wc -l)" -eq 4 ]]

declare -A protected=()
for node in "${protected_nodes[@]}"; do
    protected["${node}"]=1
done
for node in "${selected_nodes[@]}"; do
    [[ -z "${protected[${node}]:-}" ]] || {
        printf 'Selected node is protected by the concurrent campaign: %s\n' "${node}" >&2
        exit 1
    }
done

declare -A images=(
    [uccl]="${UCCL_IMAGE}"
    [deepep-v1-nvshmem]="${DEEPEP_V1_IMAGE}"
    [deepep-v2-gin-gda]="${DEEPEP_V2_IMAGE}"
)

current_case=""
namespace_created=0
lock_claimed=0

check_shared_lock() {
    local holder
    holder="$("${K[@]}" -n "${SHARED_LOCK_NAMESPACE}" get lease \
        "${SHARED_LOCK_NAME}" -o jsonpath='{.spec.holderIdentity}' 2>/dev/null || true)"
    if [[ "${LOCK_MODE}" == exclusive ]]; then
        [[ "${holder}" == "${CAMPAIGN_ID}" ]] || {
            printf 'Exclusive shared Lease is no longer held by %s: holder=%s\n' \
                "${CAMPAIGN_ID}" "${holder}" >&2
            return 1
        }
    elif [[ -n "${holder}" && "${holder}" != "${EXPECTED_LOCK_HOLDER}" ]]; then
        printf 'Shared Lease holder changed from protected campaign %s to %s\n' \
            "${EXPECTED_LOCK_HOLDER}" "${holder}" >&2
        return 1
    fi
}

claim_shared_lock() {
    local attempt current holder now candidate
    [[ "${LOCK_MODE}" == exclusive ]] || return 0
    for attempt in 1 2 3; do
        current="$("${K[@]}" -n "${SHARED_LOCK_NAMESPACE}" get lease \
            "${SHARED_LOCK_NAME}" -o json)"
        holder="$(jq -r '.spec.holderIdentity // ""' <<<"${current}")"
        if [[ "${holder}" == "${CAMPAIGN_ID}" ]]; then
            lock_claimed=1
            printf '%s\n' "${current}" >"${ARTIFACT_ROOT}/control/shared-lease-claimed.json"
            return 0
        fi
        [[ -z "${holder}" ]] || {
            printf 'Shared Lease is held by another campaign: %s\n' "${holder}" >&2
            return 1
        }
        now="$(date -u +%FT%T.000000Z)"
        candidate="$(jq \
            --arg holder "${CAMPAIGN_ID}" --arg now "${now}" \
            --argjson duration "${LOCK_DURATION_SECONDS}" '
                .spec.holderIdentity=$holder |
                .spec.acquireTime=$now |
                .spec.renewTime=$now |
                .spec.leaseDurationSeconds=$duration |
                .metadata.labels["adai.aws/campaign"]=$holder |
                .metadata.labels["adai.aws/owner"]="fair-ep-comparison"' \
            <<<"${current}")"
        if printf '%s\n' "${candidate}" | "${K[@]}" replace -f - \
            >"${ARTIFACT_ROOT}/control/shared-lease-claim-attempt-${attempt}.json" 2>&1; then
            lock_claimed=1
            "${K[@]}" -n "${SHARED_LOCK_NAMESPACE}" get lease "${SHARED_LOCK_NAME}" -o json \
                >"${ARTIFACT_ROOT}/control/shared-lease-claimed.json"
            return 0
        fi
    done
    printf 'Failed to claim shared Lease after 3 optimistic attempts\n' >&2
    return 1
}

release_shared_lock() {
    local current holder now
    ((lock_claimed == 1)) || return 0
    current="$("${K[@]}" -n "${SHARED_LOCK_NAMESPACE}" get lease \
        "${SHARED_LOCK_NAME}" -o json 2>/dev/null || true)"
    holder="$(jq -r '.spec.holderIdentity // ""' <<<"${current}")"
    [[ "${holder}" == "${CAMPAIGN_ID}" ]] || {
        printf 'Refusing to release shared Lease held by %s\n' "${holder}" >&2
        return 1
    }
    now="$(date -u +%FT%T.000000Z)"
    jq --arg now "${now}" '
        .spec.holderIdentity="" |
        .spec.renewTime=$now |
        .spec.leaseDurationSeconds=1 |
        del(.metadata.labels["adai.aws/campaign"], .metadata.labels["adai.aws/owner"])' \
        <<<"${current}" | "${K[@]}" replace -f - \
        >"${ARTIFACT_ROOT}/teardown/shared-lease-release.json"
    lock_claimed=0
}

gpu_requests_on_node() {
    local node="$1"
    "${K[@]}" get pods -A --field-selector "spec.nodeName=${node}" -o json | jq '
        [.items[]
         | select(.status.phase != "Succeeded" and .status.phase != "Failed")
         | [.spec.containers[]?.resources.requests["nvidia.com/gpu"] // 0 | tonumber]
         | add // 0]
        | add // 0'
}

verify_node_free() {
    local node="$1" ready instance_type gpu efa requests
    ready="$("${K[@]}" get node "${node}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')"
    instance_type="$("${K[@]}" get node "${node}" -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}')"
    gpu="$("${K[@]}" get node "${node}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"
    efa="$("${K[@]}" get node "${node}" -o jsonpath='{.status.allocatable.vpc\.amazonaws\.com/efa}')"
    requests="$(gpu_requests_on_node "${node}")"
    [[ "${ready}" == True && "${instance_type}" == p6-b200.48xlarge && \
       "${gpu}" == 8 && "${efa}" == 8 && "${requests}" -eq 0 ]] || {
        printf 'Node admission failed: node=%s ready=%s type=%s gpu=%s efa=%s requested_gpu=%s\n' \
            "${node}" "${ready}" "${instance_type}" "${gpu}" "${efa}" "${requests}" >&2
        return 1
    }
}

cleanup_case() {
    local case_to_delete cleanup_status=0
    [[ -n "${current_case}" ]] || return 0
    case_to_delete="${current_case}"
    "${K[@]}" -n "${CAMPAIGN_NAMESPACE}" delete statefulset "${case_to_delete}" \
        --ignore-not-found --wait=true --timeout=5m >/dev/null 2>&1 || cleanup_status=1
    # StatefulSet deletion can return before its cascading Pod deletions finish.
    # Wait for the GPU requests to disappear before admitting the next arm.
    "${K[@]}" -n "${CAMPAIGN_NAMESPACE}" wait --for=delete pod \
        -l "app=${case_to_delete}" --timeout=5m >/dev/null 2>&1 || cleanup_status=1
    "${K[@]}" -n "${CAMPAIGN_NAMESPACE}" delete service "${case_to_delete}" \
        --ignore-not-found >/dev/null 2>&1 || cleanup_status=1
    current_case=""
    return "${cleanup_status}"
}

finish() {
    local command_status=$? teardown_status=0 owned="" remaining=0
    trap - EXIT INT TERM
    set +e
    cleanup_case || teardown_status=1
    if ((namespace_created == 1)); then
        owned="$("${K[@]}" get namespace "${CAMPAIGN_NAMESPACE}" \
            -o jsonpath='{.metadata.labels.adai\.aws/campaign}' 2>/dev/null || true)"
        if [[ "${owned}" == "${CAMPAIGN_ID}" ]]; then
            "${K[@]}" delete namespace "${CAMPAIGN_NAMESPACE}" \
                --wait=true --timeout=10m >"${ARTIFACT_ROOT}/teardown/namespace-delete.log" 2>&1 || \
                teardown_status=1
        else
            printf 'Refusing to delete namespace without owned campaign label: %s\n' \
                "${CAMPAIGN_NAMESPACE}" >"${ARTIFACT_ROOT}/teardown/refused.txt"
            teardown_status=1
        fi
    fi
    release_shared_lock || teardown_status=1
    "${K[@]}" get all -A -l "adai.aws/campaign=${CAMPAIGN_ID}" -o json \
        >"${ARTIFACT_ROOT}/teardown/remaining-resources.json" 2>&1 || teardown_status=1
    remaining="$(jq '.items | length' "${ARTIFACT_ROOT}/teardown/remaining-resources.json" 2>/dev/null || printf '1')"
    [[ "${remaining}" -eq 0 ]] || teardown_status=1
    "${K[@]}" -n "${SHARED_LOCK_NAMESPACE}" get lease "${SHARED_LOCK_NAME}" -o json \
        >"${ARTIFACT_ROOT}/teardown/shared-lease-untouched.json" 2>&1 || true
    find "${ARTIFACT_ROOT}" -type f ! -name SHA256SUMS -print0 | sort -z | \
        xargs -0 sha256sum >"${ARTIFACT_ROOT}/SHA256SUMS"
    if ((command_status == 0 && teardown_status == 0)); then
        printf 'PASS\n' >"${ARTIFACT_ROOT}/STATUS"
    else
        printf 'FAIL command_status=%s_dimensionless teardown_status=%s_dimensionless remaining_resources=%s_resources\n' \
            "${command_status}" "${teardown_status}" "${remaining}" >"${ARTIFACT_ROOT}/STATUS"
    fi
    if ((command_status == 0 && teardown_status != 0)); then
        command_status=1
    fi
    exit "${command_status}"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${K[@]}" get nodes -o json >"${ARTIFACT_ROOT}/control/fleet-nodes-before.json"
"${K[@]}" get pods -A -o json >"${ARTIFACT_ROOT}/control/fleet-pods-before.json"
"${K[@]}" get namespaces -o json >"${ARTIFACT_ROOT}/control/namespaces-before.json"
"${K[@]}" -n "${SHARED_LOCK_NAMESPACE}" get lease "${SHARED_LOCK_NAME}" -o json \
    >"${ARTIFACT_ROOT}/control/shared-lease-before.json"
claim_shared_lock
check_shared_lock
aws sts get-caller-identity --output json >"${ARTIFACT_ROOT}/control/aws-caller-identity.json"
printf '%s\n' "${selected_nodes[@]}" >"${ARTIFACT_ROOT}/control/selected-nodes.txt"
printf '%s\n' "${protected_nodes[@]}" | sort -u >"${ARTIFACT_ROOT}/control/protected-nodes.txt"

for node in "${selected_nodes[@]}"; do
    verify_node_free "${node}"
done

"${K[@]}" create namespace "${CAMPAIGN_NAMESPACE}" --dry-run=client -o json | \
    jq --arg campaign "${CAMPAIGN_ID}" '
        .metadata.labels["adai.aws/campaign"]=$campaign |
        .metadata.labels["adai.aws/owner"]="fair-ep-comparison"' | \
    "${K[@]}" apply -f - >/dev/null
namespace_created=1

# Read the live host mitigation before invoking DeepEP V2.  The EFA 3.3.0g
# revalidation exposed a UVM HMM kernel panic, so a non-mitigated node is a hard
# admission failure rather than a benchmark attempt.
for index in 0 1 2 3; do
    node="${selected_nodes[${index}]}"
    "${K[@]}" -n "${CAMPAIGN_NAMESPACE}" apply -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: host-audit-${index}
  labels:
    adai.aws/campaign: ${CAMPAIGN_ID}
spec:
  nodeName: ${node}
  restartPolicy: Never
  tolerations:
    - {key: nvidia.com/gpu, operator: Exists}
    - {key: workload, operator: Exists}
    - {key: capacity-reservation, operator: Exists}
  containers:
    - name: audit
      image: public.ecr.aws/docker/library/busybox:1.36
      command: [/bin/sh, -lc]
      args:
        - |
          printf 'NODE=%s\n' "\${NODE_NAME}"
          printf 'UVM_DISABLE_HMM='; cat /host-sys/module/nvidia_uvm/parameters/uvm_disable_hmm
          printf 'EFA_KMOD='; cat /host-sys/module/efa/version
          printf 'GDRDRV='; if test -c /host-dev/gdrdrv; then echo character-device; else echo missing; fi
          sleep 300
      env:
        - name: NODE_NAME
          valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
      volumeMounts:
        - {name: host-sys, mountPath: /host-sys, readOnly: true}
        - {name: host-dev, mountPath: /host-dev, readOnly: true}
  volumes:
    - name: host-sys
      hostPath: {path: /sys, type: Directory}
    - name: host-dev
      hostPath: {path: /dev, type: Directory}
YAML
done
"${K[@]}" -n "${CAMPAIGN_NAMESPACE}" wait --for=condition=Ready pod \
    -l "adai.aws/campaign=${CAMPAIGN_ID}" --timeout=10m >/dev/null
for index in 0 1 2 3; do
    "${K[@]}" -n "${CAMPAIGN_NAMESPACE}" logs "host-audit-${index}" \
        >"${ARTIFACT_ROOT}/control/host-audit-${index}.log"
    rg -q '^UVM_DISABLE_HMM=(Y|1)$' "${ARTIFACT_ROOT}/control/host-audit-${index}.log" || {
        printf 'DeepEP V2 admission blocked by active UVM HMM on node %s\n' \
            "${selected_nodes[${index}]}" >&2
        exit 1
    }
    rg -q '^GDRDRV=character-device$' "${ARTIFACT_ROOT}/control/host-audit-${index}.log" || {
        printf 'Benchmark admission blocked by missing /dev/gdrdrv on node %s\n' \
            "${selected_nodes[${index}]}" >&2
        exit 1
    }
done
"${K[@]}" -n "${CAMPAIGN_NAMESPACE}" delete pod -l \
    "adai.aws/campaign=${CAMPAIGN_ID}" --wait=true --timeout=5m >/dev/null

"${K[@]}" -n "${CAMPAIGN_NAMESPACE}" create configmap fair-ep-scripts \
    --from-file=fair_ep_benchmark.py="${case_dir}/fair_ep_benchmark.py" \
    --from-file=run_fair_ep_rank.sh="${case_dir}/run_fair_ep_rank.sh" \
    --dry-run=client -o yaml | "${K[@]}" apply -f - >/dev/null

jq -n \
    --arg campaign_id "${CAMPAIGN_ID}" \
    --arg created_at_utc "$(date -u +%FT%TZ)" \
    --arg region ap-south-1 \
    --arg cluster ml-clusters-shared-ap-south-1 \
    --arg git_commit "$(git -C "${case_dir}" rev-parse HEAD)" \
    --arg uccl "${UCCL_IMAGE}" \
    --arg v1 "${DEEPEP_V1_IMAGE}" \
    --arg v2 "${DEEPEP_V2_IMAGE}" \
    --argjson warmups "${WARMUP_ITERATIONS}" \
    --argjson iterations "${MEASURED_ITERATIONS}" \
    --argjson starts "${INDEPENDENT_STARTS}" \
    '{campaign_id:$campaign_id,created_at_utc:$created_at_utc,region:$region,
      cluster:$cluster,git_commit:$git_commit,
      images:{uccl:$uccl,"deepep-v1-nvshmem":$v1,"deepep-v2-gin-gda":$v2},
      comparison:{tokens_per_rank:128,hidden_dimensions:7168,experts:256,
                  top_k_dimensionless:8,warmup_iterations:$warmups,
                  measured_iterations:$iterations,independent_starts:$starts}}' \
    >"${ARTIFACT_ROOT}/control/provenance.json"

run_case() {
    local arm="$1" world_size="$2" run_index="$3" dtype_order="$4"
    local warmups="${5:-${WARMUP_ITERATIONS}}" iterations="${6:-${MEASURED_ITERATIONS}}"
    local nccl_debug="${7:-WARN}" label="${8:-measurement}"
    local nodes=$((world_size / 8)) safe_arm="${arm//-}" node_values="" out=""
    current_case="fair-ep${world_size}-r${run_index}-${safe_arm:0:20}-${label}"
    current_case="${current_case:0:63}"
    out="${ARTIFACT_ROOT}/runs/ep${world_size}/${label}-repeat-${run_index}/${arm}"
    mkdir -p "${out}"

    check_shared_lock
    for ((index = 0; index < nodes; index++)); do
        verify_node_free "${selected_nodes[${index}]}"
        node_values+=$'\n                      - '"${selected_nodes[${index}]}"
    done
    "${K[@]}" get pods -A -o json >"${out}/pods-before.json"
    "${K[@]}" -n "${SHARED_LOCK_NAMESPACE}" get lease "${SHARED_LOCK_NAME}" -o json \
        >"${out}/shared-lease-before.json"

    "${K[@]}" -n "${CAMPAIGN_NAMESPACE}" apply -f - >/dev/null <<YAML
apiVersion: v1
kind: Service
metadata:
  name: ${current_case}
  labels: {adai.aws/campaign: ${CAMPAIGN_ID}}
spec:
  clusterIP: None
  publishNotReadyAddresses: true
  selector: {app: ${current_case}}
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: ${current_case}
  labels: {adai.aws/campaign: ${CAMPAIGN_ID}}
spec:
  serviceName: ${current_case}
  replicas: ${nodes}
  podManagementPolicy: Parallel
  selector:
    matchLabels: {app: ${current_case}}
  template:
    metadata:
      labels:
        app: ${current_case}
        adai.aws/campaign: ${CAMPAIGN_ID}
    spec:
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
      tolerations:
        - {key: nvidia.com/gpu, operator: Exists}
        - {key: workload, operator: Exists}
        - {key: capacity-reservation, operator: Exists}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: In
                    values:${node_values}
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector: {matchLabels: {app: ${current_case}}}
              topologyKey: kubernetes.io/hostname
      containers:
        - name: benchmark
          image: ${images[${arm}]}
          imagePullPolicy: IfNotPresent
          command: [/bin/bash, /opt/benchmark/run_fair_ep_rank.sh]
          securityContext: {privileged: true}
          env:
            - {name: EP_ARM, value: "${arm}"}
            - {name: EP_NODES, value: "${nodes}"}
            - {name: EP_SERVICE, value: "${current_case}"}
            - {name: EP_MASTER_ADDR, value: "${current_case}-0.${current_case}.${CAMPAIGN_NAMESPACE}.svc.cluster.local"}
            - name: POD_NAME
              valueFrom: {fieldRef: {fieldPath: metadata.name}}
            - {name: EP_RUN_INDEX, value: "${run_index}"}
            - {name: EP_DISPATCH_DTYPES, value: "${dtype_order}"}
            - {name: EP_WARMUPS, value: "${warmups}"}
            - {name: EP_ITERATIONS, value: "${iterations}"}
            - {name: EP_NCCL_DEBUG, value: "${nccl_debug}"}
            - {name: ADAI_IMAGE_REFERENCE, value: "${images[${arm}]}"}
          resources:
            requests: {nvidia.com/gpu: 8, vpc.amazonaws.com/efa: ${EFA_PER_NODE}}
            limits: {nvidia.com/gpu: 8, vpc.amazonaws.com/efa: ${EFA_PER_NODE}}
          volumeMounts:
            - {name: scripts, mountPath: /opt/benchmark/fair_ep_benchmark.py, subPath: fair_ep_benchmark.py}
            - {name: scripts, mountPath: /opt/benchmark/run_fair_ep_rank.sh, subPath: run_fair_ep_rank.sh}
            - {name: dshm, mountPath: /dev/shm}
            - {name: gdrdrv, mountPath: /dev/gdrdrv}
      volumes:
        - name: scripts
          configMap: {name: fair-ep-scripts, defaultMode: 0755}
        - name: dshm
          emptyDir: {medium: Memory, sizeLimit: 64Gi}
        - name: gdrdrv
          hostPath: {path: /dev/gdrdrv, type: CharDevice}
YAML
    "${K[@]}" -n "${CAMPAIGN_NAMESPACE}" get statefulset "${current_case}" -o yaml \
        >"${out}/statefulset.yaml"

    local deadline=$((SECONDS + CASE_TIMEOUT_SECONDS)) complete=0 pod=""
    while ((SECONDS < deadline)); do
        complete=0
        for ((index = 0; index < nodes; index++)); do
            pod="${current_case}-${index}"
            if "${K[@]}" -n "${CAMPAIGN_NAMESPACE}" logs "${pod}" --tail=30 2>/dev/null | \
                rg -q '^ADAI_FAIR_COMPLETE$'; then
                complete=$((complete + 1))
            elif "${K[@]}" -n "${CAMPAIGN_NAMESPACE}" logs "${pod}" --tail=80 2>/dev/null | \
                rg -q '^ADAI_FAIR_FAILED$'; then
                complete=-1
                break
            fi
        done
        ((complete == nodes || complete == -1)) && break
        sleep 10
    done

    for ((index = 0; index < nodes; index++)); do
        pod="${current_case}-${index}"
        "${K[@]}" -n "${CAMPAIGN_NAMESPACE}" logs "${pod}" >"${out}/${pod}.log" 2>&1 || true
        "${K[@]}" -n "${CAMPAIGN_NAMESPACE}" get pod "${pod}" -o yaml \
            >"${out}/${pod}.yaml" 2>&1 || true
        "${K[@]}" -n "${CAMPAIGN_NAMESPACE}" describe pod "${pod}" \
            >"${out}/${pod}-describe.txt" 2>&1 || true
    done
    ((complete == nodes)) || {
        printf 'Case failed or timed out: arm=%s EP%s repeat=%s label=%s complete=%s/%s\n' \
            "${arm}" "${world_size}" "${run_index}" "${label}" "${complete}" "${nodes}" >&2
        return 1
    }
    local rank_zero_log="${out}/${current_case}-0.log" result_count
    python3 "${case_dir}/extract_fair_results.py" \
        "${rank_zero_log}" "${out}/results.jsonl" >/dev/null
    result_count="$(wc -l <"${out}/results.jsonl")"
    [[ "${result_count}" -eq 2 ]] || {
        printf 'Expected 2 fair results in %s, found %s\n' "${rank_zero_log}" "${result_count}" >&2
        return 1
    }
    printf 'PASS arm=%s EP%s repeat=%s label=%s at %s UTC\n' \
        "${arm}" "${world_size}" "${run_index}" "${label}" "$(date -u +%FT%TZ)" \
        >"${out}/STATUS"
    cleanup_case
}

# A short V2-only admission run proves the HMM mitigation, GIN/GDAKI path, and
# common-harness correctness before any scored matrix work.
run_case deepep-v2-gin-gda 16 0 bf16,fp8 2 5 INFO admission
v2_admission_dir="${ARTIFACT_ROOT}/runs/ep16/admission-repeat-0/deepep-v2-gin-gda"
rg -q 'GDAKI.*createContext|gin GDAKI: createContext done' \
    "${v2_admission_dir}"/*.log || {
    printf 'DeepEP V2 admission completed without a GDAKI context proof\n' >&2
    exit 1
}
printf 'PASS\n' >"${v2_admission_dir}/GIN_ADMISSION_STATUS"

for world_size in 16 32; do
    for run_index in 1 2 3; do
        case "${run_index}" in
            1) order=(uccl deepep-v1-nvshmem deepep-v2-gin-gda); dtypes=fp8,bf16 ;;
            2) order=(deepep-v2-gin-gda uccl deepep-v1-nvshmem); dtypes=bf16,fp8 ;;
            3) order=(deepep-v1-nvshmem deepep-v2-gin-gda uccl); dtypes=fp8,bf16 ;;
        esac
        for arm in "${order[@]}"; do
            run_case "${arm}" "${world_size}" "${run_index}" "${dtypes}"
        done
    done
done

python3 "${case_dir}/summarize_fair_results.py" "${ARTIFACT_ROOT}/runs" \
    --starts="${INDEPENDENT_STARTS}" \
    --json="${ARTIFACT_ROOT}/summary/summary.json" \
    --markdown="${ARTIFACT_ROOT}/summary/summary.md"
"${K[@]}" get nodes -o json >"${ARTIFACT_ROOT}/control/fleet-nodes-after.json"
"${K[@]}" get pods -A -o json >"${ARTIFACT_ROOT}/control/fleet-pods-after.json"
touch "${ARTIFACT_ROOT}/CAMPAIGN_COMPLETE"
printf 'PASS fair EP comparison completed at %s UTC\n' "$(date -u +%FT%TZ)"
