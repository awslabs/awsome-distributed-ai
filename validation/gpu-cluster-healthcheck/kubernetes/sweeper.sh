#!/usr/bin/env bash
# sweeper.sh -- CronJob entrypoint for rolling DCGM L2 sweep on idle GPU nodes
#
# Lists GPU nodes, skips those that are already tainted unhealthy or have
# GPU-consuming pods, applies a maintenance taint, creates a per-node Job
# running DCGM L2, and cleans up maintenance taints when done.
#
# Optional env:
#   MAX_NODES_PER_SWEEP   -- Max nodes to check per sweep (default: 10)
#   SWEEP_IMAGE            -- Container image for per-node jobs (required)
#   SWEEP_NAMESPACE        -- Namespace for sweep jobs (default: gpu-healthcheck)
#   JOB_TIMEOUT            -- Seconds to wait for each job (default: 900)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=../lib/common.sh
source "${BASE_DIR}/lib/common.sh"

# ─── Configuration ──────────────────────────────────────────────────────────
MAX_NODES_PER_SWEEP="${MAX_NODES_PER_SWEEP:-10}"
SWEEP_NAMESPACE="${SWEEP_NAMESPACE:-gpu-healthcheck}"
JOB_TIMEOUT="${JOB_TIMEOUT:-900}"

LABEL_PREFIX="gpu-healthcheck.aws-samples.io"
MAINTENANCE_TAINT="${LABEL_PREFIX}/maintenance=sweep:NoSchedule"

if [[ -z "${SWEEP_IMAGE:-}" ]]; then
    log_error "SWEEP_IMAGE environment variable is required"
    exit 1
fi

# ─── Helpers ────────────────────────────────────────────────────────────────

# Check if a node has GPU-consuming pods
node_has_gpu_pods() {
    local node="$1"
    local gpu_pods
    gpu_pods=$(kubectl get pods --all-namespaces --field-selector="spec.nodeName=${node}" \
        -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.resources.requests}{"\n"}{end}{end}' \
        2>/dev/null | grep -c "nvidia.com/gpu" || true)
    [[ "${gpu_pods}" -gt 0 ]]
}

# Check if a node already has the unhealthy taint
node_is_unhealthy() {
    local node="$1"
    kubectl get node "${node}" -o jsonpath='{.spec.taints}' 2>/dev/null \
        | grep -q "${LABEL_PREFIX}/unhealthy"
}

# Apply maintenance taint
add_maintenance_taint() {
    local node="$1"
    kubectl taint node "${node}" "${MAINTENANCE_TAINT}" 2>/dev/null || true
}

# Remove maintenance taint
remove_maintenance_taint() {
    local node="$1"
    kubectl taint node "${node}" "${MAINTENANCE_TAINT}-" 2>/dev/null || true
}

# Wait for a Job to reach Complete or Failed condition.
# kubectl wait --for=condition=complete does not detect Failed jobs, so we poll.
wait_for_job() {
    local job_name="$1"
    local namespace="$2"
    local timeout="$3"
    local elapsed=0
    local poll_interval=10

    while [[ ${elapsed} -lt ${timeout} ]]; do
        local status
        status=$(kubectl get job "${job_name}" -n "${namespace}" \
            -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null || true)

        if [[ "${status}" == *"Complete"* ]]; then
            log_info "Job ${job_name} completed successfully"
            return 0
        fi
        if [[ "${status}" == *"Failed"* ]]; then
            log_warn "Job ${job_name} failed"
            return 1
        fi

        sleep ${poll_interval}
        elapsed=$((elapsed + poll_interval))
    done

    log_warn "Job ${job_name} did not complete within ${timeout}s"
    return 1
}

# Create a per-node DCGM L2 Job
create_sweep_job() {
    local node="$1"
    local job_name="gpu-sweep-${node//\./-}"
    # Truncate job name to 63 chars (K8s label limit)
    job_name="${job_name:0:63}"

    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job_name}
  namespace: ${SWEEP_NAMESPACE}
  labels:
    app.kubernetes.io/name: gpu-healthcheck
    app.kubernetes.io/component: sweeper
    ${LABEL_PREFIX}/target-node: ${node}
spec:
  ttlSecondsAfterFinished: 3600
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        kubernetes.io/hostname: ${node}
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
        - key: ${LABEL_PREFIX}/maintenance
          operator: Exists
          effect: NoSchedule
      hostNetwork: true
      hostPID: true
      serviceAccountName: gpu-healthcheck
      containers:
        - name: dcgm-l2
          image: ${SWEEP_IMAGE}
          command: ["/bin/bash", "-c"]
          args:
            - |
              /opt/gpu-healthcheck/gpu-healthcheck.sh --check 1 --json
              exit_code=\$?
              # Patch node labels based on result
              if [[ \$exit_code -eq 0 ]]; then
                if ! kubectl label node ${node} ${LABEL_PREFIX}/status=pass --overwrite 2>&1; then
                  echo "[WARN] Failed to label node ${node} as pass" >&2
                fi
              else
                if ! kubectl label node ${node} ${LABEL_PREFIX}/status=fail --overwrite 2>&1; then
                  echo "[WARN] Failed to label node ${node} as fail" >&2
                fi
                kubectl taint node ${node} ${LABEL_PREFIX}/unhealthy=true:NoSchedule 2>&1 || \
                  echo "[WARN] Failed to taint node ${node}" >&2
              fi
              exit \$exit_code
          securityContext:
            privileged: true
          volumeMounts:
            - name: dev
              mountPath: /dev
            - name: proc-host
              mountPath: /proc
              readOnly: true
            - name: sys
              mountPath: /sys
              readOnly: true
      volumes:
        - name: dev
          hostPath:
            path: /dev
        - name: proc-host
          hostPath:
            path: /proc
        - name: sys
          hostPath:
            path: /sys
EOF

    echo "${job_name}"
}

# ─── Main ───────────────────────────────────────────────────────────────────

main() {
    log_info "GPU Health Check Sweeper starting"
    log_info "Max nodes per sweep: ${MAX_NODES_PER_SWEEP}"

    # Get list of GPU nodes
    local gpu_nodes
    gpu_nodes=$(kubectl get nodes -l "nvidia.com/gpu.present=true" \
        -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)

    if [[ -z "${gpu_nodes}" ]]; then
        log_info "No GPU nodes found -- nothing to sweep"
        exit 0
    fi

    local nodes_checked=0
    local job_names=()
    local tainted_nodes=()

    for node in ${gpu_nodes}; do
        if [[ ${nodes_checked} -ge ${MAX_NODES_PER_SWEEP} ]]; then
            log_info "Reached max nodes per sweep (${MAX_NODES_PER_SWEEP})"
            break
        fi

        # Skip unhealthy nodes
        if node_is_unhealthy "${node}"; then
            log_info "Skipping ${node} -- already tainted unhealthy"
            continue
        fi

        # Skip nodes with GPU-consuming pods
        if node_has_gpu_pods "${node}"; then
            log_info "Skipping ${node} -- has GPU-consuming pods"
            continue
        fi

        log_info "Sweeping node: ${node}"
        add_maintenance_taint "${node}"
        tainted_nodes+=("${node}")

        local job_name
        job_name=$(create_sweep_job "${node}")
        job_names+=("${job_name}")
        nodes_checked=$((nodes_checked + 1))
    done

    if [[ ${#job_names[@]} -eq 0 ]]; then
        log_info "No eligible nodes to sweep"
        exit 0
    fi

    log_info "Created ${#job_names[@]} sweep jobs, waiting for completion"

    # Wait for all jobs to complete or fail
    local failed=0
    for job_name in "${job_names[@]}"; do
        if ! wait_for_job "${job_name}" "${SWEEP_NAMESPACE}" "${JOB_TIMEOUT}"; then
            failed=$((failed + 1))
        fi
    done

    # Clean up maintenance taints
    for node in "${tainted_nodes[@]}"; do
        remove_maintenance_taint "${node}"
    done

    log_info "Sweeper complete: ${nodes_checked} nodes checked, ${failed} failed"
    exit ${failed}
}

main "$@"
