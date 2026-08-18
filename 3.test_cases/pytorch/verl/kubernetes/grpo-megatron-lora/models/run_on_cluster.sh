#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Run Model Download on FSx Utility Pod
# Dispatches model download scripts to the fsx-utils pod via kubectl exec,
# where the FSx filesystem is mounted at /fsx. This pod is independent of the
# Ray cluster, so you can download models before or after the cluster is running.
# =============================================================================
set -euo pipefail

# --- Script directory and env loading ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/../env_vars" ]; then
    source "${SCRIPT_DIR}/../env_vars"
fi

# --- Configuration ---
KUBE_NAMESPACE="${KUBE_NAMESPACE:-default}"
FSX_UTILS_POD="fsx-utils"
REMOTE_TMP_DIR="/tmp/model-download-$$"

# --- Usage ---
usage() {
    cat <<'USAGE'
Usage: ./models/run_on_cluster.sh <script> [args...]

Dispatches a model download script to the fsx-utils pod for execution, passing
any additional arguments through to it.
The fsx-utils pod has FSx mounted at /fsx, which is required for storing
model weights that all worker nodes can access.

Prerequisites:
  # Deploy the fsx-utils pod (one-time)
  source env_vars
  envsubst < kubernetes/fsx-utils.yaml | kubectl apply -f -
  kubectl wait --for=condition=Ready pod/fsx-utils -n ${KUBE_NAMESPACE} --timeout=180s

Examples:
  # Download a model to FSx (MODEL_NAME defaults to the repo name)
  ./models/run_on_cluster.sh models/download_model.sh Qwen/Qwen3-8B
  ./models/run_on_cluster.sh models/download_model.sh Qwen/Qwen3-235B-A22B

  # See docs/configuration.md ("Model download reference") for the model IDs
  # matching each conf/model/ group.

Environment variables:
  KUBE_NAMESPACE       Kubernetes namespace (default: from env_vars or "default")
  HF_TOKEN             HuggingFace token (passed to pod for model downloads)
USAGE
    exit 1
}

# --- Validate arguments ---
if [ $# -lt 1 ]; then
    usage
fi

LOCAL_SCRIPT="$1"
shift

if [ ! -f "${LOCAL_SCRIPT}" ]; then
    echo "ERROR: Script not found: ${LOCAL_SCRIPT}"
    exit 1
fi

# --- Determine script type ---
SCRIPT_BASENAME="$(basename "${LOCAL_SCRIPT}")"
SCRIPT_EXT="${SCRIPT_BASENAME##*.}"

if [ "${SCRIPT_EXT}" != "sh" ]; then
    echo "ERROR: Unsupported file type: .${SCRIPT_EXT} (expected .sh)"
    exit 1
fi

# --- Find the FSx utility pod ---
echo "=============================================="
echo "Run on Cluster: ${SCRIPT_BASENAME}"
echo "=============================================="
echo "Namespace:  ${KUBE_NAMESPACE}"
echo "Pod:        ${FSX_UTILS_POD}"
echo "=============================================="

# Check the pod exists and is running
POD_PHASE=$(kubectl get pod "${FSX_UTILS_POD}" -n "${KUBE_NAMESPACE}" \
    -o jsonpath='{.status.phase}' 2>/dev/null) || true

if [ -z "${POD_PHASE}" ]; then
    echo "ERROR: Pod '${FSX_UTILS_POD}' not found in namespace '${KUBE_NAMESPACE}'"
    echo ""
    echo "Deploy it first:"
    echo "  source env_vars"
    echo "  envsubst < kubernetes/fsx-utils.yaml | kubectl apply -f -"
    echo "  kubectl wait --for=condition=Ready pod/${FSX_UTILS_POD} -n \${KUBE_NAMESPACE} --timeout=180s"
    exit 1
fi

if [ "${POD_PHASE}" != "Running" ]; then
    echo "ERROR: Pod '${FSX_UTILS_POD}' is in phase '${POD_PHASE}' (expected 'Running')"
    echo ""
    echo "Check pod status:"
    echo "  kubectl describe pod ${FSX_UTILS_POD} -n ${KUBE_NAMESPACE}"
    exit 1
fi

echo "Pod phase:  ${POD_PHASE}"
echo "Script:     ${LOCAL_SCRIPT}"
echo "=============================================="

# --- Cleanup trap ---
cleanup() {
    echo ""
    echo "Cleaning up remote temp files..."
    kubectl exec -n "${KUBE_NAMESPACE}" "${FSX_UTILS_POD}" -- \
        rm -rf "${REMOTE_TMP_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

# --- Create remote temp directory ---
# Mirror the repo layout so scripts' relative env_vars sourcing works:
#   ${REMOTE_TMP_DIR}/env_vars            (test-case root level -- generated, not copied)
#   ${REMOTE_TMP_DIR}/models/<script>     (models/ subdirectory)
# Scripts do: SCRIPT_DIR="$(dirname ${BASH_SOURCE[0]})" -> ${REMOTE_TMP_DIR}/models
# Then:       source "${SCRIPT_DIR}/../env_vars"        -> ${REMOTE_TMP_DIR}/env_vars
kubectl exec -n "${KUBE_NAMESPACE}" "${FSX_UTILS_POD}" -- \
    mkdir -p "${REMOTE_TMP_DIR}/models"

# --- Generate a pod-safe env_vars file ---
# The local env_vars may contain commands (aws cli) that fail on the pod.
# Instead, we generate a minimal env_vars with only the variables the model
# download scripts need, sourced from the already-loaded local environment.
LOCAL_TMP_ENV=$(mktemp)
{
    echo "# Auto-generated env_vars for pod execution"
    echo "export RAY_DATA_HOME=\"${RAY_DATA_HOME:-/fsx/data/verl}\""
    echo "export HF_TOKEN=\"${HF_TOKEN:-}\""
} > "${LOCAL_TMP_ENV}"

echo "Generating pod-safe env_vars..."
kubectl cp "${LOCAL_TMP_ENV}" \
    "${KUBE_NAMESPACE}/${FSX_UTILS_POD}:${REMOTE_TMP_DIR}/env_vars"
rm -f "${LOCAL_TMP_ENV}"

# --- Copy the script to the pod ---
echo "Copying ${SCRIPT_BASENAME} to pod..."
kubectl cp "${LOCAL_SCRIPT}" \
    "${KUBE_NAMESPACE}/${FSX_UTILS_POD}:${REMOTE_TMP_DIR}/models/${SCRIPT_BASENAME}"

# --- Build the remote command ---
# We construct an inline script that:
#   1. Sources the generated env_vars (safe for pod execution)
#   2. Runs the target script, forwarding any remaining arguments
# Arguments are shell-quoted with printf %q so a model ID or path containing a
# space or glob character survives the trip through `kubectl exec -- bash -c`.
REMOTE_ARGS=""
if [ $# -gt 0 ]; then
    REMOTE_ARGS=" $(printf '%q ' "$@")"
fi

REMOTE_CMD="set -euo pipefail; "
REMOTE_CMD+="source ${REMOTE_TMP_DIR}/env_vars; "
REMOTE_CMD+="chmod +x ${REMOTE_TMP_DIR}/models/${SCRIPT_BASENAME}; "
REMOTE_CMD+="bash ${REMOTE_TMP_DIR}/models/${SCRIPT_BASENAME}${REMOTE_ARGS}"

# --- Execute on the pod ---
echo ""
echo "Executing on pod ${FSX_UTILS_POD}..."
echo "----------------------------------------------"

kubectl exec -n "${KUBE_NAMESPACE}" "${FSX_UTILS_POD}" -- \
    bash -c "${REMOTE_CMD}"

REMOTE_EXIT=$?

echo "----------------------------------------------"
if [ ${REMOTE_EXIT} -eq 0 ]; then
    echo "SUCCESS: ${SCRIPT_BASENAME} completed on pod ${FSX_UTILS_POD}"
else
    echo "FAILED: ${SCRIPT_BASENAME} exited with code ${REMOTE_EXIT}"
fi

exit ${REMOTE_EXIT}
