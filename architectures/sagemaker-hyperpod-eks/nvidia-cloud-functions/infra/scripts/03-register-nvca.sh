#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 03-register-nvca.sh
#
# Registers the existing SageMaker HyperPod EKS cluster with NVIDIA Cloud
# Functions by installing the NVIDIA Cluster Agent (NVCA) operator.
#
# Before running this script, you must register the cluster in the NVCF UI
# (https://nvcf.ngc.nvidia.com > Settings > Register Cluster) and obtain:
#   1. The NGC Cluster Key
#   2. The NVCA Helm chart URL
#   3. The NCA ID (NVIDIA Cloud Account ID)
#   4. The Cluster ID
#
# The NVCF UI shows a Helm install command after registration. It contains
# all four values above. Example:
#   helm upgrade nvca-operator ... \
#     --set ngcConfig.serviceKey=$NGC_CLUSTER_KEY \
#     --set ncaID="Uuql..." \
#     --set clusterID="bc67f6b6-..."
#
# Prerequisites:
#   - kubeconfig configured (01-prepare-cluster.sh)
#   - GPU Operator installed (02-install-gpu-operator.sh)
#   - cluster-admin RBAC on the EKS cluster
#   - NGC account with Cloud Functions Admin role
#
# Required environment variables:
#   NGC_CLUSTER_KEY  - NGC Cluster Key from the NVCF registration UI
#   NVCA_HELM_URL    - NVCA operator Helm chart URL from the registration page
#   NVCA_NCA_ID      - NVIDIA Cloud Account ID from the registration page
#   NVCA_CLUSTER_ID  - Cluster ID from the registration page
#
# Usage:
#   export NGC_CLUSTER_KEY="your-cluster-key"
#   export NVCA_HELM_URL="https://helm.ngc.nvidia.com/nvidia/nvcf-byoc/charts/nvca-operator-X.Y.Z.tgz"
#   export NVCA_NCA_ID="your-nca-id"
#   export NVCA_CLUSTER_ID="your-cluster-id"
#   ./03-register-nvca.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# Auto-source nvcf-config.env if it exists (centralised user configuration)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "${REPO_ROOT}/nvcf-config.env" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/nvcf-config.env"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
heading() { echo -e "\n${CYAN}=== $* ===${NC}"; }

NVCA_NAMESPACE="nvca-operator"

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------
validate_env() {
    heading "Environment Validation"

    if [[ -z "${NGC_CLUSTER_KEY:-}" ]]; then
        error "NGC_CLUSTER_KEY is not set."
        echo ""
        echo "  To get the required values:"
        echo "    1. Go to https://nvcf.ngc.nvidia.com"
        echo "    2. Navigate to Settings > Register Cluster"
        echo "    3. Configure your cluster:"
        echo "       - Cluster Name: <your-name>"
        echo "       - Compute Platform: AWS"
        echo "       - Disable: Caching Support (not supported for AWS EKS)"
        echo "       - Enable: Dynamic GPU Discovery (default)"
        echo "    4. Click Save and Continue"
        echo "    5. Copy the values from the Helm install command shown"
        echo ""
        echo "  Then set all four environment variables:"
        echo "    export NGC_CLUSTER_KEY=\"<your-key>\""
        echo "    export NVCA_HELM_URL=\"<helm-chart-url>\""
        echo "    export NVCA_NCA_ID=\"<nca-id>\""
        echo "    export NVCA_CLUSTER_ID=\"<cluster-id>\""
        exit 1
    fi

    if [[ -z "${NVCA_HELM_URL:-}" ]]; then
        error "NVCA_HELM_URL is not set."
        echo ""
        echo "  The Helm chart URL is in the install command shown after"
        echo "  registering the cluster in the NVCF UI. It looks like:"
        echo "    https://helm.ngc.nvidia.com/nvidia/nvcf-byoc/charts/nvca-operator-X.Y.Z.tgz"
        echo ""
        echo "  Then: export NVCA_HELM_URL=\"https://helm.ngc.nvidia.com/...\""
        exit 1
    fi

    if [[ -z "${NVCA_NCA_ID:-}" ]]; then
        error "NVCA_NCA_ID is not set."
        echo ""
        echo "  The NCA ID (NVIDIA Cloud Account ID) is in the Helm install"
        echo "  command shown after registering the cluster. Look for:"
        echo "    --set ncaID=\"<value>\""
        echo ""
        echo "  Then: export NVCA_NCA_ID=\"<value>\""
        exit 1
    fi

    if [[ -z "${NVCA_CLUSTER_ID:-}" ]]; then
        error "NVCA_CLUSTER_ID is not set."
        echo ""
        echo "  The Cluster ID is in the Helm install command shown after"
        echo "  registering the cluster. Look for:"
        echo "    --set clusterID=\"<value>\""
        echo ""
        echo "  Then: export NVCA_CLUSTER_ID=\"<value>\""
        exit 1
    fi

    info "NGC_CLUSTER_KEY:  set (${#NGC_CLUSTER_KEY} chars)"
    info "NVCA_HELM_URL:    ${NVCA_HELM_URL}"
    info "NVCA_NCA_ID:      ${NVCA_NCA_ID}"
    info "NVCA_CLUSTER_ID:  ${NVCA_CLUSTER_ID}"
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
preflight() {
    heading "Pre-flight Checks"

    if ! kubectl cluster-info &>/dev/null; then
        error "Cannot connect to Kubernetes cluster."
        error "Run 01-prepare-cluster.sh first."
        exit 1
    fi
    info "Kubernetes cluster: reachable"

    if ! kubectl get namespace gpu-operator &>/dev/null; then
        error "GPU Operator namespace not found."
        error "Run 02-install-gpu-operator.sh first."
        exit 1
    fi
    info "GPU Operator: installed"

    if ! kubectl auth can-i '*' '*' --all-namespaces &>/dev/null; then
        error "Current user does not have cluster-admin permissions."
        error "NVCA installation requires cluster-admin RBAC."
        exit 1
    fi
    info "RBAC: cluster-admin confirmed"
}

# ---------------------------------------------------------------------------
# Install NVCA Operator
# ---------------------------------------------------------------------------
install_nvca() {
    heading "NVIDIA Cluster Agent Installation"

    # Key configuration notes for SageMaker HyperPod EKS:
    #   - ngcConfig.serviceKey is REQUIRED -- the chart uses it to create an
    #     image pull secret for pulling NVCA images from nvcr.io.
    #   - ncaID and clusterID are REQUIRED -- they link the NVCA operator to
    #     the cluster registered in the NVCF UI.
    #   - --username/--password authenticate to the Helm chart registry itself.
    #   - --reset-values ensures clean state on upgrade (matches NVCF UI command).
    #   - Caching is NOT supported on AWS EKS (do not enable CachingSupport)
    #   - Dynamic GPU Discovery is enabled by default
    #   - Function logs are NOT available on BYOC clusters
    #   - Only queue-depth autoscaling heuristic is supported on BYOC
    helm upgrade --install nvca-operator \
        --namespace "${NVCA_NAMESPACE}" \
        --create-namespace \
        --reset-values \
        --wait \
        --timeout 10m \
        "${NVCA_HELM_URL}" \
        --username='$oauthtoken' \
        --password="${NGC_CLUSTER_KEY}" \
        --set ngcConfig.serviceKey="${NGC_CLUSTER_KEY}" \
        --set ncaID="${NVCA_NCA_ID}" \
        --set clusterID="${NVCA_CLUSTER_ID}"

    info "NVCA Operator installed."
}

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
verify_nvca() {
    heading "Verification"

    info "Waiting for NVCA operator to become ready..."
    kubectl wait --for=condition=Ready pods \
        -l app.kubernetes.io/name=nvca-operator \
        -n "${NVCA_NAMESPACE}" \
        --timeout=300s 2>/dev/null || true

    echo ""
    info "NVCA Operator pods:"
    kubectl get pods -n "${NVCA_NAMESPACE}" -o wide 2>/dev/null

    echo ""
    info "NVCA Backend status:"
    kubectl get nvcfbackend -n "${NVCA_NAMESPACE}" 2>/dev/null || \
        warn "NVCFBackend resource not found yet (may take a few minutes)."

    if kubectl get namespace nvca-system &>/dev/null; then
        echo ""
        info "NVCA System pods:"
        kubectl get pods -n nvca-system -o wide 2>/dev/null
    fi
}

# ---------------------------------------------------------------------------
# Label GPU nodes for NVCF scheduling
# ---------------------------------------------------------------------------
label_gpu_nodes() {
    heading "GPU Node Labeling"

    local gpu_nodes
    gpu_nodes=$(kubectl get nodes -l nvidia.com/gpu.present=true -o name 2>/dev/null)

    if [[ -z "${gpu_nodes}" ]]; then
        warn "No GPU nodes found. Skipping labeling."
        warn "Once GPU nodes join, label them with:"
        warn "  kubectl label node <name> nvca.nvcf.nvidia.io/schedule=true"
        return
    fi

    echo "${gpu_nodes}" | while read -r node; do
        info "Labeling ${node} for NVCF scheduling..."
        kubectl label "${node}" nvca.nvcf.nvidia.io/schedule=true --overwrite
    done

    info "GPU nodes labeled."
}

# ---------------------------------------------------------------------------
# Post-install notes
# ---------------------------------------------------------------------------
post_install() {
    echo ""
    heading "Post-Installation Notes"
    echo ""
    info "1. CACHING: Disabled (not supported for AWS EKS)."
    info "2. NETWORK POLICIES: AWS VPC CNI may not enforce NetworkPolicy by default."
    info "   See docs/COMPATIBILITY-ANALYSIS.md for details."
    info "3. FUNCTION LOGS: Not supported on BYOC clusters."
    info "   Emit logs from your containers directly to CloudWatch."
    info "4. AUTOSCALING: Only queue-depth heuristic supported on BYOC."
    echo ""
    info "5. VERIFY: Check the NVCF UI at https://nvcf.ngc.nvidia.com"
    info "   Settings > Clusters -- your cluster should show as 'Ready'."
    echo ""
    info "Next step: ./04-validate-setup.sh"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo "========================================="
    echo "  NVIDIA Cluster Agent Registration"
    echo "  (Existing SageMaker HyperPod EKS Cluster)"
    echo "========================================="

    validate_env
    preflight
    install_nvca
    verify_nvca
    label_gpu_nodes
    post_install
}

main "$@"
