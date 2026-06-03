#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 02-install-gpu-operator.sh
#
# Installs the NVIDIA GPU Operator on an existing SageMaker HyperPod EKS
# cluster. This is a prerequisite for the NVCF Cluster Agent.
#
# Key configuration for SageMaker HyperPod:
#   - Driver installation is DISABLED (SageMaker HyperPod nodes have pre-installed
#     NVIDIA drivers; installing a second driver causes conflicts)
#   - Device plugin, DCGM exporter, and GPU Feature Discovery are enabled
#   - GPU Feature Discovery is required by NVCA's Dynamic GPU Discovery
#
# Prerequisites:
#   - kubeconfig configured for the EKS cluster (01-prepare-cluster.sh)
#   - helm installed
#   - At least one SageMaker HyperPod GPU node has joined the EKS cluster
#
# Usage:
#   ./02-install-gpu-operator.sh
#
# Environment variables:
#   GPU_OPERATOR_VERSION  - Helm chart version (default: v24.9.2)
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

GPU_OPERATOR_VERSION="${GPU_OPERATOR_VERSION:-v24.9.2}"
GPU_OPERATOR_NAMESPACE="gpu-operator"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
preflight() {
    heading "Pre-flight Checks"

    if ! command -v kubectl &>/dev/null; then
        error "kubectl not found. Run 01-prepare-cluster.sh first."
        exit 1
    fi

    if ! command -v helm &>/dev/null; then
        error "helm not found. Install from https://helm.sh/docs/intro/install/"
        exit 1
    fi

    if ! kubectl cluster-info &>/dev/null; then
        error "Cannot connect to Kubernetes cluster."
        error "Run 01-prepare-cluster.sh first to configure kubeconfig."
        exit 1
    fi

    local node_count
    node_count=$(kubectl get nodes --no-headers 2>/dev/null | wc -l | tr -d ' ')
    info "Nodes in cluster: ${node_count}"

    if [[ "${node_count}" == "0" ]]; then
        warn "No nodes found. GPU Operator will be installed but won't discover"
        warn "GPUs until SageMaker HyperPod nodes join the cluster."
    else
        # Show existing nodes and check for GPU instance types
        kubectl get nodes -o custom-columns='NAME:.metadata.name,INSTANCE:.metadata.labels.node\.kubernetes\.io/instance-type,STATUS:.status.conditions[-1].type' \
            --no-headers 2>/dev/null | while read -r line; do
            info "  ${line}"
        done
    fi

    info "Pre-flight checks passed."
}

# ---------------------------------------------------------------------------
# Install GPU Operator
# ---------------------------------------------------------------------------
install_gpu_operator() {
    heading "NVIDIA GPU Operator Installation"

    info "Adding NVIDIA Helm repository..."
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia 2>/dev/null || true
    helm repo update nvidia

    local action="install"
    if helm status gpu-operator -n "${GPU_OPERATOR_NAMESPACE}" &>/dev/null; then
        warn "GPU Operator already installed. Upgrading..."
        action="upgrade"
    else
        info "Installing GPU Operator ${GPU_OPERATOR_VERSION}..."
    fi

    # SageMaker HyperPod-specific configuration:
    #
    #   driver.enabled=false
    #     SageMaker HyperPod nodes come with pre-installed NVIDIA drivers. Installing
    #     a second driver via the operator causes conflicts and crashloops.
    #
    #   toolkit.enabled=true
    #     NVIDIA Container Toolkit is needed for GPU workloads in containers.
    #
    #   devicePlugin.enabled=true
    #     Required for Kubernetes to schedule GPU workloads.
    #
    #   dcgmExporter.enabled=true
    #     Exposes GPU metrics (utilization, temperature, memory).
    #
    #   gfd.enabled=true
    #     GPU Feature Discovery labels nodes with GPU product info.
    #     This is what NVCA uses for Dynamic GPU Discovery.
    #
    #   migManager.enabled=false
    #     MIG (Multi-Instance GPU) is not typically used with A10G/g5 instances.
    #
    #   vgpuManager.enabled=false
    #     Not applicable for bare-metal SageMaker HyperPod instances.
    #
    helm "${action}" gpu-operator nvidia/gpu-operator \
        --namespace "${GPU_OPERATOR_NAMESPACE}" \
        --create-namespace \
        --version "${GPU_OPERATOR_VERSION}" \
        --set driver.enabled=false \
        --set toolkit.enabled=true \
        --set devicePlugin.enabled=true \
        --set dcgmExporter.enabled=true \
        --set gfd.enabled=true \
        --set migManager.enabled=false \
        --set vgpuManager.enabled=false \
        --set node-feature-discovery.enabled=true \
        --wait \
        --timeout 10m

    info "GPU Operator ${action} complete."
}

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
verify() {
    heading "Verification"

    info "GPU Operator pods:"
    kubectl get pods -n "${GPU_OPERATOR_NAMESPACE}" -o wide 2>/dev/null || true

    echo ""
    local gpu_nodes
    gpu_nodes=$(kubectl get nodes -l nvidia.com/gpu.present=true --no-headers 2>/dev/null | wc -l | tr -d ' ')

    if [[ "${gpu_nodes}" -gt 0 ]]; then
        info "GPU nodes discovered: ${gpu_nodes}"
        kubectl get nodes -l nvidia.com/gpu.present=true \
            -o custom-columns='NAME:.metadata.name,GPU:.metadata.labels.nvidia\.com/gpu\.product,GPUS:.status.allocatable.nvidia\.com/gpu' \
            --no-headers 2>/dev/null | while read -r line; do
            info "  ${line}"
        done
    else
        warn "No GPU-labeled nodes found yet."
        warn "This is normal if SageMaker HyperPod GPU nodes are still joining."
        warn "GPU labels will appear once nodes are Ready and the GPU Operator runs."
        warn "Check with: kubectl get nodes -l nvidia.com/gpu.present=true"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo "========================================="
    echo "  NVIDIA GPU Operator Installation"
    echo "  (SageMaker HyperPod EKS - driver-skip mode)"
    echo "========================================="

    preflight
    install_gpu_operator
    verify

    echo ""
    info "Next step: ./03-register-nvca.sh"
}

main "$@"
