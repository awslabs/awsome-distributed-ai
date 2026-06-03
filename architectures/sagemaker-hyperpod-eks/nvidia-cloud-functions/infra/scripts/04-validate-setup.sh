#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 04-validate-setup.sh
#
# Validates the full NVCF + SageMaker HyperPod EKS setup. Run this after completing
# all setup steps to verify everything is working.
#
# Usage:
#   ./04-validate-setup.sh
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

PASS=0
FAIL=0
WARN=0

pass()    { echo -e "  ${GREEN}[PASS]${NC} $*"; PASS=$((PASS + 1)); }
fail()    { echo -e "  ${RED}[FAIL]${NC} $*"; FAIL=$((FAIL + 1)); }
warn()    { echo -e "  ${YELLOW}[WARN]${NC} $*"; WARN=$((WARN + 1)); }
heading() { echo -e "\n${CYAN}--- $* ---${NC}"; }

# ---------------------------------------------------------------------------
# 1. Kubernetes Cluster
# ---------------------------------------------------------------------------
check_k8s() {
    heading "Kubernetes Cluster"

    if kubectl cluster-info &>/dev/null; then
        pass "Connected to Kubernetes cluster"
    else
        fail "Cannot connect to Kubernetes cluster"
        return
    fi

    # Version
    local k8s_version
    k8s_version=$(kubectl version -o json 2>/dev/null | python3 -c "
import sys, json
v = json.load(sys.stdin)['serverVersion']
print(f\"{v['major']}.{v['minor']}\")
" 2>/dev/null || echo "unknown")

    if [[ "${k8s_version}" != "unknown" ]]; then
        pass "Kubernetes version: ${k8s_version}"
        # Check NVCF compatibility
        local minor
        minor=$(echo "${k8s_version}" | cut -d. -f2 | tr -d '+')
        if [[ "${minor}" -gt 32 ]]; then
            warn "K8s ${k8s_version} is above NVCF documented max (v1.32.x)"
        fi
    else
        warn "Could not determine Kubernetes version"
    fi

    # Nodes
    local node_count
    node_count=$(kubectl get nodes --no-headers 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${node_count}" -gt 0 ]]; then
        pass "Nodes found: ${node_count}"
    else
        fail "No nodes found in cluster"
    fi

    local ready_nodes
    ready_nodes=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready" || true)
    if [[ "${ready_nodes}" == "${node_count}" ]]; then
        pass "All ${node_count} nodes are Ready"
    else
        warn "Only ${ready_nodes}/${node_count} nodes are Ready"
    fi
}

# ---------------------------------------------------------------------------
# 2. GPU Operator
# ---------------------------------------------------------------------------
check_gpu_operator() {
    heading "NVIDIA GPU Operator"

    if kubectl get namespace gpu-operator &>/dev/null; then
        pass "gpu-operator namespace exists"
    else
        fail "gpu-operator namespace not found"
        return
    fi

    local running
    running=$(kubectl get pods -n gpu-operator --no-headers 2>/dev/null | grep -c "Running" || true)
    local total
    total=$(kubectl get pods -n gpu-operator --no-headers 2>/dev/null | wc -l | tr -d ' ')

    if [[ "${running}" -gt 0 ]]; then
        pass "GPU Operator pods running: ${running}/${total}"
    else
        fail "No GPU Operator pods running"
    fi

    # Check for driver pods (should NOT exist on SageMaker HyperPod)
    local driver_pods
    driver_pods=$(kubectl get pods -n gpu-operator -l app=nvidia-driver-daemonset --no-headers 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${driver_pods}" -gt 0 ]]; then
        warn "Driver daemonset pods found (${driver_pods}). These should not be present on SageMaker HyperPod."
        warn "Verify GPU Operator was installed with driver.enabled=false"
    fi

    # GPU nodes
    local gpu_nodes
    gpu_nodes=$(kubectl get nodes -l nvidia.com/gpu.present=true --no-headers 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${gpu_nodes}" -gt 0 ]]; then
        pass "GPU nodes discovered: ${gpu_nodes}"
        kubectl get nodes -l nvidia.com/gpu.present=true \
            -o custom-columns='NODE:.metadata.name,GPU:.metadata.labels.nvidia\.com/gpu\.product' \
            --no-headers 2>/dev/null | while read -r line; do
            pass "  ${line}"
        done
    else
        warn "No GPU nodes discovered yet (GPU nodes may still be joining)"
    fi
}

# ---------------------------------------------------------------------------
# 3. NVIDIA Cluster Agent (NVCA)
# ---------------------------------------------------------------------------
check_nvca() {
    heading "NVIDIA Cluster Agent (NVCA)"

    if kubectl get namespace nvca-operator &>/dev/null; then
        pass "nvca-operator namespace exists"
    else
        fail "nvca-operator namespace not found (NVCA not installed)"
        return
    fi

    local operator_pods
    operator_pods=$(kubectl get pods -n nvca-operator --no-headers 2>/dev/null | grep -c "Running" || true)
    if [[ "${operator_pods}" -gt 0 ]]; then
        pass "NVCA Operator pods running: ${operator_pods}"
    else
        fail "No NVCA Operator pods running"
    fi

    local backend_health
    backend_health=$(kubectl get nvcfbackend -n nvca-operator --no-headers 2>/dev/null | awk '{print $NF}' || echo "")
    if [[ "${backend_health}" == "healthy" ]]; then
        pass "NVCA Backend: healthy"
    elif [[ -z "${backend_health}" ]]; then
        warn "NVCA Backend: not available yet"
    else
        fail "NVCA Backend: ${backend_health}"
    fi

    local nvca_version
    nvca_version=$(kubectl get nvcfbackend -n nvca-operator --no-headers 2>/dev/null | awk '{print $3}' || echo "")
    if [[ -n "${nvca_version}" ]]; then
        pass "NVCA version: ${nvca_version}"
    fi

    if kubectl get namespace nvca-system &>/dev/null; then
        pass "nvca-system namespace exists"
        local nvca_pods
        nvca_pods=$(kubectl get pods -n nvca-system --no-headers 2>/dev/null | grep -c "Running" || true)
        if [[ "${nvca_pods}" -gt 0 ]]; then
            pass "NVCA system pods running: ${nvca_pods}"
        else
            warn "No NVCA system pods running yet"
        fi
    else
        warn "nvca-system namespace not found (agent may still be deploying)"
    fi
}

# ---------------------------------------------------------------------------
# 4. SageMaker HyperPod Specifics
# ---------------------------------------------------------------------------
check_hyperpod() {
    heading "SageMaker HyperPod"
    kubectl get nodes -o custom-columns='NODE:.metadata.name,INSTANCE:.metadata.labels.node\.kubernetes\.io/instance-type,STATUS:.status.conditions[-1].type' \
        --no-headers 2>/dev/null | while read -r line; do
        pass "  ${line}"
    done || warn "Could not read node info"

    # Check node resource overhead
    kubectl get nodes -l nvidia.com/gpu.present=true \
        -o custom-columns='NODE:.metadata.name,CPU:.status.allocatable.cpu,MEM:.status.allocatable.memory' \
        --no-headers 2>/dev/null | while read -r node cpu mem; do
        local cpu_val="${cpu%m}"
        if [[ "${cpu}" == *"m" ]]; then
            cpu_val=$((cpu_val / 1000))
        fi
        if [[ "${cpu_val}" -ge 6 ]]; then
            pass "${node}: CPU=${cpu} (>= 6 cores for NVCA overhead)"
        else
            warn "${node}: CPU=${cpu} (< 6 cores, resource contention likely)"
        fi
    done || true
}

# ---------------------------------------------------------------------------
# 5. VPC CNI / Network Policy
# ---------------------------------------------------------------------------
check_cni() {
    heading "VPC CNI / Network Policy"

    local cni_pods
    cni_pods=$(kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-node --no-headers 2>/dev/null | grep -c "Running" || true)
    if [[ "${cni_pods}" -gt 0 ]]; then
        pass "AWS VPC CNI pods running: ${cni_pods}"
    else
        warn "Could not verify VPC CNI pods"
    fi

    local np_pods
    np_pods=$(kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-network-policy-agent --no-headers 2>/dev/null | grep -c "Running" || true)
    if [[ "${np_pods}" -gt 0 ]]; then
        pass "Network Policy agent running: ${np_pods}"
    else
        warn "Network Policy agent not found."
        warn "NVCF network policies will NOT be enforced."
        warn "See docs/COMPATIBILITY-ANALYSIS.md for mitigation options."
    fi
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
summary() {
    echo ""
    echo "========================================="
    echo "  Validation Summary"
    echo "========================================="
    echo -e "  ${GREEN}PASS: ${PASS}${NC}"
    echo -e "  ${RED}FAIL: ${FAIL}${NC}"
    echo -e "  ${YELLOW}WARN: ${WARN}${NC}"
    echo "========================================="

    if [[ "${FAIL}" -gt 0 ]]; then
        echo ""
        echo -e "${RED}Some checks failed. See docs/TROUBLESHOOTING.md${NC}"
        exit 1
    elif [[ "${WARN}" -gt 0 ]]; then
        echo ""
        echo -e "${YELLOW}Warnings detected. Review items above.${NC}"
    else
        echo ""
        echo -e "${GREEN}All checks passed. Ready for NVCF function deployment.${NC}"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo "========================================="
    echo "  NVCF + SageMaker HyperPod EKS Validation"
    echo "========================================="

    check_k8s
    check_gpu_operator
    check_nvca
    check_hyperpod
    check_cni
    summary
}

main "$@"
