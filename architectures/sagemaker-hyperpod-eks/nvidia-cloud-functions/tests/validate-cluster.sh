#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# validate-cluster.sh -- Post-deployment end-to-end validation
#
# Runs comprehensive checks on the NVCF + SageMaker HyperPod EKS setup and optionally
# tests the sample echo function.
#
# This is a standalone test script that can be run independently of the
# setup scripts (00-04). It sources the cluster config if available.
#
# Usage:
#   chmod +x validate-cluster.sh
#   ./validate-cluster.sh [--test-function <function-id>]
# ---------------------------------------------------------------------------
set -euo pipefail

# Auto-source nvcf-config.env if it exists (centralised user configuration)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO_ROOT}/nvcf-config.env" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/nvcf-config.env"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

pass() { echo -e "  ${GREEN}[PASS]${NC} $*"; ((PASS++)); }
fail() { echo -e "  ${RED}[FAIL]${NC} $*"; ((FAIL++)); }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $*"; ((WARN++)); }

# ---------------------------------------------------------------------------
# Load cluster config (if available)
# ---------------------------------------------------------------------------
load_config() {
    local config_file
    config_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/infra/.cluster-config"

    if [[ -f "${config_file}" ]]; then
        # shellcheck disable=SC1090
        source "${config_file}"
        echo "  Cluster config loaded:"
        echo "    SageMaker HyperPod: ${HYPERPOD_CLUSTER_NAME:-unknown}"
        echo "    EKS:      ${EKS_CLUSTER_NAME:-unknown}"
        echo "    Region:   ${AWS_REGION:-unknown}"
    else
        echo "  No cluster config found at ${config_file}"
        echo "  (Run infra/scripts/00-discover-cluster.sh to generate it)"
        echo "  Proceeding with kubectl context only."
    fi
}

# ---------------------------------------------------------------------------
# Test 1: Kubernetes cluster health
# ---------------------------------------------------------------------------
test_k8s_health() {
    echo ""
    echo "=== Test 1: Kubernetes Cluster Health ==="

    if kubectl cluster-info &>/dev/null; then
        pass "Kubernetes API reachable"
    else
        fail "Cannot reach Kubernetes API"
        fail "Run: infra/scripts/01-prepare-cluster.sh to configure kubeconfig"
        return
    fi

    # Version check with NVCF compatibility warning
    local k8s_version
    k8s_version=$(kubectl version -o json 2>/dev/null | python3 -c "
import sys, json
v = json.load(sys.stdin)['serverVersion']
print(f\"{v['major']}.{v['minor']}\")
" 2>/dev/null || echo "unknown")

    if [[ "${k8s_version}" != "unknown" ]]; then
        local minor
        minor=$(echo "${k8s_version}" | cut -d. -f2 | tr -d '+')
        if [[ "${minor}" -gt 32 ]]; then
            warn "Kubernetes ${k8s_version} is above NVCF documented max (v1.32.x)"
        else
            pass "Kubernetes version: ${k8s_version} (within NVCF supported range)"
        fi
    else
        warn "Could not determine Kubernetes version"
    fi

    local nodes_ready
    nodes_ready=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready " || echo "0")
    if [[ "${nodes_ready}" -gt 0 ]]; then
        pass "Ready nodes: ${nodes_ready}"
    else
        fail "No ready nodes found"
    fi
}

# ---------------------------------------------------------------------------
# Test 2: GPU availability
# ---------------------------------------------------------------------------
test_gpu_availability() {
    echo ""
    echo "=== Test 2: GPU Availability ==="

    local gpu_nodes
    gpu_nodes=$(kubectl get nodes -l nvidia.com/gpu.present=true --no-headers 2>/dev/null | wc -l | tr -d ' ')

    if [[ "${gpu_nodes}" -gt 0 ]]; then
        pass "GPU nodes: ${gpu_nodes}"
    else
        fail "No GPU nodes detected"
        fail "Check: infra/scripts/01-prepare-cluster.sh --instance-group <name> --target-count 1"
        return
    fi

    # Check allocatable GPUs
    local total_gpus
    total_gpus=$(kubectl get nodes -l nvidia.com/gpu.present=true \
        -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null \
        | awk '{sum+=$1} END {print sum}')

    if [[ "${total_gpus}" -gt 0 ]]; then
        pass "Total allocatable GPUs: ${total_gpus}"
    else
        warn "Could not determine allocatable GPU count"
    fi
}

# ---------------------------------------------------------------------------
# Test 3: NVCA health
# ---------------------------------------------------------------------------
test_nvca_health() {
    echo ""
    echo "=== Test 3: NVCA Health ==="

    if ! kubectl get namespace nvca-operator &>/dev/null; then
        fail "nvca-operator namespace not found"
        fail "Run: infra/scripts/03-register-nvca.sh"
        return
    fi

    pass "nvca-operator namespace exists"

    local health
    health=$(kubectl get nvcfbackend -n nvca-operator --no-headers 2>/dev/null | awk '{print $NF}')

    if [[ "${health}" == "healthy" ]]; then
        pass "NVCA backend: healthy"
    elif [[ -z "${health}" ]]; then
        fail "NVCA backend: not found"
    else
        fail "NVCA backend: ${health}"
    fi

    # Check if nvca-system pods are running
    local nvca_running
    nvca_running=$(kubectl get pods -n nvca-system --no-headers 2>/dev/null | grep -c "Running" || echo "0")
    if [[ "${nvca_running}" -gt 0 ]]; then
        pass "NVCA system pods running: ${nvca_running}"
    else
        warn "No NVCA system pods running"
    fi
}

# ---------------------------------------------------------------------------
# Test 4: SageMaker HyperPod cluster status (via AWS API)
# ---------------------------------------------------------------------------
test_hyperpod_status() {
    echo ""
    echo "=== Test 4: SageMaker HyperPod Cluster Status ==="

    local cluster_name="${HYPERPOD_CLUSTER_NAME:-}"
    local region="${AWS_REGION:-us-west-2}"

    if [[ -z "${cluster_name}" ]]; then
        warn "HYPERPOD_CLUSTER_NAME not set. Skipping SageMaker HyperPod status check."
        warn "Run: infra/scripts/00-discover-cluster.sh to generate cluster config"
        return
    fi

    local status
    status=$(aws sagemaker describe-cluster --cluster-name "${cluster_name}" \
        --region "${region}" --query 'ClusterStatus' --output text 2>/dev/null || echo "ERROR")

    if [[ "${status}" == "InService" ]]; then
        pass "SageMaker HyperPod cluster '${cluster_name}': InService"
    else
        fail "SageMaker HyperPod cluster '${cluster_name}': ${status}"
    fi

    # Check GPU instance groups
    local gpu_info
    gpu_info=$(aws sagemaker describe-cluster --cluster-name "${cluster_name}" \
        --region "${region}" --output json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
gpu_prefixes = ['ml.g4','ml.g5','ml.g6','ml.p4','ml.p5']
for ig in data['InstanceGroups']:
    if any(ig['InstanceType'].startswith(p) for p in gpu_prefixes):
        print(f\"{ig['InstanceGroupName']}|{ig['InstanceType']}|{ig['CurrentCount']}|{ig['TargetCount']}\")
" 2>/dev/null || echo "")

    if [[ -n "${gpu_info}" ]]; then
        echo "${gpu_info}" | while IFS='|' read -r name itype current target; do
            if [[ "${current}" -gt 0 ]]; then
                pass "GPU group '${name}': ${itype} (${current}/${target} running)"
            else
                warn "GPU group '${name}': ${itype} (${current}/${target} -- scaled to zero)"
            fi
        done
    else
        warn "Could not retrieve GPU instance group info"
    fi
}

# ---------------------------------------------------------------------------
# Test 5: NVCF function invocation (optional)
# ---------------------------------------------------------------------------
test_function_invocation() {
    local function_id="${1:-}"
    local api_key="${NGC_API_KEY:-}"

    echo ""
    echo "=== Test 5: Function Invocation ==="

    if [[ -z "${function_id}" ]]; then
        warn "No function ID provided. Skipping invocation test."
        warn "Run with: --test-function <function-id>"
        return
    fi

    if [[ -z "${api_key}" ]]; then
        warn "NGC_API_KEY not set. Skipping invocation test."
        return
    fi

    local http_code
    http_code=$(curl --silent --output /dev/null --write-out "%{http_code}" \
        --location "https://${function_id}.invocation.api.nvcf.nvidia.com/echo" \
        --header 'Content-Type: application/json' \
        --header "Authorization: Bearer ${api_key}" \
        --data '{"message": "validation-test"}' \
        --max-time 60)

    if [[ "${http_code}" == "200" ]]; then
        pass "Function invocation returned HTTP 200"

        # Get the actual response
        local response
        response=$(curl --silent \
            --location "https://${function_id}.invocation.api.nvcf.nvidia.com/echo" \
            --header 'Content-Type: application/json' \
            --header "Authorization: Bearer ${api_key}" \
            --data '{"message": "validation-test"}' \
            --max-time 60)

        echo "  Response: ${response}"
        pass "Function invocation successful"
    elif [[ "${http_code}" == "202" ]]; then
        warn "Function returned HTTP 202 (request queued). Function may still be starting."
    else
        fail "Function invocation returned HTTP ${http_code}"
    fi
}

# ---------------------------------------------------------------------------
# Test 6: Resource overhead check
# ---------------------------------------------------------------------------
test_resource_overhead() {
    echo ""
    echo "=== Test 6: Node Resource Overhead ==="

    kubectl get nodes -l nvidia.com/gpu.present=true \
        -o custom-columns='NODE:.metadata.name,CPU:.status.allocatable.cpu,MEM:.status.allocatable.memory' \
        --no-headers 2>/dev/null | while read -r node cpu mem; do

        # Parse CPU (remove trailing 'm' if present)
        local cpu_val="${cpu%m}"
        if [[ "${cpu}" == *"m" ]]; then
            cpu_val=$((cpu_val / 1000))
        fi

        if [[ "${cpu_val}" -ge 8 ]]; then
            pass "${node}: CPU=${cpu} (comfortable headroom for NVCA)"
        elif [[ "${cpu_val}" -ge 6 ]]; then
            warn "${node}: CPU=${cpu} (>= 6 cores needed for NVCA, but tight)"
        else
            fail "${node}: CPU=${cpu} (< 6 cores, insufficient for NVCA overhead)"
        fi
    done || warn "Could not check node resources (no GPU nodes)"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
summary() {
    echo ""
    echo "========================================="
    echo "  Validation Results"
    echo "========================================="
    echo -e "  ${GREEN}PASS: ${PASS}${NC}"
    echo -e "  ${RED}FAIL: ${FAIL}${NC}"
    echo -e "  ${YELLOW}WARN: ${WARN}${NC}"
    echo "========================================="

    if [[ "${FAIL}" -gt 0 ]]; then
        echo ""
        echo -e "${RED}Validation FAILED. See docs/TROUBLESHOOTING.md for help.${NC}"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    local function_id=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --test-function)
                function_id="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    echo "========================================="
    echo "  End-to-End Cluster Validation"
    echo "  (NVCF on SageMaker HyperPod EKS)"
    echo "========================================="

    load_config
    test_k8s_health
    test_gpu_availability
    test_nvca_health
    test_hyperpod_status
    test_function_invocation "${function_id}"
    test_resource_overhead
    summary
}

main "$@"
