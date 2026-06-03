#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 01-prepare-cluster.sh
#
# Prepares an existing SageMaker HyperPod EKS cluster for NVIDIA Cloud
# Functions by:
#   1. Loading cluster config from discovery (00-discover-cluster.sh)
#   2. Checking/installing local CLI tools
#   3. Configuring kubeconfig for the EKS cluster
#   4. Scaling up a GPU instance group (if needed)
#   5. Verifying outbound network connectivity
#
# Prerequisites:
#   - Run 00-discover-cluster.sh first (generates .cluster-config)
#   - AWS CLI configured with appropriate permissions
#
# Usage:
#   ./01-prepare-cluster.sh [--instance-group <name>] [--target-count <n>]
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../.cluster-config"

# Defaults
INSTANCE_GROUP=""
TARGET_COUNT=1

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --instance-group) INSTANCE_GROUP="$2"; shift 2 ;;
        --target-count)   TARGET_COUNT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Load cluster config
# ---------------------------------------------------------------------------
load_config() {
    if [[ -f "${CONFIG_FILE}" ]]; then
        # shellcheck disable=SC1090
        source "${CONFIG_FILE}"
        info "Loaded cluster config:"
        info "  SageMaker HyperPod cluster: ${HYPERPOD_CLUSTER_NAME}"
        info "  EKS cluster:      ${EKS_CLUSTER_NAME}"
        info "  Region:            ${AWS_REGION}"
    else
        error "Cluster config not found at ${CONFIG_FILE}"
        error "Run 00-discover-cluster.sh first."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 1. Check CLI tools
# ---------------------------------------------------------------------------
check_tools() {
    heading "CLI Tools"

    local tools=("aws" "kubectl" "helm" "docker")
    local tool_urls=(
        "https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        "https://kubernetes.io/docs/tasks/tools/"
        "https://helm.sh/docs/intro/install/"
        "https://docs.docker.com/get-docker/"
    )

    for i in "${!tools[@]}"; do
        if command -v "${tools[$i]}" &>/dev/null; then
            local version
            case "${tools[$i]}" in
                kubectl) version=$(kubectl version --client 2>&1 | head -1) ;;
                helm)    version=$(helm version --short 2>&1 | head -1) ;;
                *)       version=$("${tools[$i]}" --version 2>&1 | head -1) ;;
            esac
            info "${tools[$i]}: ${version}"
        else
            if [[ "${tools[$i]}" == "docker" ]]; then
                warn "${tools[$i]} not found (needed for building function containers)"
                warn "  Install: ${tool_urls[$i]}"
            else
                error "${tools[$i]} not found."
                error "  Install: ${tool_urls[$i]}"
                exit 1
            fi
        fi
    done

    # NGC CLI is optional at this stage
    if command -v ngc &>/dev/null; then
        info "ngc: $(ngc --version 2>/dev/null || echo 'installed')"
    else
        warn "NGC CLI not found. Needed later for function deployment."
        warn "  Install: https://ngc.nvidia.com/setup/installers/cli"
    fi
}

# ---------------------------------------------------------------------------
# 2. Configure kubeconfig
# ---------------------------------------------------------------------------
configure_kubeconfig() {
    heading "Kubeconfig"

    info "Configuring kubeconfig for EKS cluster: ${EKS_CLUSTER_NAME}..."
    aws eks update-kubeconfig --name "${EKS_CLUSTER_NAME}" --region "${AWS_REGION}" 2>&1

    if kubectl cluster-info &>/dev/null; then
        info "Connected to Kubernetes cluster."

        local node_count
        node_count=$(kubectl get nodes --no-headers 2>/dev/null | wc -l | tr -d ' ')
        info "Nodes in cluster: ${node_count}"

        if [[ "${node_count}" -gt 0 ]]; then
            kubectl get nodes -o custom-columns='NAME:.metadata.name,STATUS:.status.conditions[-1].type,INSTANCE:.metadata.labels.node\.kubernetes\.io/instance-type' \
                --no-headers 2>/dev/null | while read -r line; do
                info "  ${line}"
            done
        fi
    else
        error "Failed to connect to EKS cluster. Check IAM permissions."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 3. Scale up GPU instance group
# ---------------------------------------------------------------------------
scale_gpu_group() {
    heading "GPU Instance Group"

    # Get current instance groups
    local cluster_json
    cluster_json=$(aws sagemaker describe-cluster \
        --cluster-name "${HYPERPOD_CLUSTER_NAME}" \
        --region "${AWS_REGION}" --output json)

    # Find GPU instance groups
    local gpu_groups
    gpu_groups=$(echo "${cluster_json}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
gpu_prefixes = ['ml.g4','ml.g5','ml.g6','ml.p4','ml.p5']
for ig in data['InstanceGroups']:
    if any(ig['InstanceType'].startswith(p) for p in gpu_prefixes):
        print(f\"{ig['InstanceGroupName']}|{ig['InstanceType']}|{ig['CurrentCount']}|{ig['TargetCount']}\")
" 2>/dev/null)

    if [[ -z "${gpu_groups}" ]]; then
        error "No GPU instance groups found in the cluster."
        error "The cluster needs at least one GPU instance group for NVCF."
        exit 1
    fi

    info "GPU instance groups:"
    echo "${gpu_groups}" | while IFS='|' read -r name itype current target; do
        local status_msg=""
        if [[ "${current}" == "0" && "${target}" == "0" ]]; then
            status_msg=" (scaled to zero)"
        fi
        info "  ${name}: ${itype} (current: ${current}, target: ${target})${status_msg}"
    done

    # If a specific group was requested, use it; otherwise pick the first GPU group
    if [[ -z "${INSTANCE_GROUP}" ]]; then
        INSTANCE_GROUP=$(echo "${gpu_groups}" | head -1 | cut -d'|' -f1)
        info "Auto-selected GPU group: ${INSTANCE_GROUP}"
    fi

    # Get current target for the selected group
    local current_target
    current_target=$(echo "${gpu_groups}" | grep "^${INSTANCE_GROUP}|" | cut -d'|' -f4)
    local current_count
    current_count=$(echo "${gpu_groups}" | grep "^${INSTANCE_GROUP}|" | cut -d'|' -f3)

    if [[ -z "${current_target}" ]]; then
        error "Instance group '${INSTANCE_GROUP}' not found."
        exit 1
    fi

    if [[ "${current_count}" -ge "${TARGET_COUNT}" ]]; then
        info "Instance group '${INSTANCE_GROUP}' already has ${current_count} running instance(s). No scaling needed."
        return
    fi

    if [[ "${current_target}" -ge "${TARGET_COUNT}" ]]; then
        info "Instance group '${INSTANCE_GROUP}' already targets ${current_target} instance(s). Waiting for nodes to come up."
        return
    fi

    # Build the full update payload (must include ALL instance groups)
    info "Scaling '${INSTANCE_GROUP}' from ${current_target} to ${TARGET_COUNT} instance(s)..."

    local update_payload
    update_payload=$(echo "${cluster_json}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
groups = []
for ig in data['InstanceGroups']:
    group = {
        'InstanceGroupName': ig['InstanceGroupName'],
        'InstanceType': ig['InstanceType'],
        'InstanceCount': ig['TargetCount'],
        'LifeCycleConfig': ig['LifeCycleConfig'],
        'ExecutionRole': ig['ExecutionRole'],
    }
    if ig.get('InstanceStorageConfigs'):
        group['InstanceStorageConfigs'] = ig['InstanceStorageConfigs']
    if ig['InstanceGroupName'] == '${INSTANCE_GROUP}':
        group['InstanceCount'] = ${TARGET_COUNT}
    groups.append(group)
print(json.dumps(groups))
")

    aws sagemaker update-cluster \
        --cluster-name "${HYPERPOD_CLUSTER_NAME}" \
        --instance-groups "${update_payload}" \
        --region "${AWS_REGION}" \
        --output json >/dev/null

    info "Scale-up request submitted. Waiting for nodes to join the EKS cluster..."
    info "This can take 5-15 minutes. Monitor with:"
    info "  kubectl get nodes -w"
    info "  aws sagemaker describe-cluster --cluster-name ${HYPERPOD_CLUSTER_NAME} --region ${AWS_REGION}"
}

# ---------------------------------------------------------------------------
# 4. Verify outbound network connectivity
# ---------------------------------------------------------------------------
check_outbound() {
    heading "Outbound Network Connectivity"

    info "Checking if nodes can reach NVIDIA endpoints..."
    info "(This requires at least one Running node in the cluster)"

    local node_count
    node_count=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready" || echo "0")

    if [[ "${node_count}" == "0" ]]; then
        warn "No Ready nodes found. Skipping outbound connectivity check."
        warn "Re-run after nodes have joined the cluster."
        return
    fi

    # Quick DNS check using an existing node
    local endpoints=("api.ngc.nvidia.com" "nvcr.io" "helm.ngc.nvidia.com")
    for ep in "${endpoints[@]}"; do
        if kubectl run "dns-check-$$" \
            --image=busybox:1.36 \
            --restart=Never \
            --rm -i \
            --timeout=30s \
            -- nslookup "${ep}" &>/dev/null 2>&1; then
            info "DNS OK: ${ep}"
        else
            warn "Could not verify DNS for ${ep} (may need NAT Gateway)"
        fi
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo "========================================="
    echo "  Prepare SageMaker HyperPod EKS for NVCF"
    echo "========================================="

    load_config
    check_tools
    configure_kubeconfig
    scale_gpu_group
    check_outbound

    echo ""
    heading "Done"
    info "Next step: ./02-install-gpu-operator.sh"
}

main "$@"
