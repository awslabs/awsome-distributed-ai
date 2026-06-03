#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 00-discover-cluster.sh
#
# Discovers and validates an existing SageMaker HyperPod EKS cluster.
# This is the first script to run -- it identifies your cluster, checks
# its configuration, and reports what needs to be done before NVCF can
# be installed.
#
# Usage:
#   # Set HYPERPOD_CLUSTER_NAME in nvcf-config.env, or pass it directly:
#   ./00-discover-cluster.sh --cluster-name <hyperpod-cluster-name>
#
# If --cluster-name is not provided, the script reads HYPERPOD_CLUSTER_NAME
# from nvcf-config.env. One of the two must be set.
# ---------------------------------------------------------------------------
set -euo pipefail

# Auto-source nvcf-config.env if it exists (centralised user configuration)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "${REPO_ROOT}/nvcf-config.env" ]]; then
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

AWS_REGION="${AWS_REGION:-us-west-2}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CLUSTER_NAME="${HYPERPOD_CLUSTER_NAME:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster-name) CLUSTER_NAME="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Find SageMaker HyperPod clusters
# ---------------------------------------------------------------------------
discover_clusters() {
    heading "SageMaker HyperPod Clusters in ${AWS_REGION}"

    local clusters_json
    clusters_json=$(aws sagemaker list-clusters --region "${AWS_REGION}" --output json 2>&1)

    local count
    count=$(echo "${clusters_json}" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('ClusterSummaries',[])))" 2>/dev/null || echo "0")

    if [[ "${count}" == "0" ]]; then
        error "No SageMaker HyperPod clusters found in ${AWS_REGION}."
        error "This guide requires an existing SageMaker HyperPod EKS cluster."
        error "See: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks.html"
        exit 1
    fi

    echo "${clusters_json}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'  Found {len(data[\"ClusterSummaries\"])} cluster(s):')
print(f'  {\"Name\":<30} {\"Status\":<15} {\"Created\"}')
print(f'  {\"-\"*30} {\"-\"*15} {\"-\"*25}')
for c in data['ClusterSummaries']:
    print(f'  {c[\"ClusterName\"]:<30} {c[\"ClusterStatus\"]:<15} {c[\"CreationTime\"]}')
"

    # Cluster name is required
    if [[ -z "${CLUSTER_NAME}" ]]; then
        error "No cluster name provided."
        error "Specify one with --cluster-name or set HYPERPOD_CLUSTER_NAME in nvcf-config.env."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 2. Describe the cluster
# ---------------------------------------------------------------------------
describe_cluster() {
    heading "Cluster Details: ${CLUSTER_NAME}"

    local cluster_json
    cluster_json=$(aws sagemaker describe-cluster --cluster-name "${CLUSTER_NAME}" \
        --region "${AWS_REGION}" --output json 2>&1)

    local status
    status=$(echo "${cluster_json}" | python3 -c "import sys,json; print(json.load(sys.stdin)['ClusterStatus'])")

    if [[ "${status}" != "InService" ]]; then
        error "Cluster status is '${status}', expected 'InService'."
        exit 1
    fi
    info "Cluster status: ${status}"

    # Extract EKS cluster ARN
    local eks_arn
    eks_arn=$(echo "${cluster_json}" | python3 -c "import sys,json; print(json.load(sys.stdin)['Orchestrator']['Eks']['ClusterArn'])")
    EKS_CLUSTER_NAME=$(echo "${eks_arn}" | awk -F'/' '{print $NF}')
    info "EKS cluster: ${EKS_CLUSTER_NAME}"

    # Show instance groups
    echo ""
    echo "${cluster_json}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('  Instance Groups:')
print(f'  {\"Group\":<35} {\"Type\":<22} {\"Current/Target\":<15} {\"Status\"}')
print(f'  {\"-\"*35} {\"-\"*22} {\"-\"*15} {\"-\"*12}')
for ig in data['InstanceGroups']:
    ct = f\"{ig['CurrentCount']}/{ig['TargetCount']}\"
    has_gpu = any(x in ig['InstanceType'] for x in ['g5','g6','p4','p5','g4'])
    marker = ' (GPU)' if has_gpu else ''
    print(f'  {ig[\"InstanceGroupName\"]:<35} {ig[\"InstanceType\"]:<22} {ct:<15} {ig[\"Status\"]}{marker}')
"

    # Identify GPU groups with 0 instances
    echo ""
    local gpu_groups_at_zero
    gpu_groups_at_zero=$(echo "${cluster_json}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
gpu_prefixes = ['ml.g4','ml.g5','ml.g6','ml.p4','ml.p5']
groups = []
for ig in data['InstanceGroups']:
    if any(ig['InstanceType'].startswith(p) for p in gpu_prefixes) and ig['TargetCount'] == 0:
        groups.append(f\"{ig['InstanceGroupName']} ({ig['InstanceType']})\")
if groups:
    for g in groups:
        print(g)
" 2>/dev/null)

    if [[ -n "${gpu_groups_at_zero}" ]]; then
        warn "GPU instance groups scaled to 0:"
        echo "${gpu_groups_at_zero}" | while read -r line; do
            warn "  ${line}"
        done
        warn "You will need to scale up at least one GPU group before installing NVCF."
        warn "Run: ./01-prepare-cluster.sh --cluster-name ${CLUSTER_NAME}"
    fi

    # Check VPC / networking
    heading "Network Configuration"
    local vpc_info
    vpc_info=$(echo "${cluster_json}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
vpc = data.get('VpcConfig', {})
sgs = vpc.get('SecurityGroupIds', [])
subnets = vpc.get('Subnets', [])
print(f'  Security Groups: {sgs}')
print(f'  Subnets: {subnets}')
")
    echo "${vpc_info}"

    # Check if subnets have NAT Gateway (by checking route tables)
    local subnet_id
    subnet_id=$(echo "${cluster_json}" | python3 -c "import sys,json; print(json.load(sys.stdin)['VpcConfig']['Subnets'][0])")

    local route_table
    route_table=$(aws ec2 describe-route-tables \
        --filters "Name=association.subnet-id,Values=${subnet_id}" \
        --region "${AWS_REGION}" --output json 2>/dev/null)

    local has_nat
    has_nat=$(echo "${route_table}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for rt in data.get('RouteTables', []):
    for route in rt.get('Routes', []):
        if route.get('NatGatewayId'):
            print('yes')
            sys.exit(0)
print('no')
" 2>/dev/null || echo "unknown")

    if [[ "${has_nat}" == "yes" ]]; then
        info "NAT Gateway: Found (required for NVCF outbound connectivity)"
    elif [[ "${has_nat}" == "no" ]]; then
        error "NAT Gateway: NOT FOUND"
        error "NVCF requires outbound internet access to NVIDIA control plane endpoints."
        error "Add a NAT Gateway to the VPC before proceeding."
    else
        warn "NAT Gateway: Could not determine (check manually)"
    fi
}

# ---------------------------------------------------------------------------
# 3. Describe the EKS cluster
# ---------------------------------------------------------------------------
describe_eks() {
    heading "EKS Cluster: ${EKS_CLUSTER_NAME}"

    local eks_json
    eks_json=$(aws eks describe-cluster --name "${EKS_CLUSTER_NAME}" \
        --region "${AWS_REGION}" --output json 2>&1)

    local eks_status k8s_version auth_mode
    eks_status=$(echo "${eks_json}" | python3 -c "import sys,json; print(json.load(sys.stdin)['cluster']['status'])")
    k8s_version=$(echo "${eks_json}" | python3 -c "import sys,json; print(json.load(sys.stdin)['cluster']['version'])")
    auth_mode=$(echo "${eks_json}" | python3 -c "import sys,json; print(json.load(sys.stdin)['cluster'].get('accessConfig',{}).get('authenticationMode','unknown'))")

    info "EKS status: ${eks_status}"
    info "Kubernetes version: ${k8s_version}"
    info "Auth mode: ${auth_mode}"

    # Check K8s version against NVCF supported range
    if python3 -c "
v = '${k8s_version}'.split('.')
major, minor = int(v[0]), int(v[1])
if minor > 32:
    print('ABOVE_MAX')
elif minor < 25:
    print('BELOW_MIN')
else:
    print('OK')
" 2>/dev/null | grep -q "ABOVE_MAX"; then
        warn "Kubernetes ${k8s_version} is above NVCF's documented max (v1.32.x)."
        warn "The NVCA operator may still work but is not officially tested on ${k8s_version}."
        warn "Consider testing as-is or downgrading the EKS cluster to 1.32."
    else
        info "Kubernetes version is within NVCF supported range (v1.25-v1.32)."
    fi

    # Check auth mode
    if [[ "${auth_mode}" == "API" || "${auth_mode}" == "API_AND_CONFIG_MAP" ]]; then
        info "Auth mode compatible with SageMaker HyperPod (${auth_mode})."
    else
        warn "Auth mode '${auth_mode}' may not be supported by SageMaker HyperPod (needs API or API_AND_CONFIG_MAP)."
    fi
}

# ---------------------------------------------------------------------------
# 4. Check quotas for GPU instances
# ---------------------------------------------------------------------------
check_quotas() {
    heading "GPU Instance Quotas (Cluster Usage)"

    local quotas_json
    quotas_json=$(aws service-quotas list-service-quotas \
        --service-code sagemaker \
        --region "${AWS_REGION}" \
        --query "Quotas[?contains(QuotaName, 'cluster usage') && (contains(QuotaName, 'g5') || contains(QuotaName, 'g6') || contains(QuotaName, 'p4') || contains(QuotaName, 'p5'))]" \
        --output json 2>/dev/null)

    echo "${quotas_json}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
# Filter to non-spot cluster quotas
relevant = [q for q in data if 'spot' not in q['QuotaName'].lower()]
relevant.sort(key=lambda x: x['QuotaName'])
print(f'  {\"Instance Type\":<35} {\"Quota\":<10}')
print(f'  {\"-\"*35} {\"-\"*10}')
for q in relevant:
    name = q['QuotaName'].replace(' for cluster usage', '')
    val = int(q['Value'])
    marker = ' <-- ZERO' if val == 0 else ''
    print(f'  {name:<35} {val:<10}{marker}')
" 2>/dev/null || warn "Could not retrieve quota information."
}

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
print_summary() {
    heading "Summary & Next Steps"

    echo ""
    info "Cluster:     ${CLUSTER_NAME}"
    info "EKS Cluster: ${EKS_CLUSTER_NAME}"
    info "Region:      ${AWS_REGION}"
    echo ""
    info "Next steps:"
    info "  1. Run ./01-prepare-cluster.sh to scale up GPU nodes and configure kubeconfig"
    info "  2. Run ./02-install-gpu-operator.sh to install the NVIDIA GPU Operator"
    info "  3. Run ./03-register-nvca.sh to register with NVIDIA Cloud Functions"
    info "  4. Run ./04-validate-setup.sh to verify the full setup"
    echo ""

    # Write discovered values to a config file for other scripts to use
    local config_file
    config_file="$(dirname "$0")/../.cluster-config"
    cat > "${config_file}" <<EOF
# Auto-generated by 00-discover-cluster.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
HYPERPOD_CLUSTER_NAME="${CLUSTER_NAME}"
EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME}"
AWS_REGION="${AWS_REGION}"
EOF
    info "Cluster config saved to: ${config_file}"
    info "Other scripts will read this file automatically."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo "========================================="
    echo "  SageMaker HyperPod EKS Cluster Discovery"
    echo "  (for NVIDIA Cloud Functions)"
    echo "========================================="

    discover_clusters
    describe_cluster
    describe_eks
    check_quotas
    print_summary
}

main "$@"
