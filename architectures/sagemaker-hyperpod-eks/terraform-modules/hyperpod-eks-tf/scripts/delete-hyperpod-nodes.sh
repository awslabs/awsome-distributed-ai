#!/bin/bash
#
# delete-hyperpod-nodes.sh
#
# Deletes specific nodes from a SageMaker HyperPod instance group by EC2 instance ID,
# then updates terraform.tfvars to keep Terraform state in sync.
#
# This script wraps the BatchDeleteClusterNodes API, which is the only way to remove
# specific instances from a HyperPod cluster. The standard UpdateCluster API (used by
# Terraform's awscc_sagemaker_cluster resource) only accepts a target instance_count
# and lets AWS choose which nodes to remove.
#
# Usage:
#   ./delete-hyperpod-nodes.sh \
#     --cluster-name ml-cluster \
#     --instance-group-name workers \
#     --node-ids i-0abc1234def56789a,i-0def5678abc12345b \
#     --tfvars-file /path/to/terraform.tfvars \
#     [--region us-west-2] \
#     [--auto-apply] \
#     [--yes]
#
# Prerequisites:
#   - AWS CLI v2
#   - jq
#   - Appropriate IAM permissions for sagemaker:BatchDeleteClusterNodes,
#     sagemaker:DescribeCluster, and sagemaker:ListClusterNodes
#
set -euo pipefail

# ─── Constants ────────────────────────────────────────────────────────────────

STABILIZE_TIMEOUT_SECONDS=300
STABILIZE_POLL_INTERVAL=10

# ─── Colors ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ─── Logging ──────────────────────────────────────────────────────────────────

log_info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ─── Usage ────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Delete specific nodes from a SageMaker HyperPod instance group and update
terraform.tfvars to keep Terraform state in sync.

Required:
  --cluster-name NAME         Name or ARN of the HyperPod cluster
  --instance-group-name NAME  Name of the instance group containing the nodes
  --node-ids IDS              Comma-separated list of EC2 instance IDs (i-xxx)
  --tfvars-file PATH          Path to terraform.tfvars to update after deletion

Optional:
  --region REGION             AWS region (defaults to AWS_DEFAULT_REGION or CLI config)
  --auto-apply                Run 'terraform apply -auto-approve' after patching tfvars
  --yes                       Skip confirmation prompt
  -h, --help                  Show this help message

Examples:
  # Delete a single node
  $(basename "$0") \\
    --cluster-name ml-cluster \\
    --instance-group-name workers \\
    --node-ids i-0abc1234def56789a \\
    --tfvars-file /path/to/terraform.tfvars

  # Delete multiple nodes with auto-apply
  $(basename "$0") \\
    --cluster-name ml-cluster \\
    --instance-group-name workers \\
    --node-ids i-0abc1234def56789a,i-0def5678abc12345b \\
    --tfvars-file /path/to/terraform.tfvars \\
    --auto-apply --yes
EOF
    exit "${1:-0}"
}

# ─── Argument Parsing ─────────────────────────────────────────────────────────

CLUSTER_NAME=""
INSTANCE_GROUP_NAME=""
NODE_IDS_RAW=""
REGION=""
TFVARS_FILE=""
AUTO_APPLY=false
SKIP_CONFIRM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster-name)
            CLUSTER_NAME="$2"; shift 2 ;;
        --instance-group-name)
            INSTANCE_GROUP_NAME="$2"; shift 2 ;;
        --node-ids)
            NODE_IDS_RAW="$2"; shift 2 ;;
        --region)
            REGION="$2"; shift 2 ;;
        --tfvars-file)
            TFVARS_FILE="$2"; shift 2 ;;
        --auto-apply)
            AUTO_APPLY=true; shift ;;
        --yes)
            SKIP_CONFIRM=true; shift ;;
        -h|--help)
            usage 0 ;;
        *)
            log_error "Unknown option: $1"
            usage 1 ;;
    esac
done

# ─── Validate Required Arguments ─────────────────────────────────────────────

if [[ -z "$CLUSTER_NAME" ]]; then
    log_error "Missing required argument: --cluster-name"
    usage 1
fi

if [[ -z "$INSTANCE_GROUP_NAME" ]]; then
    log_error "Missing required argument: --instance-group-name"
    usage 1
fi

if [[ -z "$NODE_IDS_RAW" ]]; then
    log_error "Missing required argument: --node-ids"
    usage 1
fi

if [[ -z "$TFVARS_FILE" ]]; then
    log_error "Missing required argument: --tfvars-file"
    usage 1
fi

# Parse comma-separated node IDs into an array
IFS=',' read -ra NODE_IDS <<< "$NODE_IDS_RAW"

# ─── Validate Prerequisites ──────────────────────────────────────────────────

if ! command -v aws &>/dev/null; then
    log_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

if ! command -v jq &>/dev/null; then
    log_error "jq is not installed. Please install it first (brew install jq / apt install jq)."
    exit 1
fi

# ─── Build AWS CLI Region Flag ───────────────────────────────────────────────

AWS_REGION_FLAG=()
if [[ -n "$REGION" ]]; then
    AWS_REGION_FLAG=(--region "$REGION")
fi

# ─── Validate Node ID Format ─────────────────────────────────────────────────

for node_id in "${NODE_IDS[@]}"; do
    if [[ ! "$node_id" =~ ^i-[a-f0-9]{8,17}$ ]]; then
        log_error "Invalid node ID format: '$node_id'. Expected format: i-[a-f0-9]{8,17} (e.g., i-0abc1234def56789a)"
        exit 1
    fi
done

log_info "Validated ${#NODE_IDS[@]} node ID(s): ${NODE_IDS[*]}"

# ─── Validate tfvars File ────────────────────────────────────────────────────

if [[ ! -f "$TFVARS_FILE" ]]; then
    log_error "terraform.tfvars not found at: $TFVARS_FILE"
    exit 1
fi

TFVARS_FILE="$(cd "$(dirname "$TFVARS_FILE")" && pwd)/$(basename "$TFVARS_FILE")"
log_info "Using tfvars file: $TFVARS_FILE"

# ─── Describe Cluster ────────────────────────────────────────────────────────

log_info "Describing cluster '$CLUSTER_NAME'..."

CLUSTER_JSON=$(aws sagemaker describe-cluster \
    "${AWS_REGION_FLAG[@]}" \
    --cluster-name "$CLUSTER_NAME" 2>&1) || {
    log_error "Failed to describe cluster '$CLUSTER_NAME'."
    log_error "$CLUSTER_JSON"
    exit 1
}

CLUSTER_STATUS=$(echo "$CLUSTER_JSON" | jq -r '.ClusterStatus')
log_info "Cluster status: $CLUSTER_STATUS"

if [[ "$CLUSTER_STATUS" != "InService" && "$CLUSTER_STATUS" != "Updating" ]]; then
    log_error "Cluster is in '$CLUSTER_STATUS' state. Expected 'InService' or 'Updating'."
    log_error "Cannot delete nodes from a cluster in this state."
    exit 1
fi

# ─── Find Instance Group and Get Current Count ───────────────────────────────

CURRENT_COUNT=$(echo "$CLUSTER_JSON" | jq -r \
    --arg name "$INSTANCE_GROUP_NAME" \
    '.InstanceGroups[] | select(.InstanceGroupName == $name) | .CurrentCount // empty')

if [[ -z "$CURRENT_COUNT" ]]; then
    log_error "Instance group '$INSTANCE_GROUP_NAME' not found in cluster '$CLUSTER_NAME'."
    log_error "Available instance groups:"
    echo "$CLUSTER_JSON" | jq -r '.InstanceGroups[].InstanceGroupName' | while read -r name; do
        log_error "  - $name"
    done
    exit 1
fi

TARGET_COUNT=$(echo "$CLUSTER_JSON" | jq -r \
    --arg name "$INSTANCE_GROUP_NAME" \
    '.InstanceGroups[] | select(.InstanceGroupName == $name) | .TargetCount')

log_info "Instance group '$INSTANCE_GROUP_NAME': CurrentCount=$CURRENT_COUNT, TargetCount=$TARGET_COUNT"

# ─── Validate Node IDs Exist in the Instance Group ───────────────────────────

log_info "Listing nodes in instance group '$INSTANCE_GROUP_NAME'..."

# Paginate through all cluster nodes for this instance group
ALL_NODES_JSON="[]"
NEXT_TOKEN=""

while true; do
    LIST_ARGS=(
        --cluster-name "$CLUSTER_NAME"
        --instance-group-name-contains "$INSTANCE_GROUP_NAME"
        "${AWS_REGION_FLAG[@]}"
    )

    if [[ -n "$NEXT_TOKEN" ]]; then
        LIST_ARGS+=(--next-token "$NEXT_TOKEN")
    fi

    PAGE_JSON=$(aws sagemaker list-cluster-nodes "${LIST_ARGS[@]}" 2>&1) || {
        log_error "Failed to list cluster nodes."
        log_error "$PAGE_JSON"
        exit 1
    }

    PAGE_NODES=$(echo "$PAGE_JSON" | jq '.ClusterNodeSummaries')
    ALL_NODES_JSON=$(echo "$ALL_NODES_JSON" "$PAGE_NODES" | jq -s 'add')

    NEXT_TOKEN=$(echo "$PAGE_JSON" | jq -r '.NextToken // empty')
    if [[ -z "$NEXT_TOKEN" ]]; then
        break
    fi
done

EXISTING_NODE_IDS=$(echo "$ALL_NODES_JSON" | jq -r '.[].InstanceId')

MISSING_NODES=()
for node_id in "${NODE_IDS[@]}"; do
    if ! echo "$EXISTING_NODE_IDS" | grep -q "^${node_id}$"; then
        MISSING_NODES+=("$node_id")
    fi
done

if [[ ${#MISSING_NODES[@]} -gt 0 ]]; then
    log_error "The following node IDs were not found in instance group '$INSTANCE_GROUP_NAME':"
    for missing in "${MISSING_NODES[@]}"; do
        log_error "  - $missing"
    done
    log_error ""
    log_error "Nodes currently in this instance group:"
    echo "$ALL_NODES_JSON" | jq -r '.[] | "  \(.InstanceId)  status=\(.InstanceStatus.Status)"'
    exit 1
fi

log_ok "All ${#NODE_IDS[@]} node ID(s) found in instance group '$INSTANCE_GROUP_NAME'."

# ─── Calculate New Count ─────────────────────────────────────────────────────

NEW_COUNT=$((CURRENT_COUNT - ${#NODE_IDS[@]}))

if [[ "$NEW_COUNT" -le 0 ]]; then
    log_error "Deleting ${#NODE_IDS[@]} node(s) from a group with $CURRENT_COUNT node(s) would leave $NEW_COUNT node(s)."
    log_error "Cannot reduce instance group to 0 or fewer nodes."
    exit 1
fi

# ─── Confirmation ─────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          HyperPod Node Deletion Summary                 ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  %-20s %-35s ║\n" "Cluster:" "$CLUSTER_NAME"
printf "║  %-20s %-35s ║\n" "Instance Group:" "$INSTANCE_GROUP_NAME"
printf "║  %-20s %-35s ║\n" "Nodes to delete:" "${#NODE_IDS[@]}"
for node_id in "${NODE_IDS[@]}"; do
    printf "║    %-52s ║\n" "$node_id"
done
printf "║  %-20s %-35s ║\n" "Current count:" "$CURRENT_COUNT"
printf "║  %-20s %-35s ║\n" "New count:" "$NEW_COUNT"
printf "║  %-20s %-35s ║\n" "tfvars file:" "$(basename "$TFVARS_FILE")"
printf "║  %-20s %-35s ║\n" "Auto-apply:" "$AUTO_APPLY"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if [[ "$SKIP_CONFIRM" != true ]]; then
    read -r -p "Proceed with deletion? [y/N]: " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        log_warn "Aborted by user."
        exit 0
    fi
fi

# ─── Call BatchDeleteClusterNodes ─────────────────────────────────────────────

log_info "Calling BatchDeleteClusterNodes..."

# Build the node-ids argument as a space-separated list
DELETE_RESPONSE=$(aws sagemaker batch-delete-cluster-nodes \
    "${AWS_REGION_FLAG[@]}" \
    --cluster-name "$CLUSTER_NAME" \
    --node-ids "${NODE_IDS[@]}" 2>&1) || {
    log_error "BatchDeleteClusterNodes API call failed."
    log_error "$DELETE_RESPONSE"
    exit 1
}

# Parse successful deletions
SUCCESSFUL=$(echo "$DELETE_RESPONSE" | jq -r '.Successful // [] | .[]' 2>/dev/null)
FAILED=$(echo "$DELETE_RESPONSE" | jq -r '.Failed // []' 2>/dev/null)
FAILED_COUNT=$(echo "$FAILED" | jq 'length')

if [[ -n "$SUCCESSFUL" ]]; then
    log_ok "Successfully deleted:"
    echo "$SUCCESSFUL" | while read -r id; do
        echo "    $id"
    done
fi

if [[ "$FAILED_COUNT" -gt 0 ]]; then
    log_error "Failed to delete ${FAILED_COUNT} node(s):"
    echo "$FAILED" | jq -r '.[] | "    \(.NodeId): \(.Code) - \(.Message)"'

    # Recalculate new count based on actual successful deletions
    SUCCESSFUL_COUNT=$(echo "$DELETE_RESPONSE" | jq -r '.Successful // [] | length')
    if [[ "$SUCCESSFUL_COUNT" -eq 0 ]]; then
        log_error "No nodes were deleted. Aborting tfvars update."
        exit 1
    fi

    NEW_COUNT=$((CURRENT_COUNT - SUCCESSFUL_COUNT))
    log_warn "Adjusting new count to $NEW_COUNT based on $SUCCESSFUL_COUNT successful deletion(s)."
fi

# ─── Wait for Cluster to Stabilize ───────────────────────────────────────────

log_info "Waiting for instance group to stabilize (timeout: ${STABILIZE_TIMEOUT_SECONDS}s)..."

ELAPSED=0
while [[ "$ELAPSED" -lt "$STABILIZE_TIMEOUT_SECONDS" ]]; do
    POLL_JSON=$(aws sagemaker describe-cluster \
        "${AWS_REGION_FLAG[@]}" \
        --cluster-name "$CLUSTER_NAME" 2>/dev/null)

    POLL_STATUS=$(echo "$POLL_JSON" | jq -r '.ClusterStatus')
    POLL_COUNT=$(echo "$POLL_JSON" | jq -r \
        --arg name "$INSTANCE_GROUP_NAME" \
        '.InstanceGroups[] | select(.InstanceGroupName == $name) | .CurrentCount')

    if [[ "$POLL_STATUS" == "InService" && "$POLL_COUNT" -eq "$NEW_COUNT" ]]; then
        log_ok "Instance group '$INSTANCE_GROUP_NAME' stabilized: CurrentCount=$POLL_COUNT"
        break
    fi

    log_info "Cluster status=$POLL_STATUS, ${INSTANCE_GROUP_NAME} CurrentCount=$POLL_COUNT (target: $NEW_COUNT). Waiting..."
    sleep "$STABILIZE_POLL_INTERVAL"
    ELAPSED=$((ELAPSED + STABILIZE_POLL_INTERVAL))
done

if [[ "$ELAPSED" -ge "$STABILIZE_TIMEOUT_SECONDS" ]]; then
    log_warn "Timed out waiting for cluster to stabilize after ${STABILIZE_TIMEOUT_SECONDS}s."
    log_warn "The cluster may still be updating. Proceeding with tfvars update anyway."
    log_warn "Current state: status=$POLL_STATUS, ${INSTANCE_GROUP_NAME} CurrentCount=$POLL_COUNT"
fi

# ─── Patch terraform.tfvars ──────────────────────────────────────────────────

log_info "Patching $TFVARS_FILE..."

# Read the current instance_count from tfvars for this instance group.
# Strategy: find the block containing name = "<INSTANCE_GROUP_NAME>", then
# replace instance_count within that block only.
#
# 1. Find the line with the group name
# 2. Search backward for the opening brace of that block
# 3. Search forward for the closing brace of that block
# 4. Use sed with that line range to replace instance_count

NAME_LINE=$(grep -n "name\s*=\s*\"${INSTANCE_GROUP_NAME}\"" "$TFVARS_FILE" | head -1 | cut -d: -f1)

if [[ -z "$NAME_LINE" ]]; then
    log_error "Could not find instance group '$INSTANCE_GROUP_NAME' in $TFVARS_FILE."
    log_error "You will need to manually update instance_count to $NEW_COUNT."
    exit 1
fi

# Find the opening brace '{' before the name line
BLOCK_START=$(head -n "$NAME_LINE" "$TFVARS_FILE" | grep -n '^\s*{' | tail -1 | cut -d: -f1)

if [[ -z "$BLOCK_START" ]]; then
    log_error "Could not find opening brace for instance group block in $TFVARS_FILE."
    log_error "You will need to manually update instance_count to $NEW_COUNT."
    exit 1
fi

# Find the closing brace '}' after the name line
BLOCK_END=$(tail -n +"$NAME_LINE" "$TFVARS_FILE" | grep -n '^\s*}' | head -1 | cut -d: -f1)
BLOCK_END=$((NAME_LINE + BLOCK_END - 1))

if [[ -z "$BLOCK_END" || "$BLOCK_END" -le "$NAME_LINE" ]]; then
    log_error "Could not find closing brace for instance group block in $TFVARS_FILE."
    log_error "You will need to manually update instance_count to $NEW_COUNT."
    exit 1
fi

# Read the old instance_count value from within the block
OLD_TFVARS_COUNT=$(sed -n "${BLOCK_START},${BLOCK_END}p" "$TFVARS_FILE" \
    | grep 'instance_count' \
    | head -1 \
    | sed 's/.*instance_count\s*=\s*\([0-9]*\).*/\1/')

if [[ -z "$OLD_TFVARS_COUNT" ]]; then
    log_error "Could not parse instance_count from the instance group block in $TFVARS_FILE."
    log_error "You will need to manually update instance_count to $NEW_COUNT."
    exit 1
fi

log_info "Found instance_count = $OLD_TFVARS_COUNT in tfvars (block lines ${BLOCK_START}-${BLOCK_END})."

# Perform the replacement within the block line range only
# Preserves the original whitespace/alignment by replacing only the numeric value
sed -i'' -e "${BLOCK_START},${BLOCK_END}s/\(instance_count\s*=\s*\)[0-9]*/\1${NEW_COUNT}/" "$TFVARS_FILE"

# Verify the change took effect
VERIFY_COUNT=$(sed -n "${BLOCK_START},${BLOCK_END}p" "$TFVARS_FILE" \
    | grep 'instance_count' \
    | head -1 \
    | sed 's/.*instance_count\s*=\s*\([0-9]*\).*/\1/')

if [[ "$VERIFY_COUNT" -eq "$NEW_COUNT" ]]; then
    log_ok "Updated $TFVARS_FILE: $INSTANCE_GROUP_NAME instance_count $OLD_TFVARS_COUNT -> $NEW_COUNT"
else
    log_error "Verification failed: expected instance_count=$NEW_COUNT but found $VERIFY_COUNT."
    log_error "Please manually check $TFVARS_FILE."
    exit 1
fi

# ─── Terraform Apply ─────────────────────────────────────────────────────────

TF_DIR="$(dirname "$TFVARS_FILE")"
TFVARS_BASENAME="$(basename "$TFVARS_FILE")"

# Terraform auto-loads terraform.tfvars and *.auto.tfvars — no -var-file needed.
# For any other filename, we must pass -var-file explicitly.
VAR_FILE_FLAG=()
if [[ "$TFVARS_BASENAME" != "terraform.tfvars" && "$TFVARS_BASENAME" != *.auto.tfvars ]]; then
    VAR_FILE_FLAG=(-var-file="$TFVARS_BASENAME")
fi

if [[ "$AUTO_APPLY" == true ]]; then
    log_info "Running terraform apply -auto-approve ${VAR_FILE_FLAG[*]} in $TF_DIR..."
    (cd "$TF_DIR" && terraform apply -auto-approve "${VAR_FILE_FLAG[@]}") || {
        log_error "terraform apply failed. Check the output above."
        log_error "Your tfvars file has already been updated. You may need to run terraform apply manually."
        exit 1
    }
    log_ok "Terraform apply completed successfully."
else
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                    Next Steps                           ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  Run the following to sync Terraform state:             ║"
    echo "║                                                         ║"
    printf "║    cd %-49s ║\n" "$TF_DIR"
    if [[ ${#VAR_FILE_FLAG[@]} -gt 0 ]]; then
        printf "║    terraform plan %-37s ║\n" "${VAR_FILE_FLAG[*]}  # verify no unexpected changes"
        printf "║    terraform apply %-36s ║\n" "${VAR_FILE_FLAG[*]}"
    else
        echo "║    terraform plan     # verify no unexpected changes    ║"
        echo "║    terraform apply                                      ║"
    fi
    echo "╚══════════════════════════════════════════════════════════╝"
fi

echo ""
log_ok "Done. Deleted ${#NODE_IDS[@]} node(s) from '$INSTANCE_GROUP_NAME' in cluster '$CLUSTER_NAME'."
