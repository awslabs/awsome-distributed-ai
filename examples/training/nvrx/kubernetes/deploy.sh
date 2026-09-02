#!/bin/bash
# deploy.sh -- Deploy K8s manifests with envsubst, safely.
#
# Uses an explicit variable list so envsubst does NOT clobber runtime
# shell variables like $HOSTNAME, $RANK, $POD_IP, $MASTER_ADDR, etc.
# that appear in the bash args blocks of our manifests.
#
# Usage:
#   source env_vars
#   ./deploy.sh training-job-local-ckpt.yaml
#   ./deploy.sh training-job-local-ckpt.yaml --dry-run
#   ./deploy.sh --delete training-job-local-ckpt.yaml
#
# Supports:
#   --dry-run    Show rendered YAML without applying
#   --delete     Delete resources instead of applying

set -euo pipefail

# ---------------------------------------------------------------------------
# Variables to substitute (must match env_vars.template exports)
# ONLY these variables are replaced; all others ($HOSTNAME, $RANK, etc.) are
# left untouched for runtime evaluation inside the container.
# ---------------------------------------------------------------------------
ENVSUBST_VARS='$IMAGE_URI $NUM_NODES $GPU_PER_NODE $EFA_PER_NODE $NAMESPACE $NODE_TYPE $FSX_PVC_NAME $DEDICATED_TAINT_VALUE $NCCL_DEBUG $CPU_LIMIT $CPU_REQUEST $MEMORY_LIMIT $MEMORY_REQUEST $MODEL_NAME $MAX_SEQ_LENGTH $BATCH_SIZE $LEARNING_RATE'

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=false
DELETE=false
MANIFEST=""

for arg in "$@"; do
  case "$arg" in
    --dry-run)  DRY_RUN=true ;;
    --delete)   DELETE=true ;;
    *)          MANIFEST="$arg" ;;
  esac
done

if [ -z "$MANIFEST" ]; then
  echo "Usage: $0 [--dry-run|--delete] <manifest.yaml>"
  echo ""
  echo "Examples:"
  echo "  $0 training-job-local-ckpt.yaml"
  echo "  $0 --dry-run training-job-inprocess.yaml"
  echo "  $0 --delete training-job-ft-launcher.yaml"
  exit 1
fi

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: Manifest not found: $MANIFEST"
  exit 1
fi

# ---------------------------------------------------------------------------
# Validate that required env vars are set
# ---------------------------------------------------------------------------
MISSING=()
for var in IMAGE_URI NUM_NODES GPU_PER_NODE EFA_PER_NODE NAMESPACE NODE_TYPE FSX_PVC_NAME NCCL_DEBUG CPU_LIMIT CPU_REQUEST MEMORY_LIMIT MEMORY_REQUEST MODEL_NAME MAX_SEQ_LENGTH BATCH_SIZE LEARNING_RATE; do
  if [ -z "${!var:-}" ]; then
    MISSING+=("$var")
  fi
done
# DEDICATED_TAINT_VALUE is optional (empty = no taint toleration)

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "ERROR: Missing environment variables. Run 'source env_vars' first."
  echo "  Missing: ${MISSING[*]}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Render and apply/delete
# ---------------------------------------------------------------------------
RENDERED=$(envsubst "${ENVSUBST_VARS}" < "$MANIFEST")

if [ "$DRY_RUN" = true ]; then
  echo "# --- Rendered: $MANIFEST ---"
  echo "$RENDERED"
  echo "# --- End ---"
elif [ "$DELETE" = true ]; then
  echo "Deleting resources from: $MANIFEST"
  echo "$RENDERED" | kubectl delete -f - --ignore-not-found
  echo "Done."
else
  echo "Deploying: $MANIFEST"
  echo "  Namespace: ${NAMESPACE}"
  echo "  Image:     ${IMAGE_URI}"
  echo "  Nodes:     ${NUM_NODES} x ${GPU_PER_NODE} GPUs (${EFA_PER_NODE} EFA)"
  echo "  Model:     ${MODEL_NAME}"
  echo "$RENDERED" | kubectl apply -f -
  echo "Done."
fi
