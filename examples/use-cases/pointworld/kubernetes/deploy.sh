#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# deploy.sh -- render a PointWorld manifest with envsubst and apply it.
#
# Substitutes ONLY the variables in the allowlists below, so in-container shell
# variables in the manifests ($BASE, $WDS, $RANK, $LOCAL_DATASET_DIR, the
# bash/python runtime reads in the data-prep Job, etc.) are left intact.
#
# Usage:
#   source env_vars
#   ./deploy.sh pointworld-pretrain.yaml
#   ./deploy.sh trainer-v2/pointworld-runtime.yaml
#   ./deploy.sh --dry-run pointworld-eval.yaml      # print rendered YAML only
#   ./deploy.sh --delete  pointworld-pretrain.yaml  # delete instead of apply
#
# DINOv3 secret note: the data-prep Job's DINOV3_URL is intentionally NOT
# rendered (it reads the value from the container env at runtime). To inject your
# gated URL without writing it to a file, sed-replace ONLY the env value line at
# apply time (matching the whole `value: "..."` so the in-script guard comparison
# against the literal placeholder is left intact):
#   ./deploy.sh --dry-run pointworld-data-prep.yaml \
#     | sed "s|value: \"<DINOV3_DOWNLOAD_URL>\"|value: \"$DINOV3_URL\"|" | kubectl apply -f -

set -euo pipefail

# Full allowlist: structural fields + tokenized training flags.
ENVSUBST_VARS_FULL='$IMAGE_URI $NAMESPACE $FSX_PVC_NAME $NUM_NODES $GPU_PER_NODE $EFA_PER_NODE $DOMAINS $DATA_DIRS $NORM_STATS_PATH $PTV3_SIZE $BATCH_SIZE $NUM_WORKERS $NUM_EPOCHS $MAX_TRAIN_STEPS $EVAL_FREQ $SAVE_FREQ $LOG_DIR $EXP_NAME $WANDB_MODE $MODEL_PATH $EVAL_NUM_BATCHES $EVAL_VIZ_NUM $EVAL_DOMAIN $EVAL_DATA_DIR $EVAL_CONFIDENCE_THRES'

# For the data-prep Job only: render structural fields, but leave the data knobs
# (BEHAVIOR_TASKS / MAX_CLIPS / DINOV3_URL) as-is so the container reads them at
# runtime via bash/python.
ENVSUBST_VARS_DATAPREP='$IMAGE_URI $NAMESPACE $FSX_PVC_NAME'

DRY_RUN=false
DELETE=false
MANIFEST=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --delete)  DELETE=true ;;
    -*) echo "Unknown flag: $arg" >&2; exit 1 ;;
    *)  MANIFEST="$arg" ;;
  esac
done

if [ -z "$MANIFEST" ]; then
  echo "Usage: $0 [--dry-run|--delete] <manifest.yaml>" >&2
  exit 1
fi
if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest not found: $MANIFEST" >&2
  exit 1
fi

# Pick the allowlist + required vars for this manifest.
case "$(basename "$MANIFEST")" in
  pointworld-data-prep.yaml)
    VARS="$ENVSUBST_VARS_DATAPREP"
    REQUIRED="IMAGE_URI NAMESPACE FSX_PVC_NAME"
    ;;
  *)
    VARS="$ENVSUBST_VARS_FULL"
    REQUIRED="IMAGE_URI NAMESPACE FSX_PVC_NAME NUM_NODES GPU_PER_NODE EFA_PER_NODE"
    ;;
esac

MISSING=()
for v in $REQUIRED; do
  [ -z "${!v:-}" ] && MISSING+=("$v")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "ERROR: missing env vars (run 'source env_vars' first): ${MISSING[*]}" >&2
  exit 1
fi

RENDERED=$(envsubst "$VARS" < "$MANIFEST")

if [ "$DRY_RUN" = true ]; then
  echo "$RENDERED"
elif [ "$DELETE" = true ]; then
  echo "Deleting: $MANIFEST (namespace ${NAMESPACE})"
  echo "$RENDERED" | kubectl delete -f - --ignore-not-found
else
  echo "Applying: $MANIFEST"
  echo "  namespace=${NAMESPACE} image=${IMAGE_URI}"
  echo "$RENDERED" | kubectl apply -f -
fi
