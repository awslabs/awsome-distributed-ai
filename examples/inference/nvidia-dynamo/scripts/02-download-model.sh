#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Download a model to FSx Lustre (one-time). Shared across all pods.
# Usage: ./scripts/02-download-model.sh <gpt-oss|qwen3.6>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS=dynamo-system
PVC=dynamo-fsx
MODEL_KEY="${1:?usage: 02-download-model.sh <gpt-oss|qwen3.6>}"

case "$MODEL_KEY" in
  gpt-oss)  MODEL_ID="openai/gpt-oss-20b";     MODEL_PATH="/fsx/models/openai-gpt-oss-20b" ;;
  qwen3.6)  MODEL_ID="Qwen/Qwen3.6-27B-FP8";   MODEL_PATH="/fsx/models/qwen3.6-27b-fp8" ;;
  *) echo "Unknown model: $MODEL_KEY (use gpt-oss | qwen3.6)"; exit 1 ;;
esac
JOB="download-$(echo $MODEL_KEY | tr -d '.')"
echo "📦 Downloading ${MODEL_ID} -> FSx ${MODEL_PATH}"

# --- Ensure FSx PVC exists in the namespace (idempotent across uninstall/reinstall) ---
if kubectl get pvc "$PVC" -n "$NS" >/dev/null 2>&1; then
  echo "  PVC ${PVC} already exists, skipping"
elif kubectl get pv dynamo-fsx-pv >/dev/null 2>&1; then
  # PV persists across a namespace delete (Retain + cluster-scoped). It ends up
  # 'Released' with a stale claimRef that blocks rebinding — clear it, then re-create the PVC.
  echo "  Reusing existing FSx PV (releasing stale claim)..."
  kubectl patch pv dynamo-fsx-pv --type merge -p '{"spec":{"claimRef":null}}' >/dev/null 2>&1 || true
  PVC="$PVC" NS="$NS" envsubst '$PVC $NS' < "$SCRIPT_DIR/../manifests/fsx/fsx-pvc.yaml" | kubectl apply -f -
  kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/"$PVC" -n "$NS" --timeout=60s
else
  # Fresh cluster: derive FSx details from any existing fsx.csi PV and create PV + PVC.
  FSX_HANDLE=$(kubectl get pv -o jsonpath='{.items[?(@.spec.csi.driver=="fsx.csi.aws.com")].spec.csi.volumeHandle}' 2>/dev/null | awk '{print $1}')
  FSX_DNS=$(kubectl get pv -o jsonpath='{.items[?(@.spec.csi.driver=="fsx.csi.aws.com")].spec.csi.volumeAttributes.dnsname}' 2>/dev/null | awk '{print $1}')
  FSX_MOUNT=$(kubectl get pv -o jsonpath='{.items[?(@.spec.csi.driver=="fsx.csi.aws.com")].spec.csi.volumeAttributes.mountname}' 2>/dev/null | awk '{print $1}')
  [ -z "$FSX_HANDLE" ] && { echo "❌ No FSx PersistentVolume found."; exit 1; }
  FSX_HANDLE="$FSX_HANDLE" FSX_DNS="$FSX_DNS" FSX_MOUNT="$FSX_MOUNT" \
    envsubst '$FSX_HANDLE $FSX_DNS $FSX_MOUNT' < "$SCRIPT_DIR/../manifests/fsx/fsx-pv.yaml" | kubectl apply -f -
  PVC="$PVC" NS="$NS" envsubst '$PVC $NS' < "$SCRIPT_DIR/../manifests/fsx/fsx-pvc.yaml" | kubectl apply -f -
  kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/"$PVC" -n "$NS" --timeout=60s
fi

kubectl delete job "$JOB" -n "$NS" --ignore-not-found >/dev/null 2>&1
JOB="$JOB" NS="$NS" MODEL_ID="$MODEL_ID" MODEL_PATH="$MODEL_PATH" PVC="$PVC" \
  envsubst '$JOB $NS $MODEL_ID $MODEL_PATH $PVC' < "$SCRIPT_DIR/../manifests/jobs/download-model.yaml" | kubectl apply -f -
echo "⏳ waiting for download (5-20 min)..."
kubectl wait --for=condition=complete job/"$JOB" -n "$NS" --timeout=2400s && echo "✅ ${MODEL_ID} on FSx" || \
  echo "  check: kubectl logs job/${JOB} -n ${NS}"
