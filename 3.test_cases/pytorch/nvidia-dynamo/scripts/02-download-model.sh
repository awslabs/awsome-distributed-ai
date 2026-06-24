#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Download a model to FSx Lustre (one-time). Shared across all pods.
# Usage: ./scripts/02-download-model.sh <gpt-oss|qwen3.6>
set -e
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
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: ${PVC}, namespace: ${NS} }
spec:
  accessModes: ["ReadWriteMany"]
  resources: { requests: { storage: 1200Gi } }
  volumeName: dynamo-fsx-pv
EOF
  kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/"$PVC" -n "$NS" --timeout=60s
else
  # Fresh cluster: derive FSx details from any existing fsx.csi PV and create PV + PVC.
  FSX_HANDLE=$(kubectl get pv -o jsonpath='{.items[?(@.spec.csi.driver=="fsx.csi.aws.com")].spec.csi.volumeHandle}' 2>/dev/null | awk '{print $1}')
  FSX_DNS=$(kubectl get pv -o jsonpath='{.items[?(@.spec.csi.driver=="fsx.csi.aws.com")].spec.csi.volumeAttributes.dnsname}' 2>/dev/null | awk '{print $1}')
  FSX_MOUNT=$(kubectl get pv -o jsonpath='{.items[?(@.spec.csi.driver=="fsx.csi.aws.com")].spec.csi.volumeAttributes.mountname}' 2>/dev/null | awk '{print $1}')
  [ -z "$FSX_HANDLE" ] && { echo "❌ No FSx PersistentVolume found."; exit 1; }
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata: { name: dynamo-fsx-pv }
spec:
  capacity: { storage: 1200Gi }
  accessModes: ["ReadWriteMany"]
  persistentVolumeReclaimPolicy: Retain
  csi:
    driver: fsx.csi.aws.com
    volumeHandle: "${FSX_HANDLE}"
    volumeAttributes: { dnsname: "${FSX_DNS}", mountname: "${FSX_MOUNT}" }
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: ${PVC}, namespace: ${NS} }
spec:
  accessModes: ["ReadWriteMany"]
  resources: { requests: { storage: 1200Gi } }
  volumeName: dynamo-fsx-pv
EOF
  kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/"$PVC" -n "$NS" --timeout=60s
fi

kubectl delete job "$JOB" -n "$NS" --ignore-not-found >/dev/null 2>&1
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata: { name: ${JOB}, namespace: ${NS} }
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      nodeSelector: { sagemaker.amazonaws.com/instance-group-name: dynamo-workers }
      tolerations: [{ effect: NoSchedule, key: nvidia.com/gpu, operator: Exists }]
      containers:
        - name: download
          image: python:3.11-slim
          command: ["/bin/bash","-c"]
          args:
            - |
              set -e
              pip install -q "huggingface_hub[hf_transfer]"; export HF_HUB_ENABLE_HF_TRANSFER=1
              if [ -f "${MODEL_PATH}/config.json" ]; then echo "already present, skipping"; exit 0; fi
              python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}', local_dir='${MODEL_PATH}')"
              echo done; du -sh ${MODEL_PATH}
          volumeMounts: [{ name: fsx, mountPath: /fsx }]
      volumes: [{ name: fsx, persistentVolumeClaim: { claimName: ${PVC} } }]
EOF
echo "⏳ waiting for download (5-20 min)..."
kubectl wait --for=condition=complete job/"$JOB" -n "$NS" --timeout=2400s && echo "✅ ${MODEL_ID} on FSx" || \
  echo "  check: kubectl logs job/${JOB} -n ${NS}"
