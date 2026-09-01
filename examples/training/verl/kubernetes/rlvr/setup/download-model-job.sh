#!/bin/bash
set -euo pipefail

# Create a Kubernetes Job that downloads a HF model into HF_MODEL_PATH.
# Defaults:
#   HF_MODEL_REPO="Qwen/Qwen3-VL-235B-A22B-Instruct"
#   HF_MODEL_PATH="${RAY_DATA_HOME:-/fsx/verl}/models/${MODEL_LOCAL_NAME:-Qwen3-VL-235B-A22B-Instruct}"
# Requirements:
#   - HF_TOKEN set in setup/env_vars
#   - fsx-claim PVC available and mounted at /fsx
#   - Image must have python + huggingface_hub (present in the training image)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_VARS_FILE="${SCRIPT_DIR}/env_vars"

if [ ! -f "${ENV_VARS_FILE}" ]; then
  echo "Missing ${ENV_VARS_FILE}. Copy env_vars.example and set your values."
  exit 1
fi

source "${ENV_VARS_FILE}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN must be set in ${ENV_VARS_FILE}."
  exit 1
fi

IMAGE="${REGISTRY}${IMAGE}:${TAG}"
JOB_NAME="download-qwen3-vl-235b"
HF_MODEL_REPO="${HF_MODEL_REPO:-Qwen/Qwen3-VL-235B-A22B-Instruct}"
MODEL_LOCAL_NAME="${MODEL_LOCAL_NAME:-Qwen3-VL-235B-A22B-Instruct}"
HF_MODEL_PATH="${HF_MODEL_PATH:-${RAY_DATA_HOME:-/fsx/verl}/models/${MODEL_LOCAL_NAME}}"

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${RAY_NAMESPACE:-default}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      dnsPolicy: ClusterFirst
      containers:
      - name: download-model
        image: ${IMAGE}
        imagePullPolicy: IfNotPresent
        env:
        - name: HF_TOKEN
          value: "${HF_TOKEN}"
        - name: HF_MODEL_REPO
          value: "${HF_MODEL_REPO}"
        - name: HF_MODEL_PATH
          value: "${HF_MODEL_PATH}"
        command:
        - /bin/bash
        - -lc
        - |
          set -euo pipefail
          mkdir -p "${HF_MODEL_PATH}"
          python3 - <<'PY'
          import os
          from huggingface_hub import snapshot_download

          model_repo = os.environ["HF_MODEL_REPO"]
          model_path = os.environ["HF_MODEL_PATH"]
          token = os.environ.get("HF_TOKEN")

          print(f"Downloading {model_repo} to {model_path}")
          snapshot_download(
              repo_id=model_repo,
              local_dir=model_path,
              local_dir_use_symlinks=False,
              token=token,
              resume_download=True,
          )
          print("Download completed.")
          PY
        volumeMounts:
        - name: fsx-storage
          mountPath: /fsx
      volumes:
      - name: fsx-storage
        persistentVolumeClaim:
          claimName: fsx-claim
EOF

echo "Job ${JOB_NAME} applied. Check status with: kubectl get pods -n ${RAY_NAMESPACE:-default} -l job-name=${JOB_NAME}"
echo "After completion, HF_MODEL_PATH is: ${HF_MODEL_PATH}"

