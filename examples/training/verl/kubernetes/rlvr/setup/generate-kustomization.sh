#!/bin/bash
set -e

# Script to generate kustomization.yaml from env_vars file
# This replaces the envsubst workflow with kustomize-based configuration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_VARS_FILE="${SCRIPT_DIR}/env_vars"
KUSTOMIZATION_FILE="${SCRIPT_DIR}/kustomization.yaml"

# Check if env_vars file exists
if [ ! -f "${ENV_VARS_FILE}" ]; then
    echo "Error: ${ENV_VARS_FILE} not found."
    echo "Please copy setup/env_vars.example to setup/env_vars and configure it."
    exit 1
fi

# Source the env_vars file to get all variables
source "${ENV_VARS_FILE}"

# Ensure required variables are set
if [ -z "${REGISTRY}" ] || [ -z "${IMAGE}" ] || [ -z "${TAG}" ]; then
    echo "Error: REGISTRY, IMAGE, and TAG must be set in env_vars"
    exit 1
fi

# Construct the full image name
FULL_IMAGE="${REGISTRY}${IMAGE}:${TAG}"

# Generate kustomization.yaml with one inline strategic merge patch (all fields)
cat > "${KUSTOMIZATION_FILE}" <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- raycluster.yaml

# Use a single inline strategic merge patch for all replacements
patches:
- target:
    kind: RayCluster
  patch: |-
    apiVersion: ray.io/v1
    kind: RayCluster
    metadata:
      name: rayml-efa
    spec:
      headGroupSpec:
        template:
          spec:
            nodeSelector:
              node.kubernetes.io/instance-type: ${INSTANCE_TYPE}
            containers:
            - name: ray-head
              image: ${FULL_IMAGE}
              env:
              - name: HF_TOKEN
                value: ${HF_TOKEN}
              volumeMounts:
              - name: fsx-storage
                mountPath: /fsx
              - name: ray-logs
                mountPath: /tmp/ray
            volumes:
            - name: ray-logs
              emptyDir: {}
            - name: fsx-storage
              persistentVolumeClaim:
                claimName: fsx-claim
      workerGroupSpecs:
      - groupName: gpu-group
        replicas: ${NUM_NODES}
        minReplicas: 1
        maxReplicas: 10
        rayStartParams:
          num-gpus: "${NUM_GPU_PER_NODE}"
        template:
          spec:
            nodeSelector:
              node.kubernetes.io/instance-type: ${INSTANCE_TYPE}
            containers:
            - name: ray-worker
              image: ${FULL_IMAGE}
              env:
              - name: HF_TOKEN
                value: ${HF_TOKEN}
              resources:
                limits:
                  nvidia.com/gpu: ${NUM_GPU_PER_NODE}
                  vpc.amazonaws.com/efa: ${NUM_EFA_PER_NODE}
                requests:
                  nvidia.com/gpu: ${NUM_GPU_PER_NODE}
                  vpc.amazonaws.com/efa: ${NUM_EFA_PER_NODE}
              volumeMounts:
              - name: fsx-storage
                mountPath: /fsx
              - name: ray-logs
                mountPath: /tmp/ray
            volumes:
            - name: ray-logs
              emptyDir: {}
            - name: fsx-storage
              persistentVolumeClaim:
                claimName: fsx-claim
EOF

echo "Generated ${KUSTOMIZATION_FILE} successfully"
echo "You can now deploy using: kubectl apply -k ${SCRIPT_DIR}/"
