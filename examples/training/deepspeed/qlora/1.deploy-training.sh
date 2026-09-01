#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Deploy the QLoRA training PyTorchJob to SageMaker HyperPod (EKS).
#
# Usage:
#   export IMAGE=<account>.dkr.ecr.<region>.amazonaws.com/qwen3-qlora-training:latest
#   export FSX_FILESYSTEM_ID=fs-0123456789abcdef0
#   export FSX_DNS_NAME=fs-0123456789abcdef0.fsx.us-east-1.amazonaws.com
#   export FSX_MOUNT_NAME=abcdef01
#   ./1.deploy-training.sh              # Default: single-node, DeepSpeed ZeRO-2
#   NUM_NODES=2 ./1.deploy-training.sh  # Multi-node (2 nodes, 8 GPUs total)
#   DEEPSPEED_STRATEGY=zero3 ./1.deploy-training.sh   # DeepSpeed ZeRO-3
#   MIG_PROFILE=3g.20gb NUM_GPUS=2 ./1.deploy-training.sh  # MIG mode
#
# Prerequisites:
#   - SageMaker HyperPod EKS cluster with the HyperPod Helm chart installed
#     (bundles Training Operator, NVIDIA device plugin, health monitoring agents)
#   - FSx for Lustre filesystem provisioned in the same VPC
#   - FSx CSI driver installed (included in the HyperPod Helm chart)
#   - Docker image built and pushed (see 0.build-image.sh)

set -euo pipefail

###########################
###### User Variables #####
###########################

export NAMESPACE=${NAMESPACE:-ml-training}
export NUM_GPUS=${NUM_GPUS:-4}
export INSTANCE_TYPE=${INSTANCE_TYPE:-ml.g5.12xlarge}
NUM_NODES=${NUM_NODES:-1}
export NUM_WORKERS=$(( NUM_NODES - 1 ))
DEEPSPEED_STRATEGY=${DEEPSPEED_STRATEGY:-zero2}

# MIG support — set MIG_PROFILE to use NVIDIA MIG partitions instead of full GPUs.
# Examples: "3g.20gb" (A100 40GB), "4g.40gb" (A100 80GB / H100 80GB)
# When set, GPU_RESOURCE becomes nvidia.com/mig-<profile> and pod resources are
# scaled down to match the partition size.
MIG_PROFILE=${MIG_PROFILE:-}

if [ -n "${MIG_PROFILE}" ]; then
    export GPU_RESOURCE="nvidia.com/mig-${MIG_PROFILE}"
    # MIG pods use a fraction of the node — reduce CPU/memory accordingly
    export POD_MEMORY_REQUEST=${POD_MEMORY_REQUEST:-"40Gi"}
    export POD_MEMORY_LIMIT=${POD_MEMORY_LIMIT:-"48Gi"}
    export POD_CPU_REQUEST=${POD_CPU_REQUEST:-"8"}
    export POD_CPU_LIMIT=${POD_CPU_LIMIT:-"12"}
    export SHM_SIZE=${SHM_SIZE:-"16Gi"}
else
    export GPU_RESOURCE=${GPU_RESOURCE:-"nvidia.com/gpu"}
    # Full-node defaults (g5.12xlarge: 48 vCPU, 192 GiB)
    export POD_MEMORY_REQUEST=${POD_MEMORY_REQUEST:-"160Gi"}
    export POD_MEMORY_LIMIT=${POD_MEMORY_LIMIT:-"180Gi"}
    export POD_CPU_REQUEST=${POD_CPU_REQUEST:-"40"}
    export POD_CPU_LIMIT=${POD_CPU_LIMIT:-"44"}
    export SHM_SIZE=${SHM_SIZE:-"64Gi"}
fi

###########################
###### Validation #########
###########################

# IMAGE must be set
if [ -z "${IMAGE:-}" ]; then
    echo "ERROR: IMAGE environment variable not set."
    echo "Run: export IMAGE=<account>.dkr.ecr.<region>.amazonaws.com/qwen3-qlora-training:latest"
    exit 1
fi

# FSx variables must be set
for var in FSX_FILESYSTEM_ID FSX_DNS_NAME FSX_MOUNT_NAME; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: ${var} environment variable not set."
        echo "Set it from the FSx for Lustre console (File system details)."
        exit 1
    fi
    export "${var}"
done

# Select manifest based on DeepSpeed strategy
case "${DEEPSPEED_STRATEGY}" in
    zero2)
        MANIFEST="qwen3_8b-qlora-zero2.yaml"
        JOB_NAME="qwen3-qlora-training-zero2"
        ;;
    zero3)
        MANIFEST="qwen3_8b-qlora-zero3.yaml"
        JOB_NAME="qwen3-qlora-training-zero3"
        ;;
    *)
        echo "ERROR: Invalid DEEPSPEED_STRATEGY '${DEEPSPEED_STRATEGY}'. Must be 'zero2' or 'zero3'."
        exit 1
        ;;
esac

###########################
###### Create Namespace ###
###########################

echo "Ensuring namespace ${NAMESPACE} exists..."
kubectl create namespace ${NAMESPACE} 2>/dev/null || true

###########################
###### Apply Storage ######
###########################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Applying FSx storage resources..."
envsubst '$FSX_FILESYSTEM_ID $FSX_DNS_NAME $FSX_MOUNT_NAME $NAMESPACE' \
    < "${SCRIPT_DIR}/kubernetes/storage.yaml" | kubectl apply -f -

###########################
###### Deploy Job #########
###########################

echo "Deploying training job..."
echo "  Image:         ${IMAGE}"
echo "  GPU resource:  ${GPU_RESOURCE}"
echo "  GPUs per node: ${NUM_GPUS}"
echo "  Nodes:         ${NUM_NODES} (1 master + ${NUM_WORKERS} workers)"
echo "  Total GPUs:    $(( NUM_NODES * NUM_GPUS ))"
echo "  Instance Type: ${INSTANCE_TYPE}"
echo "  Namespace:     ${NAMESPACE}"
echo "  Strategy:      DeepSpeed ${DEEPSPEED_STRATEGY}"
echo "  Manifest:      ${MANIFEST}"
if [ -n "${MIG_PROFILE}" ]; then
    echo "  MIG profile:   ${MIG_PROFILE}"
fi
echo "  Pod memory:    ${POD_MEMORY_REQUEST} (request) / ${POD_MEMORY_LIMIT} (limit)"
echo "  Pod CPU:       ${POD_CPU_REQUEST} (request) / ${POD_CPU_LIMIT} (limit)"
echo "  SHM size:      ${SHM_SIZE}"
echo ""

envsubst '$IMAGE $NAMESPACE $NUM_GPUS $NUM_WORKERS $INSTANCE_TYPE $GPU_RESOURCE $POD_MEMORY_REQUEST $POD_MEMORY_LIMIT $POD_CPU_REQUEST $POD_CPU_LIMIT $SHM_SIZE' \
    < "${SCRIPT_DIR}/kubernetes/${MANIFEST}" | kubectl apply -f -

echo ""
echo "Training job deployed. Monitor with:"
echo "  kubectl get pods -n ${NAMESPACE} -w"
echo "  kubectl logs -f ${JOB_NAME}-master-0 -n ${NAMESPACE}"
