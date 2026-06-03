#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Cleanup training resources from SageMaker HyperPod (EKS).
#
# Deletes PyTorchJobs but leaves the FSx PVC and namespace intact
# (shared storage may contain training data; namespace may be admin-managed).
#
# Usage:
#   ./2.cleanup.sh

set -euo pipefail

###########################
###### User Variables #####
###########################

NAMESPACE=${NAMESPACE:-ml-training}

###########################
###### Delete Jobs ########
###########################

echo "Deleting PyTorchJobs..."
kubectl delete pytorchjob qwen3-qlora-training-zero2 -n "${NAMESPACE}" 2>/dev/null || true
kubectl delete pytorchjob qwen3-qlora-training-zero3 -n "${NAMESPACE}" 2>/dev/null || true

echo ""
echo "Training jobs deleted."
echo ""
echo "Optional — delete the ECR repository:"
echo "  aws ecr delete-repository --repository-name qwen3-qlora-training --region \$AWS_REGION --force"
echo ""
echo "Optional — delete FSx PVC and PV (WARNING: training data on FSx is preserved either way):"
echo "  kubectl delete pvc fsx-claim-qlora -n ${NAMESPACE}"
echo "  kubectl delete pv fsx-pv-qlora"
echo ""
echo "Optional — delete the namespace (may be admin-managed on HyperPod):"
echo "  kubectl delete namespace ${NAMESPACE}"
