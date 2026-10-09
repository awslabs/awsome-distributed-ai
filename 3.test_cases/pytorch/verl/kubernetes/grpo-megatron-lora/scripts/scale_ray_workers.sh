#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Scale the Ray p6-b200 worker group up or down.
#
# Stopping a Ray *job* returns logical resources to the Ray scheduler, but the
# Ray *worker pods* keep holding the `nvidia.com/gpu` reservations on their
# HyperPod nodes. To actually free a node (e.g. to land the vllm-eval
# Deployment), the worker group must be scaled down.
#
# Usage:
#   ./scripts/scale_ray_workers.sh <replicas>
#
#   # Typical eval window:
#   ./scripts/scale_ray_workers.sh 5      # free one p6-b200 for vLLM
#   # ... run eval ...
#   ./scripts/scale_ray_workers.sh 6      # restore full training capacity
#
# Env overrides:
#   RAY_CLUSTER   — RayCluster name (default: ray-cluster)
#   WORKER_GROUP  — workerGroupSpecs entry to scale (default: p6-b200-workers)
#   KUBE_NAMESPACE — K8s namespace (default: default)
# =============================================================================
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <replicas>"
    echo ""
    echo "Example:"
    echo "  $0 5     # scale down to 5 workers"
    echo "  $0 6     # restore to 6 workers"
    exit 1
fi

REPLICAS="$1"
# Discover both for your cluster with:
#   kubectl get rayclusters -n "$KUBE_NAMESPACE"
#   kubectl get raycluster <name> -o jsonpath='{.spec.workerGroupSpecs[*].groupName}'
RAY_CLUSTER="${RAY_CLUSTER:-ray-cluster}"
WORKER_GROUP="${WORKER_GROUP:-p6-b200-workers}"
KUBE_NAMESPACE="${KUBE_NAMESPACE:-default}"

case "${REPLICAS}" in
    ''|*[!0-9]*) echo "error: replicas must be a non-negative integer, got '${REPLICAS}'" >&2; exit 1 ;;
esac

echo "=============================================="
echo "Scale Ray worker group"
echo "=============================================="
echo "RayCluster:    ${RAY_CLUSTER}"
echo "Worker group:  ${WORKER_GROUP}"
echo "Namespace:     ${KUBE_NAMESPACE}"
echo "Target:        ${REPLICAS} replicas"
echo "=============================================="

# A JSON patch targets the worker group's replicas/minReplicas by array index
# WITHOUT touching the rest of the spec. A strategic --type=merge on the
# workerGroupSpecs LIST would replace the whole element and drop the required
# `template`, which KubeRay rejects ("spec.workerGroupSpecs[0].template: Required value").
# Resolve the group's index by name so we patch the correct entry.
GROUP_INDEX=$(kubectl get raycluster "${RAY_CLUSTER}" -n "${KUBE_NAMESPACE}" \
    -o jsonpath="{range .spec.workerGroupSpecs[*]}{.groupName}{'\n'}{end}" \
    | grep -nxF "${WORKER_GROUP}" | head -1 | cut -d: -f1)
if [ -z "${GROUP_INDEX}" ]; then
    echo "error: worker group '${WORKER_GROUP}' not found in RayCluster '${RAY_CLUSTER}'" >&2
    exit 1
fi
GROUP_INDEX=$((GROUP_INDEX - 1))  # jsonpath/array is 0-based

# maxReplicas MUST be patched too. KubeRay/the autoscaler clamps `replicas` to
# maxReplicas, so a group left at maxReplicas=0 (e.g. after scaling to 0 to free
# nodes for eval) silently refuses to scale back up: `replicas` and `minReplicas`
# read 6 while AVAILABLE WORKERS stays empty and no pods are ever created.
# Pinning all three to REPLICAS keeps this self-healing — any later invocation
# with a higher count raises the ceiling with it.
PATCH=$(cat <<JSON
[
  {"op": "replace", "path": "/spec/workerGroupSpecs/${GROUP_INDEX}/replicas", "value": ${REPLICAS}},
  {"op": "replace", "path": "/spec/workerGroupSpecs/${GROUP_INDEX}/minReplicas", "value": ${REPLICAS}},
  {"op": "replace", "path": "/spec/workerGroupSpecs/${GROUP_INDEX}/maxReplicas", "value": ${REPLICAS}}
]
JSON
)

kubectl patch raycluster "${RAY_CLUSTER}" \
    -n "${KUBE_NAMESPACE}" \
    --type=json \
    -p "${PATCH}"

echo ""
echo "Patch applied. Watch workers converge:"
echo "  kubectl get pods -n ${KUBE_NAMESPACE} -l ray.io/cluster=${RAY_CLUSTER} -w"
echo ""
echo "Confirm freed capacity on a HyperPod node:"
echo "  kubectl describe node <hyperpod-i-...> | grep -A1 'Allocated resources' | grep nvidia.com/gpu"
