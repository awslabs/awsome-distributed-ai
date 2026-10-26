#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Create vllm-prefill and vllm-decode from manifests/serving-pod.yaml on PREFILL_NODE and DECODE_NODE, and wait
# until they run. Source env_vars first.
#
# Usage: source env_vars && ./0.create-pods.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${NAMESPACE:?source env_vars}" "${IMAGE:?}" "${PREFILL_NODE:?}" "${DECODE_NODE:?}"
# shellcheck disable=SC2016  # the list names the variables envsubst may substitute
VARS='$NAMESPACE $IMAGE $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $MODEL_VOLUME_SIZE $POD_MEMORY $ROLE $NODE'
for pair in "prefill:$PREFILL_NODE" "decode:$DECODE_NODE"; do
  ROLE=${pair%%:*} NODE=${pair#*:} envsubst "$VARS" < "$HERE/manifests/serving-pod.yaml" | kubectl apply -f -
done
kubectl -n "$NAMESPACE" wait --for=condition=Ready pod -l app=vllm-pd --timeout=30m
kubectl -n "$NAMESPACE" get pods -l app=vllm-pd -o wide
