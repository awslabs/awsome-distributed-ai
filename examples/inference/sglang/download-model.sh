#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Render download-model-daemonset.yaml for a given model + node type and apply
# it, pre-staging the weights to every matching node's NVMe (/opt/dlami/nvme).
#
# Usage:
#   ./download-model.sh <HF_REPO_ID> <INSTANCE_TYPE>
#
# Examples:
#   ./download-model.sh deepseek-ai/DeepSeek-V4-Pro ml.p6-b300.48xlarge
#   ./download-model.sh moonshotai/Kimi-K2.6 ml.p5en.48xlarge
#
# The weights are staged in HF cache layout under
# /opt/dlami/nvme/huggingface — the dir every serving manifest mounts at
# /root/.cache/huggingface, with engines loading by repo id, so the staged
# snapshot is found as a cache hit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <HF_REPO_ID> <INSTANCE_TYPE>" >&2
    exit 1
fi

export HF_REPO_ID="$1"
export INSTANCE_TYPE="$2"

# The two cluster types label instance-type differently: a plain EKS managed
# nodegroup uses the bare EC2 type (p5en.48xlarge), a SageMaker HyperPod instance
# group prefixes it with `ml.` (ml.p5en.48xlarge). Derive the other spelling so
# the DaemonSet's nodeAffinity matches GPU nodes on either cluster regardless of
# which form was passed in.
if [[ "${INSTANCE_TYPE}" == ml.* ]]; then
    export INSTANCE_TYPE_ALT="${INSTANCE_TYPE#ml.}"
else
    export INSTANCE_TYPE_ALT="ml.${INSTANCE_TYPE}"
fi

echo "==> Pre-staging ${HF_REPO_ID}"
echo "    nodes:  ${INSTANCE_TYPE} / ${INSTANCE_TYPE_ALT}"
echo "    target: <nvme>/huggingface (HF cache layout)"

envsubst '${INSTANCE_TYPE} ${INSTANCE_TYPE_ALT} ${HF_REPO_ID}' \
    < "${SCRIPT_DIR}/download-model-daemonset.yaml" \
    | kubectl apply -f -

echo
echo "==> Applied. Watch progress with:"
echo "    kubectl logs -f -l app=model-downloader"
echo "    Each node prints 'Download complete!' when its copy is staged."
echo "    Remove the downloader once done: kubectl delete daemonset model-downloader"
