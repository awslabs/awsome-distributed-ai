#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Render download-model-daemonset.yaml for a given model + node type and apply
# it, pre-staging the weights to every matching node's NVMe (/opt/dlami/nvme).
#
# Usage:
#   ./download-model.sh <HF_REPO_ID> <INSTANCE_TYPE> [LOCAL_DIR_NAME]
#
# Examples:
#   ./download-model.sh moonshotai/Kimi-K2.5  ml.p5en.48xlarge
#   ./download-model.sh deepseek-ai/DeepSeek-V4-Pro ml.p6-b300.48xlarge
#
# LOCAL_DIR_NAME defaults to the repo id with '/' replaced by '-'
# (e.g. moonshotai/Kimi-K2.5 -> moonshotai-Kimi-K2.5). The weights land at
# /opt/dlami/nvme/<LOCAL_DIR_NAME> on each node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <HF_REPO_ID> <INSTANCE_TYPE> [LOCAL_DIR_NAME]" >&2
    exit 1
fi

export HF_REPO_ID="$1"
export INSTANCE_TYPE="$2"
export LOCAL_DIR_NAME="${3:-${HF_REPO_ID//\//-}}"

echo "==> Pre-staging ${HF_REPO_ID}"
echo "    nodes:  ${INSTANCE_TYPE}"
echo "    target: /opt/dlami/nvme/${LOCAL_DIR_NAME}"

envsubst '${INSTANCE_TYPE} ${HF_REPO_ID} ${LOCAL_DIR_NAME}' \
    < "${SCRIPT_DIR}/download-model-daemonset.yaml" \
    | kubectl apply -f -

echo
echo "==> Applied. Watch progress with:"
echo "    kubectl logs -f -l app=model-downloader"
echo "    Each node prints 'Download complete!' when its copy is staged."
echo "    Remove the downloader once done: kubectl delete daemonset model-downloader"
