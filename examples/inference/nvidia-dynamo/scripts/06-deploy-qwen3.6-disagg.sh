#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Deploy Qwen3.6-27B-FP8 — Disaggregated (prefill + decode workers, 2 GPUs, NIXL).
# Requires the platform etcd/nats (run 01-install-platform.sh first).
# Note: decode caps --max-running-requests (mamba/SSM state) to fit a 48GB L40S.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "🚀 Deploying qwen3.6-disagg (Qwen3.6-27B-FP8, prefill+decode, 2 GPUs)..."
kubectl apply -f "$DIR/manifests/scenarios/qwen3.6-disagg"
echo ""
echo "⏳ watch: kubectl get pods -n dynamo-system -l nvidia.com/dynamo-graph-deployment-name=qwen36-disagg -w"
echo "   test:  ./scripts/07-test-inference.sh qwen36-disagg"
