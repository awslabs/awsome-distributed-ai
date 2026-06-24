#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Deploy GPT-OSS-20B — Disaggregated (prefill + decode workers, 2 GPUs, NIXL).
# Requires the platform etcd/nats (run 01-install-platform.sh first).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "🚀 Deploying gpt-oss-disagg (GPT-OSS-20B, prefill+decode, 2 GPUs)..."
kubectl apply -f "$DIR/manifests/scenarios/gpt-oss-disagg"
echo ""
echo "⏳ watch: kubectl get pods -n dynamo-system -l nvidia.com/dynamo-graph-deployment-name=gpt-oss-disagg -w"
echo "   test:  ./scripts/07-test-inference.sh gpt-oss-disagg"
