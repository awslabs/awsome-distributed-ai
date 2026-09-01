#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Deploy GPT-OSS-20B — Aggregated (1 worker = prefill+decode, 1 GPU).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "🚀 Deploying gpt-oss-agg (GPT-OSS-20B, aggregated, 1 GPU)..."
kubectl apply -f "$DIR/manifests/scenarios/gpt-oss-agg"
echo ""
echo "⏳ watch: kubectl get pods -n dynamo-system -l nvidia.com/dynamo-graph-deployment-name=gpt-oss-agg -w"
echo "   test:  ./scripts/07-test-inference.sh gpt-oss-agg"
