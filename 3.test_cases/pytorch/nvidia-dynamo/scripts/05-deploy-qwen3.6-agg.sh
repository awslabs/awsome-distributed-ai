#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Deploy Qwen3.6-27B-FP8 — Aggregated (1 worker = prefill+decode, 1 GPU).
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "🚀 Deploying qwen3.6-agg (Qwen3.6-27B-FP8, aggregated, 1 GPU)..."
kubectl apply -f "$DIR/scenarios/qwen3.6-agg"
echo ""
echo "⏳ watch: kubectl get pods -n dynamo-system -l nvidia.com/dynamo-graph-deployment-name=qwen36-agg -w"
echo "   test:  ./scripts/07-test-inference.sh qwen36-agg"
