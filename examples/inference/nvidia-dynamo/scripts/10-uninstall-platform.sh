#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Uninstall Dynamo platform completely
set -euo pipefail

echo "🧹 Removing Dynamo Platform..."
helm uninstall dynamo-platform -n dynamo-system 2>/dev/null || true
helm uninstall dynamo-crds -n default 2>/dev/null || true
kubectl delete namespace dynamo-system --ignore-not-found
echo "✅ Platform removed"
