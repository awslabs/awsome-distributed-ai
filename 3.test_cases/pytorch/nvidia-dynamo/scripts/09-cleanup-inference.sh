#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Remove a deployed scenario (DynamoGraphDeployment), or all. Platform is left running.
# Auto-detects deployed DGDs if no name is given.
# Usage: ./scripts/09-cleanup-inference.sh [gpt-oss-agg|gpt-oss-disagg|qwen36-agg|qwen36-disagg|all]
set -e
NS=dynamo-system
TARGET="${1:-}"
DET=$(kubectl get dynamographdeployment -n "$NS" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null)

if [ -z "$TARGET" ]; then
  n=$(echo "$DET" | grep -c .)
  if   [ "$n" -eq 0 ]; then echo "ℹ️  No DynamoGraphDeployment deployed — nothing to clean."; exit 0
  elif [ "$n" -eq 1 ]; then TARGET="$DET"; echo "🔎 auto-detected: $TARGET"
  elif [ -t 0 ]; then echo "Multiple deployed — pick what to remove:"; select s in $DET all; do [ -n "$s" ] && { TARGET="$s"; break; }; done
  else echo "❌ Multiple deployed ($(echo $DET)). Pass one, or 'all'."; exit 1; fi
fi

if [ "$TARGET" = "all" ]; then
  for s in $DET; do echo "🧹 removing $s"; kubectl delete dynamographdeployment "$s" -n "$NS" --ignore-not-found --wait=false; done
else
  echo "🧹 removing $TARGET"; kubectl delete dynamographdeployment "$TARGET" -n "$NS" --ignore-not-found --wait=false
fi
echo "✅ cleanup done (operator removes the pods)"
