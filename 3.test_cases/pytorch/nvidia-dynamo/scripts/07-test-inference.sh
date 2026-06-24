#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Smoke test a deployed scenario (models list + chat) via kubectl exec.
# Auto-detects the deployed DynamoGraphDeployment if no name is given.
# Usage: ./scripts/07-test-inference.sh [gpt-oss-agg|gpt-oss-disagg|qwen36-agg|qwen36-disagg]
set -e
NS=dynamo-system
NAME="${1:-}"
if [ -z "$NAME" ]; then
  DET=$(kubectl get dynamographdeployment -n "$NS" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null)
  n=$(echo "$DET" | grep -c .)
  if   [ "$n" -eq 0 ]; then echo "❌ No DynamoGraphDeployment deployed. Deploy one with scripts/03..06."; exit 1
  elif [ "$n" -eq 1 ]; then NAME="$DET"; echo "🔎 auto-detected: $NAME"
  elif [ -t 0 ]; then echo "Multiple deployed — pick one:"; select s in $DET; do [ -n "$s" ] && { NAME="$s"; break; }; done
  else echo "❌ Multiple deployed ($(echo $DET)). Pass one as arg."; exit 1; fi
fi
case "$NAME" in
  gpt-oss-*) MODEL="openai/gpt-oss-20b" ;;
  qwen36-*)  MODEL="Qwen/Qwen3.6-27B-FP8" ;;
  *) echo "Unknown scenario: $NAME"; exit 1 ;;
esac

POD=$(kubectl get pod -l "nvidia.com/dynamo-graph-deployment-name=${NAME},nvidia.com/dynamo-component=Frontend" -n "$NS" -o jsonpath='{.items[0].metadata.name}')
echo "🔌 ${NAME} via pod/${POD} (model ${MODEL})"
echo ""; echo "🧪 Models:"
kubectl exec "$POD" -n "$NS" -- curl -s http://localhost:8000/v1/models | python3 -m json.tool
echo ""; echo "🧪 Chat:"
kubectl exec "$POD" -n "$NS" -- curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what is Amazon S3?\"}],\"max_tokens\":2000}" \
  | python3 -m json.tool
echo ""; echo "✅ Done"
