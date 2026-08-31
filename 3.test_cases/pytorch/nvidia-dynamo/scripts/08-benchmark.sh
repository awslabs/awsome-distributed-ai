#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Benchmark a deployed scenario with AIPerf (concurrency sweep) -> bench_results/<name>/.
# Auto-detects the deployed DynamoGraphDeployment if no name is given.
# Usage: ./scripts/08-benchmark.sh [gpt-oss-agg|gpt-oss-disagg|qwen36-agg|qwen36-disagg] [isl] [osl]
set -euo pipefail
NS=dynamo-system
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load-generator harness only (AIPerf). Intentionally pinned to 0.7.0 and independent of
# the 1.2.1 serving image — this container drives traffic, it does not serve the model.
IMAGE="nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.7.0"
NAME="${1:-}"
if [ -z "$NAME" ]; then
  DET=$(kubectl get dynamographdeployment -n "$NS" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null)
  n=$(echo "$DET" | grep -c . || true)
  if   [ "$n" -eq 0 ]; then echo "❌ No DynamoGraphDeployment deployed."; exit 1
  elif [ "$n" -eq 1 ]; then NAME="$DET"; echo "🔎 auto-detected: $NAME"
  elif [ -t 0 ]; then echo "Multiple deployed — pick one:"; select s in $DET; do [ -n "$s" ] && { NAME="$s"; break; }; done
  else echo "❌ Multiple deployed ($(echo $DET)). Pass one as arg (custom ISL/OSL also need an explicit name)."; exit 1; fi
fi
ISL="${2:-200}"; OSL="${3:-256}"
CONCURRENCIES="${CONCURRENCIES:-1,2,5,10}"
case "$NAME" in
  gpt-oss-*) MODEL="openai/gpt-oss-20b" ;;
  qwen36-*)  MODEL="Qwen/Qwen3.6-27B-FP8" ;;
  *) echo "Unknown scenario: $NAME"; exit 1 ;;
esac
URL="http://${NAME}-frontend.${NS}.svc.cluster.local:8000"
JOB="bench-${NAME}"
OUT="${SCRIPT_DIR}/../bench_results/${NAME}"; mkdir -p "$OUT"

echo "📊 Benchmark ${NAME}  model=${MODEL}  ISL/OSL=${ISL}/${OSL}  conc=${CONCURRENCIES}  url=${URL}"
kubectl delete job "$JOB" -n "$NS" --ignore-not-found >/dev/null 2>&1; sleep 2
JOB="$JOB" NS="$NS" IMAGE="$IMAGE" URL="$URL" CONCURRENCIES="$CONCURRENCIES" NAME="$NAME" MODEL="$MODEL" ISL="$ISL" OSL="$OSL" \
  envsubst '$JOB $NS $IMAGE $URL $CONCURRENCIES $NAME $MODEL $ISL $OSL' < "$SCRIPT_DIR/../manifests/jobs/benchmark.yaml" | kubectl apply -f - >/dev/null
echo "⏳ benchmark running (polling; saving to ${OUT}/benchmark.log)..."
for i in $(seq 1 60); do
  kubectl logs job/$JOB -n $NS --tail=-1 > "${OUT}/benchmark.log" 2>/dev/null || true
  grep -q "BENCH_DONE" "${OUT}/benchmark.log" 2>/dev/null && { echo done; break; }
  [ "$(kubectl get pod -l job-name=$JOB -n $NS -o jsonpath='{.items[0].status.phase}' 2>/dev/null)" = "Failed" ] && { echo "pod failed"; break; }
  sleep 20
done
kubectl logs job/$JOB -n $NS --tail=-1 > "${OUT}/benchmark.log" 2>/dev/null || true
kubectl delete job "$JOB" -n "$NS" --ignore-not-found >/dev/null 2>&1
echo "✅ saved ${OUT}/benchmark.log"
