#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Download MODEL_REPO at MODEL_REVISION into /model on every pod, in parallel and detached inside the pods (so a
# dropped kubectl exec can't interrupt it), then check that every pod holds the same file listing and sizes.
# Kimi K2 Instruct-0905 (1,029 GB) took about 7 minutes per pod from Hugging Face in the tested Region.
#
# Usage: source env_vars && ./2.stage-model.sh
set -uo pipefail
: "${NAMESPACE:?source env_vars}" "${MODEL_REPO:?}" "${MODEL_REVISION:?}"
k() { kubectl -n "$NAMESPACE" "$@"; }
DIR=/model/$(basename "$MODEL_REPO")
PODS="vllm-head vllm-near vllm-far"
for p in $PODS; do
  k exec "$p" -- bash -c "rm -f /tmp/dl.rc; cat > /tmp/dl.sh <<EOF
#!/bin/bash
export HF_XET_HIGH_PERFORMANCE=1
hf download $MODEL_REPO --revision $MODEL_REVISION --local-dir $DIR --max-workers 16 > /tmp/dl.log 2>&1
echo \\\$? > /tmp/dl.rc
EOF
chmod +x /tmp/dl.sh; setsid /tmp/dl.sh </dev/null >/dev/null 2>&1 &" </dev/null || exit 1
  echo "download started in $p"
done
for p in $PODS; do
  until k exec "$p" -- test -f /tmp/dl.rc 2>/dev/null; do sleep 30; done
  rc=$(k exec "$p" -- cat /tmp/dl.rc)
  echo "$p: download exit code $rc"
  [ "$rc" = 0 ] || { k exec "$p" -- tail -20 /tmp/dl.log; exit 1; }
done
ref=""
for p in $PODS; do
  sum=$(k exec "$p" -- bash -c "cd $DIR && find . -type f ! -path './.cache/*' -printf '%s %p\n' | sort -k2 | md5sum")
  size=$(k exec "$p" -- du -sh "$DIR" | cut -f1)
  echo "$p: $size, listing $sum"
  [ -z "$ref" ] && ref=$sum
  [ "$sum" = "$ref" ] || { echo "file listing on $p differs from vllm-head"; exit 1; }
done
echo "model staged at $DIR on all pods"
