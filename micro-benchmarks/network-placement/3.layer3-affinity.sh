#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Check that the scheduler can keep a group of pods under one layer-3 network node with plain pod affinity:
# apply kubernetes/layer3-affinity.yaml, see where the follower pod lands relative to the anchor, and clean up.
# By default the anchor goes on the node of a probe pod in the largest layer-3 group, so a second node qualifies.
#
# Usage: NAMESPACE=default [ANCHOR_NODE=<node name>] ./3.layer3-affinity.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NAMESPACE=${NAMESPACE:-default}; SELECTOR=${SELECTOR:-app=placement-probe}
IMAGE=${IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-efa}   # already on the probe nodes
WAIT=${WAIT:-180}
K="kubectl -n $NAMESPACE"
TMP=$(mktemp -d); trap '$K delete pod placement-anchor placement-follower --ignore-not-found --wait=false >/dev/null; rm -rf "$TMP"' EXIT

kubectl get nodes -o json > "$TMP/nodes.json"
if [ -z "${ANCHOR_NODE:-}" ]; then
  $K get pods -l "$SELECTOR" -o json > "$TMP/pods.json"
  ANCHOR_NODE=$(python3 - "$TMP/nodes.json" "$TMP/pods.json" <<'PY'
import collections, json, sys
labels = {n["metadata"]["name"]: n["metadata"]["labels"] for n in json.load(open(sys.argv[1]))["items"]}
nodes = sorted({p["spec"]["nodeName"] for p in json.load(open(sys.argv[2]))["items"] if p["spec"].get("nodeName")})
key = "topology.k8s.aws/network-node-layer-3"
groups = collections.defaultdict(list)
for n in nodes:
    if labels.get(n, {}).get(key):
        groups[labels[n][key]].append(n)
print(max(groups.values(), key=len)[0] if groups else "")
PY
)
fi
[ -n "$ANCHOR_NODE" ] || { echo "no labelled probe node found; set ANCHOR_NODE"; exit 1; }
sed -e "s|ANCHOR_NODE|$ANCHOR_NODE|" -e "s|IMAGE|$IMAGE|" "$HERE/kubernetes/layer3-affinity.yaml" > "$TMP/affinity.yaml"
$K apply -f "$TMP/affinity.yaml" > /dev/null || exit 1

t=0
while [ $t -lt "$WAIT" ]; do
  FOLLOWER_NODE=$($K get pod placement-follower -o jsonpath='{.spec.nodeName}' 2>/dev/null)
  [ -n "$FOLLOWER_NODE" ] && break
  sleep 5; t=$((t + 5))
done
if [ -z "$FOLLOWER_NODE" ]; then
  echo "follower not scheduled after ${WAIT}s:"
  $K get events --field-selector involvedObject.name=placement-follower -o custom-columns=REASON:.reason,MESSAGE:.message | tail -3
  echo "(expected if no other schedulable node shares the anchor's layer-3 network node)"
  exit 1
fi
python3 - "$TMP/nodes.json" "$ANCHOR_NODE" "$FOLLOWER_NODE" <<'PY'
import json, sys
labels = {n["metadata"]["name"]: n["metadata"]["labels"] for n in json.load(open(sys.argv[1]))["items"]}
a, f = sys.argv[2], sys.argv[3]
same = [labels[a].get(f"topology.k8s.aws/network-node-layer-{i}") == labels[f].get(f"topology.k8s.aws/network-node-layer-{i}")
        for i in (1, 2, 3)]
print(f"anchor   {a}\nfollower {f}")
print(f"different host: {a != f}; same network node at layers 1, 2, 3: {same}")
sys.exit(0 if a != f and all(same) else 1)
PY
