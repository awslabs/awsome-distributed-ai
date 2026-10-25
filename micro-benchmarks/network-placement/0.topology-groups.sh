#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Show how the cluster's nodes, and the probe pods on them, group by EC2 network node, from the
# topology.k8s.aws/network-node-layer-1..3 node labels. Raw network node IDs are printed relabelled (L3-a, ...).
#
# Usage: INSTANCE_TYPE=p5en.48xlarge NAMESPACE=default ./0.topology-groups.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
INSTANCE_TYPE=${INSTANCE_TYPE:-p5en.48xlarge}
NAMESPACE=${NAMESPACE:-default}
SELECTOR=${SELECTOR:-app=placement-probe}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

kubectl get nodes -o json > "$TMP/nodes.json"
echo "== Nodes of type $INSTANCE_TYPE"
python3 "$HERE/topology.py" groups --nodes "$TMP/nodes.json" --instance-type "$INSTANCE_TYPE"

kubectl -n "$NAMESPACE" get pods -l "$SELECTOR" -o json > "$TMP/pods.json"
if python3 -c 'import json, sys; sys.exit(0 if any(p["status"].get("phase") == "Running" for p in json.load(open(sys.argv[1]))["items"]) else 1)' "$TMP/pods.json"; then
  echo; echo "== Running probe pods ($SELECTOR)"
  python3 "$HERE/topology.py" groups --nodes "$TMP/nodes.json" --instance-type "$INSTANCE_TYPE" --pods "$TMP/pods.json"
fi
