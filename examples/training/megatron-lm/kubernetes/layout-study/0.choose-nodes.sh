#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Suggest HEAD_NODE, NEAR_NODE and FAR_NODE from the EC2 network-node labels that Amazon EKS puts on each node
# (topology.k8s.aws/network-node-layer-1..3): HEAD and NEAR share all three layers, FAR shares layers 1 and 2 and
# differs at layer 3. Prints the node groups, then export lines for env_vars. Pick idle nodes if the suggestion
# lands on busy ones; the scripts check the layers again before measuring.
#
# Usage: INSTANCE_TYPE=p5en.48xlarge ./0.choose-nodes.sh
set -euo pipefail
INSTANCE_TYPE=${INSTANCE_TYPE:-p5en.48xlarge}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
kubectl get nodes -l "node.kubernetes.io/instance-type=$INSTANCE_TYPE" -o json > "$TMP/nodes.json"
python3 - "$TMP/nodes.json" <<'PY'
import json
import sys

key = "topology.k8s.aws/network-node-layer-%d"
nodes = {}
with open(sys.argv[1], encoding="utf-8") as fh:
    items = json.load(fh)["items"]
for n in items:
    labels = n["metadata"]["labels"]
    nodes[n["metadata"]["name"]] = tuple(labels.get(key % i) for i in (1, 2, 3))
if not nodes or any(None in v for v in nodes.values()):
    sys.exit("nodes without network-node labels: does the instance type support EC2 instance topology?")
names = {v: f"L3-{chr(97 + i)}" for i, v in enumerate(sorted({v[2] for v in nodes.values()}))}
groups = {}
for name, layers in sorted(nodes.items()):
    groups.setdefault(layers, []).append(name)
for layers, members in sorted(groups.items(), key=lambda g: -len(g[1])):
    print(f"{names[layers[2]]}: {len(members)} nodes: {' '.join(members)}")
pairs = [(layers, m) for layers, m in groups.items() if len(m) >= 2]
if not pairs:
    sys.exit("no two nodes share a layer-3 network node, so there is no near pair")
layers, members = max(pairs, key=lambda g: len(g[1]))
far = [m for lay, ms in groups.items() if lay != layers and lay[:2] == layers[:2] for m in ms]
if not far:
    sys.exit("no node shares layers 1-2 with the near pair but differs at layer 3")
print(f"\nexport HEAD_NODE={members[0]}\nexport NEAR_NODE={members[1]}\nexport FAR_NODE={far[0]}")
PY
