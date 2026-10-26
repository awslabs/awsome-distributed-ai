#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Render test-internode.yaml with the workers placed by EC2 network topology, from the
# topology.k8s.aws/network-node-layer-3 label that Amazon EKS puts on supported instance types:
#   PLACEMENT=near  every worker under one layer-3 network node: the workers' node selector adds that label with the
#                   value NEAR_LAYER3. Unset, NEAR_LAYER3 is the value most INSTANCE_TYPE nodes share (at least two).
#                   (Pod affinity on the label alone isn't enough: it lets the first worker land on a node that has
#                   no neighbor under its layer-3 node, and the second worker then stays Pending.)
#   PLACEMENT=far   every worker under a different layer-3 network node in the same Availability Zone
#                   (the manifest's zone affinity, plus pod anti-affinity on the label)
# Optional TEST_ARGS replaces test_ep.py's arguments, for example a decode-sized batch with Kimi K2's shapes:
#   TEST_ARGS="--test-first-only --num-tokens=128 --num-topk=8 --num-sms=12 --hidden 7168 --num-experts 384"
#
# Usage: source env_vars && PLACEMENT=near ./placement.sh | kubectl apply -f -
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PLACEMENT=${PLACEMENT:?set PLACEMENT=near or PLACEMENT=far}
TEST_ARGS=${TEST_ARGS:-}
: "${IMAGE_URI:?source env_vars}" "${INSTANCE_TYPE:?}" "${GPU_PER_NODE:?}" "${EFA_PER_NODE:?}" "${NUM_NODES:?}"
NEAR_LAYER3=${NEAR_LAYER3:-}
if [ "$PLACEMENT" = near ] && [ -z "$NEAR_LAYER3" ]; then
  NEAR_LAYER3=$(kubectl get nodes -l "node.kubernetes.io/instance-type=$INSTANCE_TYPE" \
    -o jsonpath='{range .items[*]}{.metadata.labels.topology\.k8s\.aws/network-node-layer-3}{"\n"}{end}' |
    grep . | sort | uniq -c | sort -rn | awk '$1 >= 2 {print $2; exit}')
  [ -n "$NEAR_LAYER3" ] || { echo "no two $INSTANCE_TYPE nodes share a layer-3 network node" >&2; exit 1; }
  echo "near: workers on layer-3 network node $NEAR_LAYER3 (set NEAR_LAYER3 to choose another)" >&2
fi
# shellcheck disable=SC2016  # the list names the variables envsubst may substitute
envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES' < "$HERE/test-internode.yaml" |
  python3 -c '
import sys

placement, test_args, near_layer3 = sys.argv[1], sys.argv[2], sys.argv[3]
doc = sys.stdin.read()
zone = "                  topologyKey: topology.kubernetes.io/zone\n"
layer3 = "topology.k8s.aws/network-node-layer-3"
selector = "          nodeSelector:\n            node.kubernetes.io/instance-type: "
launcher, sep, worker = doc.partition("    Worker:\n")
if doc.count(zone) != 1 or not sep or worker.count(selector) != 1:
    sys.exit("test-internode.yaml changed: the worker affinity or node selector was not found")
if placement == "near":
    pinned = selector.replace("nodeSelector:\n", f"nodeSelector:\n            {layer3}: {near_layer3}\n")
    doc = launcher + sep + worker.replace(selector, pinned)
elif placement == "far":
    anti = (
        "            podAntiAffinity:\n"
        "              requiredDuringSchedulingIgnoredDuringExecution:\n"
        "                - labelSelector:\n"
        "                    matchExpressions:\n"
        "                      - key: training.kubeflow.org/job-name\n"
        "                        operator: In\n"
        "                        values:\n"
        "                          - deepep-v2-internode\n"
        "                      - key: training.kubeflow.org/job-role\n"
        "                        operator: In\n"
        "                        values:\n"
        "                          - worker\n"
        f"                  topologyKey: {layer3}\n"
    )
    doc = doc.replace(zone, zone + anti)
else:
    sys.exit("PLACEMENT must be near or far")
default_args = "--test-first-only --num-tokens=8192 --num-topk=8 --num-sms=12"
if test_args:
    if doc.count(default_args) != 1:
        sys.exit("test-internode.yaml changed: the test_ep.py arguments were not found")
    doc = doc.replace(default_args, test_args)
sys.stdout.write(doc)
' "$PLACEMENT" "$TEST_ARGS" "$NEAR_LAYER3"
