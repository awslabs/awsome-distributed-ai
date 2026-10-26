#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Render test-internode.yaml with the workers placed by EC2 network topology, from the
# topology.k8s.aws/network-node-layer-3 label that Amazon EKS puts on supported instance types:
#   PLACEMENT=near  every worker under the same layer-3 network node (pod affinity on the label)
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
# shellcheck disable=SC2016  # the list names the variables envsubst may substitute
envsubst '$IMAGE_URI $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $NUM_NODES' < "$HERE/test-internode.yaml" |
  python3 -c '
import sys

placement, test_args = sys.argv[1], sys.argv[2]
doc = sys.stdin.read()
zone = "                  topologyKey: topology.kubernetes.io/zone\n"
layer3 = "topology.k8s.aws/network-node-layer-3"
if doc.count(zone) != 1:
    sys.exit("test-internode.yaml changed: the worker zone affinity was not found")
if placement == "near":
    doc = doc.replace(zone, zone.replace("topology.kubernetes.io/zone", layer3))
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
' "$PLACEMENT" "$TEST_ARGS"
