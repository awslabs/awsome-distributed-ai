#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# ============================================================================
# Remove a CPU instance group from a HyperPod (EKS) cluster.
#
# UpdateCluster replaces the full InstanceGroups list, so this submits the
# current set MINUS the named group, which scales it down and deletes it.
# Use this to stop the cost of the reward CPU pool when you are done.
#
# Usage:
#   CLUSTER_NAME=<hp-cluster> [REGION=us-east-1] \
#   GROUP_NAME=reward-spot-c5 \
#   infra/remove_cpu_instance_group.sh
# ============================================================================
set -euo pipefail

: "${CLUSTER_NAME:?Set CLUSTER_NAME}"
: "${GROUP_NAME:?Set GROUP_NAME to the instance group to remove}"
REGION="${REGION:-us-east-1}"

TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}"' EXIT

aws sagemaker describe-cluster --region "${REGION}" \
    --cluster-name "${CLUSTER_NAME}" > "${TMP}/cluster.json"

CLUSTER_NAME="${CLUSTER_NAME}" GROUP_NAME="${GROUP_NAME}" TMP="${TMP}" \
python3 - <<'PY'
import json, os, sys
d = json.load(open(os.path.join(os.environ["TMP"], "cluster.json")))
remove = os.environ["GROUP_NAME"]
groups = d["InstanceGroups"]
if not any(g["InstanceGroupName"] == remove for g in groups):
    sys.exit(f"Instance group {remove} not found; nothing to do.")

def keep(g):
    out = {
        "InstanceGroupName": g["InstanceGroupName"],
        "InstanceType": g["InstanceType"],
        "InstanceCount": g["TargetCount"],
        "LifeCycleConfig": g["LifeCycleConfig"],
        "ExecutionRole": g["ExecutionRole"],
        "ThreadsPerCore": g.get("ThreadsPerCore", 1),
    }
    if "InstanceStorageConfigs" in g:
        out["InstanceStorageConfigs"] = g["InstanceStorageConfigs"]
    if "OverrideVpcConfig" in g:
        out["OverrideVpcConfig"] = g["OverrideVpcConfig"]
    if "CapacityRequirements" in g:
        out["CapacityRequirements"] = g["CapacityRequirements"]
    return out

remaining = [keep(g) for g in groups if g["InstanceGroupName"] != remove]
json.dump({"ClusterName": os.environ["CLUSTER_NAME"], "InstanceGroups": remaining},
          open(os.path.join(os.environ["TMP"], "update.json"), "w"), indent=2)
print(f"Removing {remove}; {len(remaining)} group(s) will remain.")
PY

aws sagemaker update-cluster --region "${REGION}" \
    --cli-input-json "file://${TMP}/update.json"
echo "Submitted removal of ${GROUP_NAME}."
