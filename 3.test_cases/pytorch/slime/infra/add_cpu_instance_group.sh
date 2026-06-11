#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# ============================================================================
# Add a CPU instance group to an existing SageMaker HyperPod (EKS) cluster.
#
# Used to provision the CPU reward pool (e.g. c5) that hosts the remote reward
# service (see kubernetes/reward-service.yaml). The new group is placed in the
# SAME subnet/AZ and uses the SAME security group, execution role, and lifecycle
# config as an existing CPU ("general") group, so it joins EKS identically and
# stays in-AZ with the GPU pool. No EFA is requested -- the reward RPC is HTTP.
#
# CAPACITY_TYPE defaults to "spot": the reward service is stateless and fault
# tolerant (SLIME's remote_rm client retries with backoff), so EC2 Spot is a
# good fit and saves up to ~90% vs On-Demand. HyperPod handles Spot interruptions
# by tainting the node, gracefully evicting pods (terminationGracePeriodSeconds),
# and replacing capacity. Requires Continuous provisioning (this cluster uses it).
# See https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-spot.html
#
# NOTE: CapacityRequirements (Spot vs On-Demand) CANNOT be changed after a group
# is created -- pick the right CAPACITY_TYPE up front.
#
# HyperPod's UpdateCluster replaces the FULL InstanceGroups list, so this script
# reads the current cluster, appends the new group, and submits the union.
#
# Usage:
#   CLUSTER_NAME=<hp-cluster> [REGION=us-east-1] \
#   [GROUP_NAME=reward-spot-c5] [INSTANCE_TYPE=ml.c5.4xlarge] \
#   [INSTANCE_COUNT=4] [CAPACITY_TYPE=spot|on-demand] \
#   [TEMPLATE_GROUP=<existing-cpu-group>] [EBS_GB=100] \
#   infra/add_cpu_instance_group.sh
#
# Requires: awscli, python3.
# ============================================================================
set -euo pipefail

: "${CLUSTER_NAME:?Set CLUSTER_NAME to your HyperPod cluster name}"
REGION="${REGION:-us-east-1}"
GROUP_NAME="${GROUP_NAME:-reward-spot-c5}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.c5.4xlarge}"
INSTANCE_COUNT="${INSTANCE_COUNT:-4}"
CAPACITY_TYPE="${CAPACITY_TYPE:-spot}"   # spot (default) | on-demand
EBS_GB="${EBS_GB:-100}"
TEMPLATE_GROUP="${TEMPLATE_GROUP:-}"   # auto-detect a non-GPU group if empty

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "[1/3] Reading current cluster config for ${CLUSTER_NAME} ..."
aws sagemaker describe-cluster --region "${REGION}" \
    --cluster-name "${CLUSTER_NAME}" > "${TMP}/cluster.json"

echo "[2/3] Building UpdateCluster payload (existing groups + ${GROUP_NAME}) ..."
CLUSTER_NAME="${CLUSTER_NAME}" GROUP_NAME="${GROUP_NAME}" \
INSTANCE_TYPE="${INSTANCE_TYPE}" INSTANCE_COUNT="${INSTANCE_COUNT}" \
CAPACITY_TYPE="${CAPACITY_TYPE}" \
EBS_GB="${EBS_GB}" TEMPLATE_GROUP="${TEMPLATE_GROUP}" TMP="${TMP}" \
python3 - <<'PY'
import json, os, sys

d = json.load(open(os.path.join(os.environ["TMP"], "cluster.json")))
groups = d["InstanceGroups"]

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
    # Preserve Spot/On-Demand on existing groups (immutable; must be resubmitted).
    if "CapacityRequirements" in g:
        out["CapacityRequirements"] = g["CapacityRequirements"]
    return out

new_name = os.environ["GROUP_NAME"]
if any(g["InstanceGroupName"] == new_name for g in groups):
    sys.exit(f"Instance group {new_name} already exists; nothing to do.")

# Pick a template: explicit TEMPLATE_GROUP, else the first non-p/g GPU group.
tmpl_name = os.environ.get("TEMPLATE_GROUP") or ""
if tmpl_name:
    tmpl = next(g for g in groups if g["InstanceGroupName"] == tmpl_name)
else:
    tmpl = next(g for g in groups
                if not g["InstanceType"].split(".")[1].startswith(("p", "g")))
print(f"  template group: {tmpl['InstanceGroupName']} ({tmpl['InstanceType']})")
print(f"  subnet/AZ + SG + role + lifecycle inherited from template")

capacity_type = os.environ.get("CAPACITY_TYPE", "spot").strip().lower()
new_group = {
    "InstanceGroupName": new_name,
    "InstanceType": os.environ["INSTANCE_TYPE"],
    "InstanceCount": int(os.environ["INSTANCE_COUNT"]),
    "LifeCycleConfig": tmpl["LifeCycleConfig"],
    "ExecutionRole": tmpl["ExecutionRole"],
    "ThreadsPerCore": 1,
    "InstanceStorageConfigs": [
        {"EbsVolumeConfig": {"VolumeSizeInGB": int(os.environ["EBS_GB"]),
                             "RootVolume": False}}
    ],
    "OverrideVpcConfig": tmpl["OverrideVpcConfig"],
}
# Spot capacity for this group. CapacityRequirements is immutable after create;
# omitting it (on-demand) is the only alternative. On-Demand groups simply leave
# the key out.
if capacity_type == "spot":
    new_group["CapacityRequirements"] = {"Spot": {}}
elif capacity_type not in ("on-demand", "ondemand", "on_demand"):
    sys.exit(f"CAPACITY_TYPE must be 'spot' or 'on-demand', got: {capacity_type}")
print(f"  capacity type: {capacity_type}")

payload_groups = [keep(g) for g in groups]
payload_groups.append(new_group)

json.dump({"ClusterName": os.environ["CLUSTER_NAME"],
           "InstanceGroups": payload_groups},
          open(os.path.join(os.environ["TMP"], "update.json"), "w"), indent=2)

print("  groups in payload:")
for g in payload_groups:
    cap = "spot" if g.get("CapacityRequirements", {}).get("Spot") is not None else "on-demand"
    print(f"    {g['InstanceGroupName']:35s} {g['InstanceType']:16s} x{g['InstanceCount']}  [{cap}]")
PY

echo "[3/3] Submitting UpdateCluster ..."
aws sagemaker update-cluster --region "${REGION}" \
    --cli-input-json "file://${TMP}/update.json"

echo
echo "Submitted. The new group provisions in ~10-15 min. Watch with:"
echo "  aws sagemaker describe-cluster --region ${REGION} --cluster-name ${CLUSTER_NAME} \\"
echo "    --query 'InstanceGroups[].[InstanceGroupName,CurrentCount,TargetCount,Status]' --output table"
echo "  kubectl get nodes -l node.kubernetes.io/instance-type=${INSTANCE_TYPE}"
echo
echo "Then point the reward service at it: REWARD_NODE_GROUP=${GROUP_NAME}"
