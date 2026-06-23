#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Discover the P6e-GB200 NVLink domain (clique) membership.
#
# The NVLink domain corresponds to the UltraServer, identified by capacityBlockId from
# the EC2 Instance Topology API and surfaced on EKS as the nvidia.com/gpu.clique label.
# Use this to co-schedule a job onto exactly one 72-GPU NVLink domain (do not rely on
# `nvidia-smi topo -m`, which only sees the local host, not the cross-instance domain).
set -euo pipefail

echo "== EC2 Instance Topology (NVLink domain = capacityBlockId) =="
# Requires ec2:DescribeInstanceTopology. Groups instances by their UltraServer.
aws ec2 describe-instance-topology \
  --filters "Name=instance-type,Values=p6e-gb200.36xlarge" \
  --query 'Instances[].{Instance:InstanceId,CapacityBlock:CapacityBlockId,NetworkNodes:NetworkNodes}' \
  --output table 2>/dev/null || echo "(DescribeInstanceTopology unavailable -- check IAM / region)"

echo
echo "== Kubernetes clique labels (one value per NVLink domain) =="
if command -v kubectl >/dev/null 2>&1; then
  kubectl get nodes -L nvidia.com/gpu.clique \
    -l node.kubernetes.io/instance-type=p6e-gb200.36xlarge 2>/dev/null \
    || echo "(no kubectl context or no GB200 nodes)"
else
  echo "(kubectl not present -- Slurm cluster?)"
fi

echo
echo "Co-schedule a job to ONE clique value above (= one 72-GPU NVLink domain)."
echo "Two distinct clique values = two UltraServers bridged by EFA (cross-domain)."
