#!/bin/bash

# Easy SSH script for SageMaker HyperPod clusters
# Usage: ./easy-ssh.sh <cluster-name> [region]

set -e

CLUSTER_NAME=${1}
REGION=${2:-us-west-2}

if [ -z "$CLUSTER_NAME" ]; then
    echo "Usage: $0 <cluster-name> [region]"
    echo "Example: $0 ml-cluster us-west-2"
    exit 1
fi

echo "Connecting to HyperPod cluster: $CLUSTER_NAME in region: $REGION"

# Get cluster information
CLUSTER_INFO=$(aws sagemaker describe-cluster --cluster-name "$CLUSTER_NAME" --region "$REGION" 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "Error: Could not find cluster '$CLUSTER_NAME' in region '$REGION'"
    echo "Available clusters:"
    aws sagemaker list-clusters --region "$REGION" --query 'ClusterSummaries[].ClusterName' --output table
    exit 1
fi

# Extract cluster ID from ARN
CLUSTER_ARN=$(echo "$CLUSTER_INFO" | jq -r '.ClusterArn')
CLUSTER_ID=$(echo "$CLUSTER_ARN" | cut -d'/' -f2)

echo "Cluster ID: $CLUSTER_ID"

# Get controller node information
NODES_INFO=$(aws sagemaker list-cluster-nodes --cluster-name "$CLUSTER_NAME" --region "$REGION")
# Try to find login node first, fallback to controller
LOGIN_NODE=$(echo "$NODES_INFO" | jq -r '.ClusterNodeSummaries[] | select(.InstanceGroupName == "login-nodes") | .InstanceId' | head -1)
CONTROLLER_NODE=$(echo "$NODES_INFO" | jq -r '.ClusterNodeSummaries[] | select(.InstanceGroupName == "controller-machine") | .InstanceId' | head -1)

if [ -n "$LOGIN_NODE" ] && [ "$LOGIN_NODE" != "null" ]; then
    NODE_ID="$LOGIN_NODE"
    NODE_GROUP="login-nodes"
    echo "Login node: $LOGIN_NODE"
elif [ -n "$CONTROLLER_NODE" ] && [ "$CONTROLLER_NODE" != "null" ]; then
    NODE_ID="$CONTROLLER_NODE"
    NODE_GROUP="controller-machine"
    echo "Controller node: $CONTROLLER_NODE"
else
    echo "Error: Could not find login or controller node in cluster"
    echo "Available nodes:"
    echo "$NODES_INFO" | jq -r '.ClusterNodeSummaries[] | "\(.InstanceGroupName): \(.InstanceId) (\(.InstanceStatus.Status))"'
    exit 1
fi

# Construct target ID for SSM
TARGET_ID="sagemaker-cluster:${CLUSTER_ID}_${NODE_GROUP}-${NODE_ID}"

echo "Connecting via SSM..."
echo "Target ID: $TARGET_ID"

# Start SSM session
aws ssm start-session --target "$TARGET_ID" --region "$REGION"