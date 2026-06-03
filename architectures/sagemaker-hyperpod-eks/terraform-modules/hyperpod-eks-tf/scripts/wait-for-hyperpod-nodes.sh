#!/bin/bash
set -e

REGION=$1
EKS_CLUSTER_NAME=$2
HYPERPOD_CLUSTER_NAME=$3

aws eks update-kubeconfig --region $REGION --name $EKS_CLUSTER_NAME

echo "Waiting for HyperPod nodes to join EKS cluster..."
for i in {1..60}; do
  node_count=$(kubectl get nodes -l sagemaker.amazonaws.com/cluster-name=$HYPERPOD_CLUSTER_NAME --no-headers 2>/dev/null | wc -l)
  if [ "$node_count" -gt 0 ]; then
    echo "Found $node_count HyperPod nodes in EKS cluster"
    break
  fi
  echo "No HyperPod nodes found yet, waiting... ($i/60)"
  sleep 30
done

if [ "$node_count" -eq 0 ]; then
  echo "Timeout: No HyperPod nodes found after 30 minutes"
  exit 1
fi

echo "Waiting for EKS Pod Identity Agent to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=eks-pod-identity-agent -n kube-system --timeout=300s
echo "Pod Identity Agent is ready"