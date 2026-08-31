#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Install Dynamo 0.7.0 Platform on HyperPod EKS
# Prerequisites: kubectl configured, helm installed
set -euo pipefail
export AWS_PAGER=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE=dynamo-system
RELEASE_VERSION=0.7.0

# Detect cluster name and region from current kubeconfig context
CURRENT_CTX=$(kubectl config current-context 2>/dev/null)
CLUSTER_NAME="${EKS_CLUSTER_NAME:-$(echo "$CURRENT_CTX" | sed 's|.*/||')}"
REGION="${AWS_REGION:-$(echo "$CURRENT_CTX" | sed -n 's|.*:\([a-z]*-[a-z]*-[0-9]*\):.*|\1|p')}"
REGION="${REGION:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

if [ -z "$CLUSTER_NAME" ]; then
  echo "❌ Could not detect EKS cluster name. Set EKS_CLUSTER_NAME env var."
  exit 1
fi

echo "🚀 Installing Dynamo ${RELEASE_VERSION} on ${CLUSTER_NAME} (${REGION})"

# --- Step 1: EBS CSI Driver (required for etcd/nats PVCs) ---
echo ""
echo "📦 Step 1: EBS CSI Driver..."
ROLE_NAME="AmazonEKS_EBS_CSI_${CLUSTER_NAME}"
ROLE_NAME="${ROLE_NAME:0:64}"

echo "  Ensuring IAM role ${ROLE_NAME}..."
aws iam create-role --role-name "$ROLE_NAME" \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": { "Service": "pods.eks.amazonaws.com" },
      "Action": ["sts:AssumeRole", "sts:TagSession"]
    }]
  }' --no-cli-pager 2>/dev/null || echo "  Role already exists"

aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy --no-cli-pager 2>/dev/null || true

echo "  Adding HyperPod volume permissions..."
aws iam put-role-policy --role-name "$ROLE_NAME" \
  --policy-name HyperPodVolumeAccess \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Effect\": \"Allow\",
        \"Action\": [
          \"sagemaker:AttachClusterNodeVolume\",
          \"sagemaker:DetachClusterNodeVolume\"
        ],
        \"Resource\": \"arn:aws:sagemaker:${REGION}:${ACCOUNT_ID}:cluster/*\"
      },
      {
        \"Effect\": \"Allow\",
        \"Action\": [\"eks:DescribeCluster\"],
        \"Resource\": \"arn:aws:eks:${REGION}:${ACCOUNT_ID}:cluster/${CLUSTER_NAME}\"
      }
    ]
  }" --no-cli-pager 2>/dev/null || true

ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text --no-cli-pager)

echo "  Ensuring EBS CSI addon..."
aws eks create-addon --cluster-name "$CLUSTER_NAME" \
  --addon-name aws-ebs-csi-driver --region "$REGION" --no-cli-pager 2>/dev/null || true

echo "  Ensuring Pod Identity association..."
aws eks create-pod-identity-association --cluster-name "$CLUSTER_NAME" \
  --namespace kube-system --service-account ebs-csi-controller-sa \
  --role-arn "$ROLE_ARN" --region "$REGION" --no-cli-pager 2>/dev/null || true

echo "  Waiting for EBS CSI addon..."
aws eks wait addon-active --cluster-name "$CLUSTER_NAME" \
  --addon-name aws-ebs-csi-driver --region "$REGION" 2>/dev/null || true

kubectl rollout restart deployment ebs-csi-controller -n kube-system 2>/dev/null || true
echo "  Waiting for EBS CSI controller..."
sleep 10
kubectl wait --for=condition=available deployment/ebs-csi-controller \
  -n kube-system --timeout=120s 2>/dev/null || echo "  ⚠️  EBS CSI not ready yet"

# --- Step 2: Download Dynamo charts ---
echo ""
echo "📦 Step 2: Downloading Dynamo charts..."
cd "$SCRIPT_DIR/.."
for chart in dynamo-crds dynamo-platform; do
  [ -f "${chart}-${RELEASE_VERSION}.tgz" ] && echo "  ${chart} already downloaded" && continue
  echo "  Downloading ${chart}..."
  helm fetch "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/${chart}-${RELEASE_VERSION}.tgz"
done

# --- Step 3: Install CRDs ---
echo ""
echo "📋 Step 3: Installing CRDs..."
helm upgrade --install dynamo-crds "dynamo-crds-${RELEASE_VERSION}.tgz" --namespace default

# --- Step 4: Install Platform ---
echo ""
echo "🚀 Step 4: Installing Platform..."
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install dynamo-platform "dynamo-platform-${RELEASE_VERSION}.tgz" \
  --namespace "$NAMESPACE" \
  -f "$SCRIPT_DIR/../manifests/platform/values.yaml"

# The 0.7.0 chart pins a kube-rbac-proxy image that no longer exists (gcr.io/kubebuilder).
# Repoint the operator sidecar to the quay.io/brancz mirror (not exposed as a Helm value).
echo "  Patching operator kube-rbac-proxy sidecar image..."
sleep 5
kubectl -n "$NAMESPACE" set image deployment/dynamo-platform-dynamo-operator-controller-manager \
  kube-rbac-proxy=quay.io/brancz/kube-rbac-proxy:v0.15.0 2>/dev/null || true

# --- Step 5: Create HF token secret ---
echo ""
echo "🔑 Step 5: Creating HF token secret..."
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="${HF_TOKEN:-placeholder}" \
  -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# --- Step 6: Observability ---
echo ""
echo "📊 Step 6: Setting up observability..."
# Apply PodMonitor if Prometheus Operator CRDs exist
if kubectl get crd podmonitors.monitoring.coreos.com >/dev/null 2>&1; then
  kubectl apply -f "$SCRIPT_DIR/../manifests/observability/pod-monitor.yaml"
  echo "  ✅ PodMonitor created"
else
  echo "  ⚠️  Prometheus Operator CRDs not found, skipping PodMonitor"
fi

# Add prometheus scrape annotations to dynamo pods (for HyperPod OTEL collector)
echo "  Adding scrape annotations to platform namespace..."
kubectl annotate namespace "$NAMESPACE" \
  prometheus.io/scrape="true" --overwrite 2>/dev/null || true

# --- Step 7: Wait for platform (operator + etcd + NATS) ---
echo ""
echo "⏳ Step 7: Waiting for platform components..."
kubectl rollout status deployment/dynamo-platform-dynamo-operator-controller-manager -n "$NAMESPACE" --timeout=240s 2>/dev/null || true
kubectl rollout status statefulset/dynamo-platform-etcd -n "$NAMESPACE" --timeout=180s 2>/dev/null || true
kubectl rollout status statefulset/dynamo-platform-nats -n "$NAMESPACE" --timeout=180s 2>/dev/null || true
kubectl get pods -n "$NAMESPACE"

echo ""
echo "✅ Platform ready (operator + etcd + NATS)."
echo "   If etcd/nats are Pending, give EBS volumes 1-2 min to provision."
echo "   Next: ./scripts/02-download-model.sh <gpt-oss|qwen3.6>"
