#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}========================================${NC}"
echo -e "${RED}EKS GPU Observability Cleanup${NC}"
echo -e "${RED}========================================${NC}"
echo ""

# Function to prompt for input with default value
prompt_with_default() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    
    if [ -n "$default" ]; then
        read -p "$prompt [$default]: " input
        eval "$var_name=\"${input:-$default}\""
    else
        read -p "$prompt: " input
        eval "$var_name=\"$input\""
    fi
}

# Collect environment variables
echo -e "${YELLOW}Configuration${NC}"
echo ""

prompt_with_default "EKS Cluster Region" "us-east-1" AWS_REGION
prompt_with_default "AMP/AMG Region" "us-east-1" AWS_REGION_AMGP
prompt_with_default "EKS Cluster Name" "" CLUSTER_NAME
prompt_with_default "CloudFormation Stack Name" "eks-observability" STACK_NAME

echo ""
echo -e "${RED}WARNING: This will delete:${NC}"
echo "  - ADOT Collector and configuration"
echo "  - ADOT Operator add-on"
echo "  - EKS Pod Identity Agent add-on"
echo "  - cert-manager"
echo "  - IAM role: adot-collector-prometheus-role"
echo "  - CloudFormation stack: $STACK_NAME (AMP & AMG workspaces)"
echo "  - Custom DCGM metrics ConfigMap"
echo ""

read -p "Are you sure you want to continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 1: Delete ADOT Collector${NC}"
echo ""

kubectl delete -f adot-collector-prometheus-generated.yaml 2>/dev/null || \
kubectl delete -f adot-collector-prometheus.yaml 2>/dev/null || \
echo "ADOT Collector not found"

echo -e "${GREEN}✓ ADOT Collector deleted${NC}"
echo ""

echo -e "${YELLOW}Step 2: Delete ADOT Operator add-on${NC}"
echo ""

aws eks delete-addon \
  --cluster-name $CLUSTER_NAME \
  --addon-name adot \
  --region $AWS_REGION 2>/dev/null || echo "ADOT add-on not found"

echo "Waiting for ADOT add-on deletion..."
sleep 10

echo -e "${GREEN}✓ ADOT Operator deleted${NC}"
echo ""

echo -e "${YELLOW}Step 3: Delete Pod Identity Agent add-on${NC}"
echo ""

aws eks delete-addon \
  --cluster-name $CLUSTER_NAME \
  --addon-name eks-pod-identity-agent \
  --region $AWS_REGION 2>/dev/null || echo "Pod Identity Agent not found"

echo "Waiting for Pod Identity Agent deletion..."
sleep 10

echo -e "${GREEN}✓ Pod Identity Agent deleted${NC}"
echo ""

echo -e "${YELLOW}Step 4: Delete cert-manager${NC}"
echo ""

kubectl delete -f https://github.com/cert-manager/cert-manager/releases/download/v1.19.2/cert-manager.yaml 2>/dev/null || echo "cert-manager not found"

echo -e "${GREEN}✓ cert-manager deleted${NC}"
echo ""

echo -e "${YELLOW}Step 5: Delete IAM role${NC}"
echo ""

# Detach policies
aws iam detach-role-policy \
  --role-name adot-collector-prometheus-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess 2>/dev/null || true

aws iam detach-role-policy \
  --role-name adot-collector-prometheus-role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy 2>/dev/null || true

# Delete role
aws iam delete-role \
  --role-name adot-collector-prometheus-role 2>/dev/null || echo "IAM role not found"

echo -e "${GREEN}✓ IAM role deleted${NC}"
echo ""

echo -e "${YELLOW}Step 6: Delete CloudFormation stack (AMP & AMG)${NC}"
echo ""

aws cloudformation delete-stack \
  --stack-name $STACK_NAME \
  --region $AWS_REGION_AMGP 2>/dev/null || echo "CloudFormation stack not found"

echo "Waiting for CloudFormation stack deletion (this takes ~5 minutes)..."
aws cloudformation wait stack-delete-complete \
  --stack-name $STACK_NAME \
  --region $AWS_REGION_AMGP 2>/dev/null || echo "Stack deletion complete or not found"

echo -e "${GREEN}✓ CloudFormation stack deleted${NC}"
echo ""

echo -e "${YELLOW}Step 7: Reset DCGM Exporter to default metrics${NC}"
echo ""

# Delete custom metrics ConfigMap
kubectl delete configmap custom-dcgm-metrics -n gpu-operator 2>/dev/null || echo "ConfigMap not found"

# Reset GPU Operator to default
helm upgrade gpu-operator nvidia/gpu-operator \
  -n gpu-operator \
  --reuse-values \
  --set dcgmExporter.config.name="" 2>/dev/null || echo "GPU Operator not found or already reset"

# Restart DCGM Exporter
kubectl rollout restart daemonset/nvidia-dcgm-exporter -n gpu-operator 2>/dev/null || echo "DCGM Exporter not found"

echo -e "${GREEN}✓ DCGM Exporter reset to defaults${NC}"
echo ""

echo -e "${YELLOW}Step 8: Cleanup temporary files${NC}"
echo ""

rm -f adot-pod-identity-trust-policy.json
rm -f adot-collector-prometheus-generated.yaml

echo -e "${GREEN}✓ Temporary files deleted${NC}"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cleanup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
