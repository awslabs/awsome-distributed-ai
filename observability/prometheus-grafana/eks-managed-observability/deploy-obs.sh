#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}EKS GPU Observability Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Helm >= 3.14 is required for `--reset-then-reuse-values` (introduced in
# v3.14.0) used by the GPU Operator upgrades in Step 7. Fail loudly up front
# rather than aborting mid-deploy with `unknown flag`.
if ! helm version --short 2>/dev/null | grep -qE 'v3\.(1[4-9]|[2-9][0-9])'; then
    echo -e "${RED}Helm >= 3.14 is required (for --reset-then-reuse-values).${NC}"
    echo -e "${RED}Current version: $(helm version --short 2>/dev/null || echo 'helm not found')${NC}"
    exit 1
fi

# envsubst renders the ADOT Collector template in Step 6.
if ! command -v envsubst >/dev/null 2>&1; then
    echo -e "${RED}envsubst is required (install the gettext package).${NC}"
    exit 1
fi

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
echo -e "${YELLOW}Step 1: Configuration${NC}"
echo ""

prompt_with_default "EKS Cluster Region" "us-east-1" AWS_REGION
prompt_with_default "AMP/AMG Region (must support AMG)" "us-east-1" AWS_REGION_AMGP
prompt_with_default "EKS Cluster Name" "" CLUSTER_NAME
prompt_with_default "CloudFormation Stack Name" "eks-observability" STACK_NAME
prompt_with_default "Scrape DCGM Exporter GPU metrics (y/n)" "y" ENABLE_DCGM_METRICS
prompt_with_default "Scrape SGLang metrics (y/n)" "n" ENABLE_SGLANG_METRICS
echo "  (EFA/RDMA metrics need the EFA node exporter, see ../../efa-node-exporter/EKS/README.md)"
prompt_with_default "Scrape EFA node exporter RDMA metrics (y/n)" "n" ENABLE_EFA_METRICS

# Collect the selected scrape jobs from scrape-configs/
SCRAPE_CONFIG_FILES=()
[ "$ENABLE_DCGM_METRICS" = "y" ] && SCRAPE_CONFIG_FILES+=(scrape-configs/dcgm-exporter.yaml)
[ "$ENABLE_SGLANG_METRICS" = "y" ] && SCRAPE_CONFIG_FILES+=(scrape-configs/sglang.yaml)
[ "$ENABLE_EFA_METRICS" = "y" ] && SCRAPE_CONFIG_FILES+=(scrape-configs/efa-exporter.yaml)
if [ ${#SCRAPE_CONFIG_FILES[@]} -eq 0 ]; then
    echo -e "${RED}At least one metrics source must be enabled.${NC}"
    exit 1
fi

# Get AWS Account ID
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo ""
echo -e "${GREEN}Configuration Summary:${NC}"
echo "  EKS Cluster Region: $AWS_REGION"
echo "  AMP/AMG Region: $AWS_REGION_AMGP"
echo "  Cluster Name: $CLUSTER_NAME"
echo "  Stack Name: $STACK_NAME"
echo "  Scrape Configs: ${SCRAPE_CONFIG_FILES[*]}"
echo "  AWS Account ID: $AWS_ACCOUNT_ID"
echo ""

read -p "Continue with this configuration? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 2: Deploy CloudFormation Stack (AMP & AMG)${NC}"
echo ""

aws cloudformation create-stack \
  --stack-name $STACK_NAME \
  --template-body file://cluster-observability.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --region $AWS_REGION_AMGP

echo "Waiting for CloudFormation stack to complete (this takes ~5 minutes)..."
aws cloudformation wait stack-create-complete \
  --stack-name $STACK_NAME \
  --region $AWS_REGION_AMGP

echo -e "${GREEN}✓ CloudFormation stack created${NC}"

# Get stack outputs
export AMP_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME \
  --query 'Stacks[0].Outputs[?OutputKey==`AMPRemoteWriteURL`].OutputValue' \
  --output text \
  --region $AWS_REGION_AMGP)

export AMP_QUERY_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME \
  --query 'Stacks[0].Outputs[?OutputKey==`AMPEndPointUrl`].OutputValue' \
  --output text \
  --region $AWS_REGION_AMGP)

export GRAFANA_WORKSPACE_URL=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME \
  --query 'Stacks[0].Outputs[?OutputKey==`GrafanaWorkspaceURL`].OutputValue' \
  --output text \
  --region $AWS_REGION_AMGP)

echo "  AMP Endpoint: $AMP_ENDPOINT"
echo "  Grafana URL: $GRAFANA_WORKSPACE_URL"
echo ""

echo -e "${YELLOW}Step 3: Install EKS Pod Identity Agent${NC}"
echo ""

# Check if Pod Identity Agent is already installed
POD_IDENTITY_STATUS=$(aws eks describe-addon \
  --cluster-name $CLUSTER_NAME \
  --addon-name eks-pod-identity-agent \
  --region $AWS_REGION \
  --query 'addon.status' \
  --output text 2>/dev/null || echo "NOT_FOUND")

if [ "$POD_IDENTITY_STATUS" = "ACTIVE" ]; then
    echo -e "${GREEN}✓ Pod Identity Agent already installed and active${NC}"
elif [ "$POD_IDENTITY_STATUS" = "NOT_FOUND" ]; then
    aws eks create-addon \
      --cluster-name $CLUSTER_NAME \
      --addon-name eks-pod-identity-agent \
      --region $AWS_REGION

    echo "Waiting for Pod Identity Agent to be active..."
    aws eks wait addon-active \
      --cluster-name $CLUSTER_NAME \
      --addon-name eks-pod-identity-agent \
      --region $AWS_REGION

    echo -e "${GREEN}✓ Pod Identity Agent installed${NC}"
else
    echo "Pod Identity Agent status: $POD_IDENTITY_STATUS"
    echo "Waiting for it to become active..."
    aws eks wait addon-active \
      --cluster-name $CLUSTER_NAME \
      --addon-name eks-pod-identity-agent \
      --region $AWS_REGION
    echo -e "${GREEN}✓ Pod Identity Agent is now active${NC}"
fi
echo ""

echo -e "${YELLOW}Step 4: Install cert-manager${NC}"
echo ""

kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.19.2/cert-manager.yaml

echo "Waiting for cert-manager to be ready..."
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/instance=cert-manager \
  -n cert-manager \
  --timeout=300s

echo -e "${GREEN}✓ cert-manager installed${NC}"
echo ""

echo -e "${YELLOW}Step 5: Install ADOT Operator${NC}"
echo ""

# Create IAM role trust policy
cat > adot-pod-identity-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "pods.eks.amazonaws.com"},
    "Action": ["sts:AssumeRole", "sts:TagSession"]
  }]
}
EOF

# Create IAM role
aws iam create-role \
  --role-name adot-collector-prometheus-role \
  --assume-role-policy-document file://adot-pod-identity-trust-policy.json \
  --description "IAM role for ADOT Collector to write metrics to Amazon Managed Prometheus" 2>/dev/null || echo "IAM role already exists"

# Attach policies
aws iam attach-role-policy \
  --role-name adot-collector-prometheus-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess 2>/dev/null || true

aws iam attach-role-policy \
  --role-name adot-collector-prometheus-role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy 2>/dev/null || true

export ADOT_ROLE_ARN=$(aws iam get-role \
  --role-name adot-collector-prometheus-role \
  --query 'Role.Arn' \
  --output text)

echo "  ADOT Role ARN: $ADOT_ROLE_ARN"

# Check if ADOT add-on is already installed
ADOT_STATUS=$(aws eks describe-addon \
  --cluster-name $CLUSTER_NAME \
  --addon-name adot \
  --region $AWS_REGION \
  --query 'addon.status' \
  --output text 2>/dev/null || echo "NOT_FOUND")

if [ "$ADOT_STATUS" = "ACTIVE" ]; then
    echo -e "${GREEN}✓ ADOT add-on already installed and active${NC}"
elif [ "$ADOT_STATUS" = "NOT_FOUND" ]; then
    # Install ADOT add-on
    aws eks create-addon \
      --cluster-name $CLUSTER_NAME \
      --addon-name adot \
      --addon-version v0.141.0-eksbuild.1 \
      --region $AWS_REGION

    echo "Waiting for ADOT add-on to be active..."
    aws eks wait addon-active \
      --cluster-name $CLUSTER_NAME \
      --addon-name adot \
      --region $AWS_REGION

    echo -e "${GREEN}✓ ADOT add-on installed${NC}"
else
    echo "ADOT add-on status: $ADOT_STATUS"
    echo "Waiting for it to become active..."
    aws eks wait addon-active \
      --cluster-name $CLUSTER_NAME \
      --addon-name adot \
      --region $AWS_REGION
    echo -e "${GREEN}✓ ADOT add-on is now active${NC}"
fi

# Create Pod Identity association
aws eks create-pod-identity-association \
  --cluster-name $CLUSTER_NAME \
  --namespace adot-col \
  --service-account adot-col-prom-metrics \
  --role-arn $ADOT_ROLE_ARN \
  --region $AWS_REGION 2>/dev/null || echo "Pod Identity association already exists"

# Pod Identity credentials are injected when a pod is created. Give the new
# association a few seconds to propagate so the collector pod created in
# Step 6 gets them; otherwise it falls back to the node role and remote write
# to AMP fails with 403.
sleep 10

echo -e "${GREEN}✓ ADOT Operator installed${NC}"
echo ""

echo -e "${YELLOW}Step 6: Deploy ADOT Collector${NC}"
echo ""

# Render the ADOT Collector template. The scrape jobs are indented to sit
# under scrape_configs (the first line inherits the placeholder's indent).
export SCRAPE_CONFIGS=$(cat "${SCRAPE_CONFIG_FILES[@]}" | sed '1!s/^/          /')
# envsubst only sees exported variables; prompt_with_default does not export.
export AWS_REGION_AMGP
envsubst '${AMP_ENDPOINT} ${AWS_REGION_AMGP} ${SCRAPE_CONFIGS}' \
  < adot-collector-prometheus.yaml > adot-collector-prometheus-generated.yaml

kubectl apply -f adot-collector-prometheus-generated.yaml

echo "Waiting for ADOT Collector to be ready..."
sleep 10
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/component=opentelemetry-collector \
  -n adot-col \
  --timeout=120s 2>/dev/null || echo "Collector starting..."

echo -e "${GREEN}✓ ADOT Collector deployed${NC}"
echo ""

echo -e "${YELLOW}Step 7: Configure Custom DCGM Metrics${NC}"
echo ""

if [ "$ENABLE_DCGM_METRICS" != "y" ]; then
    echo -e "${YELLOW}DCGM Exporter scraping disabled; skipping.${NC}"
else
    # If a local custom-dcgm-metrics.csv exists, create a ConfigMap from it and
    # point DCGM Exporter at it. Otherwise leave the exporter on the GPU Operator's
    # built-in default metrics rather than pointing it at a non-existent ConfigMap
    # (which would wedge the pod in Init:0/1 on a failed mount).
    if [ -f "custom-dcgm-metrics.csv" ]; then
        echo "Using custom DCGM metrics from: custom-dcgm-metrics.csv"

        # Recreate the ConfigMap. The key must be exactly dcgm-metrics.csv for the
        # exporter to pick it up.
        kubectl delete configmap custom-dcgm-metrics -n gpu-operator --ignore-not-found
        kubectl create configmap custom-dcgm-metrics \
          --from-file=dcgm-metrics.csv=custom-dcgm-metrics.csv \
          -n gpu-operator

        echo "Updating GPU Operator to use custom metrics ConfigMap..."
        helm upgrade gpu-operator nvidia/gpu-operator \
          -n gpu-operator \
          --reset-then-reuse-values \
          --set dcgmExporter.config.name=custom-dcgm-metrics
    else
        echo -e "${YELLOW}No custom-dcgm-metrics.csv found; using GPU Operator default DCGM metrics.${NC}"

        echo "Updating GPU Operator to use default metrics..."
        helm upgrade gpu-operator nvidia/gpu-operator \
          -n gpu-operator \
          --reset-then-reuse-values \
          --set dcgmExporter.config.name=null

        # Now that the release no longer references it, drop any stale ConfigMap.
        kubectl delete configmap custom-dcgm-metrics -n gpu-operator --ignore-not-found
    fi

    # Restart DCGM Exporter
    echo "Restarting DCGM Exporter..."
    kubectl rollout restart daemonset/nvidia-dcgm-exporter -n gpu-operator

    echo "Waiting for DCGM Exporter rollout..."
    kubectl rollout status daemonset/nvidia-dcgm-exporter -n gpu-operator --timeout=120s

    echo -e "${GREEN}✓ Custom DCGM metrics configured${NC}"
fi
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Access Grafana:"
echo "   ${GRAFANA_WORKSPACE_URL}"
echo ""
echo "2. Add AMP data source in Grafana:"
echo "   - AWS icon → Amazon Managed Service for Prometheus"
echo "   - Select region: $AWS_REGION_AMGP"
echo "   - Select your workspace"
echo ""
echo "3. Import NVIDIA DCGM dashboard:"
echo "   - Dashboards → Import → ID: 12239"
echo ""
echo "4. (Optional) Import the SGLang PD / EFA RDMA dashboard:"
echo "   - Dashboards → Import → upload dashboards/sglang-pd.json"
echo ""
echo "5. Add custom metric panels for:"
echo "   - XID Errors: DCGM_EXP_XID_ERRORS_COUNT_total"
echo "   - Power Violations: DCGM_FI_DEV_POWER_VIOLATION_total"
echo "   - Thermal Violations: DCGM_FI_DEV_THERMAL_VIOLATION_total"
echo ""
echo -e "${YELLOW}Verify metrics:${NC}"
echo "  kubectl logs -n adot-col -l app.kubernetes.io/component=opentelemetry-collector --tail=50"
echo ""
echo -e "${YELLOW}Cleanup (if needed):${NC}"
echo "  ./cleanup-obs.sh"
echo ""
