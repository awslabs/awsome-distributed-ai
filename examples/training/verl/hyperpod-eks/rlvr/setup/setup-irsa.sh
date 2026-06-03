#!/usr/bin/env bash
set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_vars"

# Configuration
EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME}"
AWS_REGION="${AWS_REGION}"
AWS_ACCOUNT="${ACCOUNT}"
S3_BUCKET_ARN="${S3_BUCKET_ARN}"
NAMESPACE="${NAMESPACE:-default}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-ray-s3-sa}"
IAM_ROLE_NAME="ray-s3-access-role"
IAM_POLICY_NAME="ray-s3-access-policy"

# Validate required environment variables
if [ -z "${EKS_CLUSTER_NAME}" ]; then
    echo "ERROR: EKS_CLUSTER_NAME not set. Please set in env_vars file."
    exit 1
fi

if [ -z "${AWS_REGION}" ]; then
    echo "ERROR: AWS_REGION not set. Please set in env_vars file."
    exit 1
fi

if [ -z "${ACCOUNT}" ]; then
    echo "ERROR: ACCOUNT not set. Please set in env_vars file."
    exit 1
fi

if [ -z "${S3_BUCKET_ARN}" ]; then
    echo "ERROR: S3_BUCKET_ARN environment variable is not set"
    echo "Please set it in your env_vars file: export S3_BUCKET_ARN='arn:aws:s3:::your-bucket-name'"
    exit 1
fi

# Validate S3 bucket exists
BUCKET_NAME=$(echo "${S3_BUCKET_ARN}" | sed 's/arn:aws:s3::://')
echo "Validating S3 bucket: ${BUCKET_NAME}..."
if ! aws s3api head-bucket --bucket "${BUCKET_NAME}" 2>/dev/null; then
    echo "ERROR: S3 bucket '${BUCKET_NAME}' does not exist or you don't have access to it"
    exit 1
fi
echo "✓ S3 bucket validated"

echo "=== Setting up IRSA for Ray Pods ==="
echo "EKS Cluster: ${EKS_CLUSTER_NAME}"
echo "Region: ${AWS_REGION}"
echo "Account: ${AWS_ACCOUNT}"
echo "Namespace: ${NAMESPACE}"
echo "Service Account: ${SERVICE_ACCOUNT_NAME}"
echo ""

# Step 1: Check if OIDC provider exists for the cluster
echo "Step 1: Checking OIDC provider..."
OIDC_ID=$(aws eks describe-cluster --name ${EKS_CLUSTER_NAME} --region ${AWS_REGION} --query "cluster.identity.oidc.issuer" --output text | cut -d '/' -f 5)

if [ -z "$OIDC_ID" ]; then
    echo "ERROR: Could not get OIDC ID from cluster"
    exit 1
fi

echo "OIDC ID: ${OIDC_ID}"

# Check if OIDC provider exists in IAM
OIDC_PROVIDER_ARN="arn:aws:iam::${AWS_ACCOUNT}:oidc-provider/oidc.eks.${AWS_REGION}.amazonaws.com/id/${OIDC_ID}"
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn ${OIDC_PROVIDER_ARN} 2>/dev/null; then
    echo "✓ OIDC provider already exists"
else
    echo "Creating OIDC provider..."
    eksctl utils associate-iam-oidc-provider --cluster=${EKS_CLUSTER_NAME} --region=${AWS_REGION} --approve
    echo "✓ OIDC provider created"
fi

echo ""

# Step 2: Create IAM policy for S3 access
echo "Step 2: Creating IAM policy for S3 access..."

# Check if policy already exists
POLICY_ARN="arn:aws:iam::${AWS_ACCOUNT}:policy/${IAM_POLICY_NAME}"
if aws iam get-policy --policy-arn ${POLICY_ARN} 2>/dev/null; then
    echo "✓ IAM policy already exists: ${POLICY_ARN}"
else
    cat > /tmp/ray-s3-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": "${S3_BUCKET_ARN}/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": "${S3_BUCKET_ARN}"
        }
    ]
}
EOF

    aws iam create-policy \
        --policy-name ${IAM_POLICY_NAME} \
        --policy-document file:///tmp/ray-s3-policy.json \
        --description "Restricted S3 access for Ray pods to ${S3_BUCKET_ARN}"
    
    echo "✓ IAM policy created: ${POLICY_ARN}"
    rm /tmp/ray-s3-policy.json
fi

echo ""

# Step 3: Create IAM role with trust policy for the service account
echo "Step 3: Creating IAM role with OIDC trust policy..."

# Check if role already exists
if aws iam get-role --role-name ${IAM_ROLE_NAME} 2>/dev/null; then
    echo "✓ IAM role already exists: ${IAM_ROLE_NAME}"
else
    cat > /tmp/trust-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "${OIDC_PROVIDER_ARN}"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "oidc.eks.${AWS_REGION}.amazonaws.com/id/${OIDC_ID}:sub": "system:serviceaccount:${NAMESPACE}:${SERVICE_ACCOUNT_NAME}",
                    "oidc.eks.${AWS_REGION}.amazonaws.com/id/${OIDC_ID}:aud": "sts.amazonaws.com"
                }
            }
        }
    ]
}
EOF

    aws iam create-role \
        --role-name ${IAM_ROLE_NAME} \
        --assume-role-policy-document file:///tmp/trust-policy.json \
        --description "IAM role for Ray pods to access S3"
    
    echo "✓ IAM role created: ${IAM_ROLE_NAME}"
    rm /tmp/trust-policy.json
fi

echo ""

# Step 4: Attach policy to role
echo "Step 4: Attaching policy to role..."
aws iam attach-role-policy \
    --role-name ${IAM_ROLE_NAME} \
    --policy-arn ${POLICY_ARN}

echo "✓ Policy attached to role"
echo ""

# Step 5: Create Kubernetes service account
echo "Step 5: Creating Kubernetes service account..."

ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT}:role/${IAM_ROLE_NAME}"

cat > /tmp/ray-service-account.yaml <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${SERVICE_ACCOUNT_NAME}
  namespace: ${NAMESPACE}
  annotations:
    eks.amazonaws.com/role-arn: ${ROLE_ARN}
EOF

kubectl apply -f /tmp/ray-service-account.yaml
echo "✓ Kubernetes service account created"
rm /tmp/ray-service-account.yaml

echo ""
echo "=== IRSA Setup Complete ==="
echo ""
echo "Service Account: ${SERVICE_ACCOUNT_NAME}"
echo "IAM Role ARN: ${ROLE_ARN}"
echo ""
echo "Next steps:"
echo "1. Update your RayCluster YAML to use this service account"
echo "2. Add 'serviceAccountName: ${SERVICE_ACCOUNT_NAME}' to the pod spec"
echo ""
echo "Example:"
echo "  spec:"
echo "    serviceAccountName: ${SERVICE_ACCOUNT_NAME}"
echo "    containers:"
echo "    - name: ray-head"
echo "      ..."
echo ""
