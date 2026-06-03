#!/bin/bash

# Script to extract Terraform outputs and create environment variables

set -e

echo "Extracting Terraform outputs..."

# Check if terraform state exists
if [ ! -f "terraform.tfstate" ]; then
    echo "Error: terraform.tfstate not found. Please run 'terraform apply' first."
    exit 1
fi

# Extract outputs
VPC_ID=$(terraform output -raw vpc_id 2>/dev/null || echo "")
PRIVATE_SUBNET_ID=$(terraform output -raw private_subnet_id 2>/dev/null || echo "")
SECURITY_GROUP_ID=$(terraform output -raw security_group_id 2>/dev/null || echo "")
S3_BUCKET_NAME=$(terraform output -raw s3_bucket_name 2>/dev/null || echo "")
SAGEMAKER_ROLE_ARN=$(terraform output -raw sagemaker_iam_role_arn 2>/dev/null || echo "")
FSX_DNS_NAME=$(terraform output -raw fsx_lustre_dns_name 2>/dev/null || echo "")
FSX_MOUNT_NAME=$(terraform output -raw fsx_lustre_mount_name 2>/dev/null || echo "")
HYPERPOD_CLUSTER_NAME=$(terraform output -raw hyperpod_cluster_name 2>/dev/null || echo "")
HYPERPOD_CLUSTER_ARN=$(terraform output -raw hyperpod_cluster_arn 2>/dev/null || echo "")

# Create environment variables file
cat > env_vars.sh << EOF
#!/bin/bash
# Environment variables from Terraform deployment

export VPC_ID="$VPC_ID"
export PRIVATE_SUBNET_ID="$PRIVATE_SUBNET_ID"
export SECURITY_GROUP_ID="$SECURITY_GROUP_ID"
export S3_BUCKET_NAME="$S3_BUCKET_NAME"
export SAGEMAKER_ROLE_ARN="$SAGEMAKER_ROLE_ARN"
export FSX_DNS_NAME="$FSX_DNS_NAME"
export FSX_MOUNT_NAME="$FSX_MOUNT_NAME"
export HYPERPOD_CLUSTER_NAME="$HYPERPOD_CLUSTER_NAME"
export HYPERPOD_CLUSTER_ARN="$HYPERPOD_CLUSTER_ARN"

echo "Environment variables loaded:"
echo "  VPC_ID: \$VPC_ID"
echo "  PRIVATE_SUBNET_ID: \$PRIVATE_SUBNET_ID"
echo "  SECURITY_GROUP_ID: \$SECURITY_GROUP_ID"
echo "  S3_BUCKET_NAME: \$S3_BUCKET_NAME"
echo "  HYPERPOD_CLUSTER_NAME: \$HYPERPOD_CLUSTER_NAME"
EOF

chmod +x env_vars.sh

echo "Environment variables file created: env_vars.sh"
echo "To load variables, run: source env_vars.sh"
echo ""
echo "Current values:"
echo "  VPC ID: $VPC_ID"
echo "  Private Subnet ID: $PRIVATE_SUBNET_ID"
echo "  Security Group ID: $SECURITY_GROUP_ID"
echo "  S3 Bucket: $S3_BUCKET_NAME"
echo "  HyperPod Cluster: $HYPERPOD_CLUSTER_NAME"