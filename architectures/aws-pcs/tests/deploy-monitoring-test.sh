#!/bin/bash
#
# Deploy minimal PCS cluster for monitoring stack testing
#
set -e

STACK_NAME="pcs-monitoring-test"
REGION="us-east-2"
PROFILE="claude"

echo "=== Monitoring Stack Test Deployment ==="
echo "Stack: $STACK_NAME"
echo "Region: $REGION"
echo ""

# Load prerequisites environment
if [ ! -f /tmp/pcs-env.sh ]; then
    echo "Error: /tmp/pcs-env.sh not found"
    echo "Run the following to create it:"
    echo ""
    echo "  OUTPUTS=\$(aws cloudformation describe-stacks --stack-name pcs-shared-prerequisites --region $REGION --profile $PROFILE --query 'Stacks[0].Outputs' --output json)"
    echo "  echo \"export VPC_ID=\$(echo \$OUTPUTS | jq -r '.[] | select(.OutputKey==\"VPC\") | .OutputValue')\" > /tmp/pcs-env.sh"
    echo "  echo \"export PUBLIC_SUBNET=\$(echo \$OUTPUTS | jq -r '.[] | select(.OutputKey==\"PublicSubnet\") | .OutputValue')\" >> /tmp/pcs-env.sh"
    echo "  echo \"export PRIVATE_SUBNET=\$(echo \$OUTPUTS | jq -r '.[] | select(.OutputKey==\"PrimaryPrivateSubnet\") | .OutputValue')\" >> /tmp/pcs-env.sh"
    echo "  echo \"export SECURITY_GROUP=\$(echo \$OUTPUTS | jq -r '.[] | select(.OutputKey==\"SecurityGroup\") | .OutputValue')\" >> /tmp/pcs-env.sh"
    echo "  echo \"export FSX_LUSTRE_ID=\$(echo \$OUTPUTS | jq -r '.[] | select(.OutputKey==\"FSxLustreFilesystemId\") | .OutputValue')\" >> /tmp/pcs-env.sh"
    echo "  echo \"export FSX_LUSTRE_MOUNT=\$(echo \$OUTPUTS | jq -r '.[] | select(.OutputKey==\"FSxLustreFilesystemMountname\") | .OutputValue')\" >> /tmp/pcs-env.sh"
    echo "  echo \"export FSX_OPENZFS_ID=\$(echo \$OUTPUTS | jq -r '.[] | select(.OutputKey==\"FSxOpenZFSFilesystemId\") | .OutputValue')\" >> /tmp/pcs-env.sh"
    exit 1
fi

source /tmp/pcs-env.sh

echo "Prerequisites loaded:"
echo "  VPC: $VPC_ID"
echo "  Public Subnet: $PUBLIC_SUBNET"
echo "  Security Group: $SECURITY_GROUP"
echo ""

# Deploy cluster
echo "Deploying monitoring test cluster..."
aws cloudformation create-stack \
  --stack-name $STACK_NAME \
  --region $REGION \
  --profile $PROFILE \
  --template-body file://$(dirname $0)/../assets/pcluster-monitoring-test-cluster.yaml \
  --parameters \
    ParameterKey=VpcId,ParameterValue=$VPC_ID \
    ParameterKey=PublicSubnetId,ParameterValue=$PUBLIC_SUBNET \
    ParameterKey=PrivateSubnetId,ParameterValue=$PRIVATE_SUBNET \
    ParameterKey=DefaultSecurityGroupId,ParameterValue=$SECURITY_GROUP \
    ParameterKey=FSxLustreFilesystemId,ParameterValue=$FSX_LUSTRE_ID \
    ParameterKey=FSxLustreFilesystemMountName,ParameterValue=$FSX_LUSTRE_MOUNT \
    ParameterKey=FSxOpenZFSFilesystemId,ParameterValue=$FSX_OPENZFS_ID \
    ParameterKey=MonitoringBranch,ParameterValue=fix/install-script-local-variable-declaration \
    ParameterKey=MonitoringRepo,ParameterValue=DaisukeMiyamoto/aws-parallelcluster-monitoring \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM

echo ""
echo "Waiting for stack creation (~5-7 minutes)..."
aws cloudformation wait stack-create-complete \
  --stack-name $STACK_NAME \
  --region $REGION \
  --profile $PROFILE

echo "✅ Stack created successfully"
echo ""

# Get cluster ID and login instance
CLUSTER_ID=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME \
  --region $REGION \
  --profile $PROFILE \
  --query 'Stacks[0].Outputs[?OutputKey==`ClusterId`].OutputValue' \
  --output text)

echo "Cluster ID: $CLUSTER_ID"
echo ""
echo "Waiting for login instance to start (~3 minutes)..."
sleep 180

LOGIN_INSTANCE=$(aws ec2 describe-instances \
  --region $REGION \
  --profile $PROFILE \
  --filters \
    "Name=tag:pcs-cluster-id,Values=$CLUSTER_ID" \
    "Name=tag:Name,Values=pcs-monitoring-test-login" \
    "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

echo "Login Instance: $LOGIN_INSTANCE"
echo ""
echo "=== Verification ==="
echo ""
echo "1. Check installation log:"
echo "   aws ssm send-command --instance-ids $LOGIN_INSTANCE --region $REGION --profile $PROFILE --document-name 'AWS-RunShellScript' --parameters 'commands=[\"cat /var/log/monitoring-install-test.log\"]'"
echo ""
echo "2. Verify Docker containers:"
echo "   aws ssm send-command --instance-ids $LOGIN_INSTANCE --region $REGION --profile $PROFILE --document-name 'AWS-RunShellScript' --parameters 'commands=[\"docker ps\"]'"
echo ""
echo "3. Check monitoring files:"
echo "   aws ssm send-command --instance-ids $LOGIN_INSTANCE --region $REGION --profile $PROFILE --document-name 'AWS-RunShellScript' --parameters 'commands=[\"ls -la /home/ubuntu/aws-parallelcluster-monitoring/\"]'"
echo ""
echo "4. Get Grafana password:"
echo "   aws ssm get-parameter --name '/pcs/$CLUSTER_ID/grafana/admin-password' --region $REGION --profile $PROFILE --with-decryption --query 'Parameter.Value' --output text"
echo ""
echo "=== Cleanup ==="
echo "aws cloudformation delete-stack --stack-name $STACK_NAME --region $REGION --profile $PROFILE"
