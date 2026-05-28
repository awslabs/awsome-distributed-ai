# AWS PCS Testing Guide

This directory contains test guides and verification procedures for AWS PCS cluster deployments.

## Available Tests

### [Monitoring Stack Test](./monitoring-stack-test.md)

Complete guide for testing PCS cluster with integrated monitoring stack:
- Deploy cluster with Prometheus, Grafana, and DCGM exporter
- Verify Slurm OpenMetrics configuration
- Access Grafana dashboard via Session Manager
- Validate monitoring components on login and compute nodes
- Troubleshooting common issues

**Use cases:**
- Testing monitoring integration before production deployment
- Verifying Ubuntu 24.04 compatibility workarounds
- Validating IAM policies for monitoring
- Testing with Capacity Blocks for ML or On-Demand instances

## Quick Start

### Test Monitoring Stack

```bash
# 1. Deploy cluster with monitoring (CPU-only)
STACK_NAME="pcs-test-monitoring"
AWS_REGION="us-east-1"
AZ_ID="us-east-1a"

aws cloudformation create-stack \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME} \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=${AZ_ID} \
    ParameterKey=DeployMonitoring,ParameterValue=true \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM

# 2. Wait for completion (~30-40 minutes)
aws cloudformation wait stack-create-complete \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME}

# 3. Access login node
INSTANCE_ID=$(aws ec2 describe-instances \
  --region ${AWS_REGION} \
  --filters \
    "Name=tag:aws:pcs:compute-node-group-name,Values=login" \
    "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

aws ssm start-session \
  --region ${AWS_REGION} \
  --target ${INSTANCE_ID}

# 4. Verify monitoring (on login node)
sudo su - ubuntu
sinfo
docker ps
scontrol show config | grep -i metric

# 5. Access Grafana
# Get password
aws ssm get-parameter \
  --region ${AWS_REGION} \
  --name "/pcs/${STACK_NAME}/grafana/admin-password" \
  --with-decryption \
  --query 'Parameter.Value' \
  --output text

# Start port forwarding (in a new terminal)
aws ssm start-session \
  --region ${AWS_REGION} \
  --target ${INSTANCE_ID} \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["443"],"localPortNumber":["8443"]}'

# Open browser: https://localhost:8443/grafana/
# Username: admin, Password: (from above)
```

For detailed steps and troubleshooting, see [monitoring-stack-test.md](./monitoring-stack-test.md).

## Test Environment Recommendations

### Development/Testing
- **Region**: us-east-1 or us-west-2 (lower cost)
- **Instances**: CPU-only (c6i.4xlarge) for basic testing
- **Monitoring**: Enabled to test integration
- **Deployment time**: ~30 minutes

### GPU/P-series Testing
- **Region**: Region with P5/P5en/P6 availability
- **Instances**: Use Capacity Reservation or On-Demand Capacity Reservation
- **Network**: 16 or 32 EFA interfaces for P-series
- **Monitoring**: Enabled to test DCGM exporter
- **Deployment time**: ~40 minutes

## Cleanup

Always clean up test resources after testing:

```bash
# Delete stack
aws cloudformation delete-stack \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME}

# Verify deletion
aws cloudformation wait stack-delete-complete \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME}
```

## Contributing

When adding new tests:
1. Create a new markdown file in this directory
2. Follow the structure of existing test guides
3. Include deployment, verification, troubleshooting, and cleanup steps
4. Update this README with a link to the new test guide
