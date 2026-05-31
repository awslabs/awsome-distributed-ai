# PCS Monitoring Stack Test Guide

## Overview

This guide provides step-by-step instructions to deploy and verify AWS PCS cluster with integrated monitoring stack (Prometheus, Grafana, DCGM exporter).

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Session Manager Plugin** for accessing Grafana dashboard
   ```bash
   # Install on macOS
   brew install --cask session-manager-plugin
   
   # Install on Linux
   curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb" -o "session-manager-plugin.deb"
   sudo dpkg -i session-manager-plugin.deb
   ```

3. **IAM Permissions**: CloudFormation, EC2, PCS, SSM, IAM

## Test Deployment

### Step 1: Deploy PCS Cluster with Monitoring

#### Option A: With Capacity Reservation (P5/P5en instances)

```bash
# Set variables
STACK_NAME="pcs-test-monitoring"
AWS_REGION="us-east-2"
CAPACITY_RESERVATION_ID="cr-0123456789abcdef0"

# Get AZ from Capacity Reservation
AZ_ID=$(aws ec2 describe-capacity-reservations \
  --region ${AWS_REGION} \
  --capacity-reservation-ids ${CAPACITY_RESERVATION_ID} \
  --query 'CapacityReservations[0].AvailabilityZone' \
  --output text)

# Get instance type and count
CR_INFO=$(aws ec2 describe-capacity-reservations \
  --region ${AWS_REGION} \
  --capacity-reservation-ids ${CAPACITY_RESERVATION_ID} \
  --query 'CapacityReservations[0].[InstanceType,TotalInstanceCount]' \
  --output text)

INSTANCE_TYPE=$(echo ${CR_INFO} | cut -f1)
INSTANCE_COUNT=$(echo ${CR_INFO} | cut -f2)

echo "Deploying to AZ: ${AZ_ID}"
echo "Instance Type: ${INSTANCE_TYPE}"
echo "Instance Count: ${INSTANCE_COUNT}"

# Deploy stack
aws cloudformation create-stack \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME} \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=${AZ_ID} \
    ParameterKey=DeployMonitoring,ParameterValue=true \
    ParameterKey=DeployPseriesCNG,ParameterValue=true \
    ParameterKey=PseriesCngName,ParameterValue=gpu-p5en \
    ParameterKey=PseriesQueueName,ParameterValue=gpu-p5en \
    ParameterKey=PseriesInstanceType,ParameterValue=${INSTANCE_TYPE} \
    ParameterKey=NetworkInterfaceCount,ParameterValue=16 \
    ParameterKey=PseriesMinCount,ParameterValue=${INSTANCE_COUNT} \
    ParameterKey=PseriesMaxCount,ParameterValue=${INSTANCE_COUNT} \
    ParameterKey=CapacityReservationId,ParameterValue=${CAPACITY_RESERVATION_ID} \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

#### Option B: CPU-only cluster (no GPU)

```bash
# Set variables
STACK_NAME="pcs-test-monitoring"
AWS_REGION="us-east-1"
AZ_ID="us-east-1a"

# Deploy stack
aws cloudformation create-stack \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME} \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=${AZ_ID} \
    ParameterKey=DeployMonitoring,ParameterValue=true \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

### Step 2: Monitor Deployment Progress

```bash
# Watch stack events
aws cloudformation describe-stack-events \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME} \
  --max-items 20

# Wait for completion (takes ~30-40 minutes)
aws cloudformation wait stack-create-complete \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME}

# Check stack status
aws cloudformation describe-stacks \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME} \
  --query 'Stacks[0].[StackStatus,StackStatusReason]' \
  --output table
```

**Expected timeline:**
- VPC/FSx creation: ~5 minutes
- AMI building: ~15-20 minutes
- PCS cluster: ~5 minutes
- Login node + monitoring: ~10 minutes
- Compute nodes: ~5 minutes

### Step 3: Get Stack Outputs

```bash
# Get all outputs
aws cloudformation describe-stacks \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME} \
  --query 'Stacks[0].Outputs' \
  --output table

# Get PCS Console URL
aws cloudformation describe-stacks \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME} \
  --query 'Stacks[0].Outputs[?OutputKey==`PcsConsoleUrl`].OutputValue' \
  --output text
```

## Verification Steps

### 1. Access Login Node

```bash
# Get login node instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --region ${AWS_REGION} \
  --filters \
    "Name=tag:aws:pcs:compute-node-group-name,Values=login" \
    "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

echo "Login node instance ID: ${INSTANCE_ID}"

# Connect via Session Manager
aws ssm start-session \
  --region ${AWS_REGION} \
  --target ${INSTANCE_ID}

# Once connected, switch to ubuntu user
sudo su - ubuntu
```

### 2. Verify Slurm Cluster

```bash
# Check cluster status
sinfo

# Expected output:
# PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
# cpu1*        up   infinite      4  idle~ cpu1-dy-[1-4]
# gpu-p5en     up   infinite      2  idle  gpu-p5en-st-[1-2]

# Check node details
scontrol show nodes

# Check Slurm version
scontrol --version
```

### 3. Verify Slurm OpenMetrics Configuration

```bash
# Check slurmctld configuration
scontrol show config | grep -i metric

# Expected output:
# MetricsType = metrics/openmetrics
# CommunicationParameters = enable_http

# Test OpenMetrics endpoint (from login node)
curl -s http://localhost:6817/metrics | head -20

# Expected output: Prometheus-format metrics from slurmctld
```

### 4. Verify Monitoring Containers on Login Node

```bash
# On login node, check Docker containers
docker ps

# Expected containers:
# - prometheus
# - grafana
# - nginx
# - cloudwatch-exporter
# - node-exporter

# Check specific containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check container logs
docker logs prometheus | tail -20
docker logs grafana | tail -20
```

### 5. Verify Monitoring Installation Log

```bash
# Check installation log on login node
sudo cat /var/log/monitoring-install.log

# Verify the installer detected the Ubuntu user and installed to /home/ubuntu
# (v2.6.3+ auto-detects 'ubuntu'; no ec2-user shim is created any more)
ls -la /home/ubuntu/aws-parallelcluster-monitoring
grep -E 'PLATFORM_USER|MONITORING_HOME' /var/log/monitoring-install.log

# Expected: PLATFORM_USER=ubuntu, MONITORING_HOME=/home/ubuntu/aws-parallelcluster-monitoring
```

### 6. Verify Compute Node Exporters

#### On GPU nodes (if deployed)

```bash
# From login node, check DCGM exporter on GPU nodes
srun -N 1 -p gpu-p5en docker ps | grep dcgm-exporter

# SSH to compute node and verify
ssh gpu-p5en-st-1
docker logs dcgm-exporter | tail -20

# Test DCGM metrics endpoint
curl -s http://localhost:9400/metrics | grep dcgm_gpu
```

#### On CPU nodes

```bash
# From login node, check node-exporter on CPU nodes
srun -N 1 -p cpu1 docker ps | grep node-exporter

# Verify node-exporter metrics
ssh cpu1-dy-1  # if node is running
curl -s http://localhost:9100/metrics | grep node_cpu
```

### 7. Verify Prometheus Targets

```bash
# On login node, check Prometheus targets via API
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, instance: .labels.instance, health: .health}'

# Expected targets:
# - Login node exporters (node-exporter, cloudwatch-exporter)
# - Compute node exporters (node-exporter on CPU, dcgm-exporter on GPU)
# - slurmctld (Slurm OpenMetrics endpoint)
```

### 8. Verify IAM Policy for Monitoring

```bash
# Get IAM role name
ROLE_NAME=$(aws cloudformation describe-stacks \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME} \
  --query 'Stacks[0].Outputs[?OutputKey==`RoleName`].OutputValue' \
  --output text)

echo "IAM Role: ${ROLE_NAME}"

# List attached policies
aws iam list-attached-role-policies \
  --role-name ${ROLE_NAME} \
  --query 'AttachedPolicies[*].[PolicyName,PolicyArn]' \
  --output table

# Expected: Monitoring policy attached
# - <STACK_NAME>-monitoring-policy

# Get policy document
POLICY_ARN=$(aws iam list-attached-role-policies \
  --role-name ${ROLE_NAME} \
  --query 'AttachedPolicies[?contains(PolicyName, `monitoring`)].PolicyArn' \
  --output text)

aws iam get-policy-version \
  --policy-arn ${POLICY_ARN} \
  --version-id $(aws iam get-policy --policy-arn ${POLICY_ARN} --query 'Policy.DefaultVersionId' --output text) \
  --query 'PolicyVersion.Document' \
  --output json
```

## Access Grafana Dashboard

### 1. Get Grafana Admin Password

```bash
# Retrieve password from SSM Parameter Store
aws ssm get-parameter \
  --region ${AWS_REGION} \
  --name "/pcs/${STACK_NAME}/grafana/admin-password" \
  --with-decryption \
  --query 'Parameter.Value' \
  --output text
```

### 2. Start Port Forwarding

```bash
# Get login node instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --region ${AWS_REGION} \
  --filters \
    "Name=tag:aws:pcs:compute-node-group-name,Values=login" \
    "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

# Start port forwarding (local 8443 -> remote 443)
aws ssm start-session \
  --region ${AWS_REGION} \
  --target ${INSTANCE_ID} \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["443"],"localPortNumber":["8443"]}'
```

### 3. Access Grafana

Open browser: **https://localhost:8443/grafana/**

- Username: `admin`
- Password: (from Step 1)

### 4. Verify Dashboards

Expected pre-configured dashboards:
- **Cluster Overview**: CPU, memory, node status
- **GPU Metrics**: DCGM metrics (utilization, temperature, power)
- **Slurm Jobs**: Queue status, running/pending jobs
- **Cost Analysis**: EC2/FSx pricing

### 5. Check Prometheus Data Source

In Grafana:
1. Go to **Configuration** > **Data Sources**
2. Verify Prometheus is configured (URL: http://prometheus:9090)
3. Click **Test** button - should show "Data source is working"

## Troubleshooting

### Monitoring Installation Failed

```bash
# Check installation log on login node
sudo cat /var/log/monitoring-install.log

# Confirm the installer detected the Ubuntu user (v2.6.3+)
grep -E 'PLATFORM_USER|MONITORING_HOME' /var/log/monitoring-install.log
ls -la /home/ubuntu/aws-parallelcluster-monitoring

# Manually re-run installation (pin to the same release tag as the deployment)
curl -fsSL https://raw.githubusercontent.com/aws-samples/aws-parallelcluster-monitoring/v2.6.3/post-install.sh -o /tmp/post-install.sh
sudo bash /tmp/post-install.sh v2.6.3
```

### Containers Not Running

```bash
# On login node, check Docker service
sudo systemctl status docker

# Restart Docker
sudo systemctl restart docker

# Check container logs
docker logs prometheus
docker logs grafana
docker logs nginx

# Restart containers
cd /home/ubuntu/aws-parallelcluster-monitoring
docker-compose restart
```

### Grafana Not Accessible

```bash
# Check nginx container
docker logs nginx

# Check if port 443 is listening
sudo netstat -tlnp | grep 443

# Check certificate
docker exec nginx ls -la /etc/nginx/ssl/

# Manually access Grafana (from login node)
curl -k https://localhost/grafana/api/health
```

### DCGM Exporter Not Running on GPU Nodes

```bash
# SSH to GPU compute node
ssh gpu-p5en-st-1

# Check Docker containers
docker ps -a

# Check DCGM exporter logs
docker logs dcgm-exporter

# Verify GPU is detected
nvidia-smi

# Restart DCGM exporter
docker restart dcgm-exporter
```

### Slurm OpenMetrics Not Working

```bash
# On login node, check slurmctld config
scontrol show config | grep -i metric

# If MetricsType not set, update cluster configuration
aws pcs update-cluster \
  --region ${AWS_REGION} \
  --cluster-identifier $(aws cloudformation describe-stacks \
    --region ${AWS_REGION} \
    --stack-name ${STACK_NAME} \
    --query 'Stacks[0].Outputs[?OutputKey==`ClusterId`].OutputValue' \
    --output text) \
  --slurm-configuration 'slurmCustomSettings=[
    {parameterName=MetricsType,parameterValue=metrics/openmetrics},
    {parameterName=CommunicationParameters,parameterValue=enable_http}
  ]'

# Restart slurmctld
sudo systemctl restart slurmctld
```

### Prometheus Targets Down

```bash
# On login node, check Prometheus config
docker exec prometheus cat /etc/prometheus/prometheus.yml

# Check Prometheus logs
docker logs prometheus | grep -i error

# Verify network connectivity to compute nodes
ping -c 3 gpu-p5en-st-1
curl -s http://gpu-p5en-st-1:9400/metrics | head  # DCGM exporter
curl -s http://cpu1-dy-1:9100/metrics | head      # node-exporter (if node running)
```

### Stack Creation Failed

```bash
# Check stack events for errors
aws cloudformation describe-stack-events \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME} \
  --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`].[LogicalResourceId,ResourceStatusReason]' \
  --output table

# Check nested stack status
aws cloudformation list-stacks \
  --region ${AWS_REGION} \
  --query "StackSummaries[?contains(StackName, '${STACK_NAME}')].[StackName,StackStatus]" \
  --output table

# Common issues:
# - RoleName output not found: Fixed in cluster.yaml
# - Capacity reservation full: Check reservation availability
# - AMI build timeout: Check EC2 Image Builder pipeline
# - Monitoring IAM policy failed: Check IAM permissions
```

## Test Checklist

- [ ] Stack deploys successfully (all nested stacks CREATE_COMPLETE)
- [ ] Login node accessible via Session Manager
- [ ] Slurm shows expected partitions (cpu1, gpu-p5en if deployed)
- [ ] Slurm OpenMetrics enabled (MetricsType=metrics/openmetrics)
- [ ] slurmctld metrics endpoint accessible (http://localhost:6817/metrics)
- [ ] Monitoring containers running on login node (prometheus, grafana, nginx)
- [ ] DCGM exporter running on GPU nodes (if deployed)
- [ ] Node exporter running on CPU nodes
- [ ] Grafana accessible at https://localhost:8443/grafana/
- [ ] Grafana password retrievable from SSM Parameter Store
- [ ] Prometheus targets are UP (check in Grafana or API)
- [ ] Dashboards display metrics
- [ ] Installer detected Ubuntu user (MONITORING_HOME=/home/ubuntu/aws-parallelcluster-monitoring)
- [ ] IAM monitoring policy attached to cluster role
- [ ] Stack deletes cleanly

## Cleanup

### Delete the Stack

```bash
# Delete stack (includes all nested stacks)
aws cloudformation delete-stack \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME}

# Wait for deletion to complete
aws cloudformation wait stack-delete-complete \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME}

# Verify deletion
aws cloudformation describe-stacks \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME} 2>&1 | grep -q "does not exist" && echo "Stack deleted successfully"
```

**Important notes:**
- Backup any data in FSx filesystems before deletion
- Capacity Reservation will NOT be deleted (must be released manually)
- Stack deletion takes ~10-15 minutes

### Manual Cleanup (if needed)

```bash
# If stack deletion fails, check for stuck resources
aws cloudformation describe-stack-resources \
  --region ${AWS_REGION} \
  --stack-name ${STACK_NAME} \
  --query 'StackResources[?ResourceStatus!=`DELETE_COMPLETE`].[LogicalResourceId,ResourceType,ResourceStatus]' \
  --output table

# List all related stacks
aws cloudformation list-stacks \
  --region ${AWS_REGION} \
  --query "StackSummaries[?contains(StackName, '${STACK_NAME}')].[StackName,StackStatus,DeletionTime]" \
  --output table

# Delete nested stacks individually if needed
aws cloudformation delete-stack \
  --region ${AWS_REGION} \
  --stack-name <nested-stack-name>
```

## Expected Results

✅ **Success criteria:**
1. All nested stacks CREATE_COMPLETE
2. Slurm OpenMetrics configuration enabled
3. Monitoring containers running on login node
4. DCGM exporter running on GPU nodes (if deployed)
5. Grafana accessible and showing metrics
6. No errors in /var/log/monitoring-install.log
7. IAM monitoring policy attached

⚠️ **Known issues:**
- First Grafana login may take 10-20 seconds (containers starting)
- Self-signed certificate warning (expected)
- GPU metrics may take 1-2 minutes to appear after node boot
- Dynamic CPU nodes won't show exporters until nodes are allocated

## References

- [AWS PCS Documentation](https://docs.aws.amazon.com/pcs/)
- [AWS ParallelCluster Monitoring](https://github.com/aws-samples/aws-parallelcluster-monitoring)
- [Slurm OpenMetrics](https://slurm.schedmd.com/rest.html#openmetrics)
- [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)
- [Session Manager Plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
