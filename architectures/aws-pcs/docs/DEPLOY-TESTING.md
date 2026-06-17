# Development & Testing Deploy Procedures

## Prerequisites

- AWS CLI configured (`--profile claude` in this project)
- Test S3 bucket: `midaisuk-llm-dev` (us-east-1)
- Production S3 bucket: `awsome-distributed-ai` (us-east-1)
- Test region: `us-east-2`

---

## Upload templates + scripts to test bucket

```bash
cd architectures/aws-pcs
aws s3 sync assets/ s3://midaisuk-llm-dev/templates/ \
  --exclude "*" --include "*.yaml" --include "*.sh" \
  --profile claude
```

This uploads both CloudFormation templates and boot scripts in one command
(scripts live under `assets/scripts/`, synced to `s3://bucket/templates/scripts/`).

---

## Deploy a test cluster

### Minimal (single ubuntu user, default)

```bash
aws cloudformation create-stack \
  --stack-name pcs-test \
  --template-url https://midaisuk-llm-dev.s3.us-east-1.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-2b \
    ParameterKey=S3BucketName,ParameterValue=midaisuk-llm-dev \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region us-east-2 --profile claude
```

### With multi-user (OpenLDAP on login node)

```bash
aws cloudformation create-stack \
  --stack-name pcs-multiuser \
  --template-url https://midaisuk-llm-dev.s3.us-east-1.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-2b \
    ParameterKey=S3BucketName,ParameterValue=midaisuk-llm-dev \
    ParameterKey=DirectoryService,ParameterValue=OpenLDAP-LoginNode \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region us-east-2 --profile claude
```

### With EFA on HPC instances

```bash
aws cloudformation create-stack \
  --stack-name pcs-hpc \
  --template-url https://midaisuk-llm-dev.s3.us-east-1.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-2b \
    ParameterKey=S3BucketName,ParameterValue=midaisuk-llm-dev \
    ParameterKey=OnDemandInstanceType,ParameterValue=hpc8a.96xlarge \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region us-east-2 --profile claude
```

---

## Monitor deploy progress

```bash
# Quick status
aws cloudformation describe-stacks --stack-name <stack-name> --region us-east-2 --profile claude \
  --query 'Stacks[0].StackStatus' --output text

# Nested stack progress
aws cloudformation describe-stack-events --stack-name <stack-name> --region us-east-2 --profile claude \
  --query 'StackEvents[?ResourceStatus!=`CREATE_IN_PROGRESS`].[Timestamp,LogicalResourceId,ResourceStatus]' \
  --output text | head -20
```

Typical timeline (~25-30 min):
1. PrerequisitesStack (VPC + FSx) — ~10 min
2. ClusterStack (PCS cluster) — ~5 min
3. LoginNodeGroupStack + OnDemandCNGStack (parallel) — ~5 min
4. Node boot + cloud-init (post-install, monitoring, directory) — ~3-8 min

---

## Connect to the login node

```bash
# Find login node
CLUSTER_ID=$(aws cloudformation describe-stacks --stack-name <stack-name> --region us-east-2 --profile claude \
  --query 'Stacks[0].Outputs[?OutputKey==`ClusterId`].OutputValue' --output text)
LOGIN_ID=$(aws ec2 describe-instances --region us-east-2 --profile claude \
  --filters "Name=tag:pcs-cluster-id,Values=$CLUSTER_ID" \
            "Name=tag:monitoring-role,Values=login" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)

# SSH config (one-time)
cat >> ~/.ssh/config << EOF
Host pcs-login
  HostName $LOGIN_ID
  User ubuntu
  ProxyCommand sh -c "aws ssm start-session --profile claude --region us-east-2 --target %h --document-name AWS-StartSSHSession --parameters 'portNumber=%p'"
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
EOF

# Inject SSH key (one-time)
aws ssm send-command --instance-ids $LOGIN_ID --region us-east-2 --profile claude \
  --document-name "AWS-RunShellScript" \
  --parameters "commands=[\"mkdir -p /home/ubuntu/.ssh; echo '$(cat ~/.ssh/id_rsa.pub)' >> /home/ubuntu/.ssh/authorized_keys; chown -R ubuntu:ubuntu /home/ubuntu/.ssh; chmod 700 /home/ubuntu/.ssh; chmod 600 /home/ubuntu/.ssh/authorized_keys\"]"

# Connect
ssh pcs-login
```

---

## Verification checks

### Basic cluster health
```bash
ssh pcs-login 'export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH; sinfo -N; squeue'
```

### Monitoring
```bash
ssh pcs-login 'sudo docker ps --format "table {{.Names}}\t{{.Status}}"'
```

### Multi-user (when DirectoryService=OpenLDAP-LoginNode)
```bash
# On login node
ssh pcs-login 'systemctl status slapd | head -3'
ssh pcs-login 'getent passwd testuser1'  # after creating a user
ssh pcs-login 'cat /var/log/directory-setup.log | tail -10'

# On compute node (via srun)
ssh pcs-login 'export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH; srun -N 1 -n 1 -p cpu1 bash -c "getent passwd testuser1"'
```

### LDAP admin password
```bash
# From SSM (preferred)
aws ssm get-parameter --name "/pcs/${CLUSTER_ID}/ldap/admin-password" \
  --with-decryption --query 'Parameter.Value' --output text --region us-east-2 --profile claude

# Fallback (if SSM put failed at boot)
ssh pcs-login 'sudo cat /home/ldap-db/.admin-password'
```

---

## Cleanup

```bash
aws cloudformation delete-stack --stack-name <stack-name> --region us-east-2 --profile claude
```

Nested stacks are deleted automatically. If DELETE_FAILED on CNG stacks
(PCS timing dependency), delete PCS CNGs first:
```bash
CLUSTER_ID=<id>
for cng in $(aws pcs list-compute-node-groups --cluster-identifier $CLUSTER_ID --region us-east-2 --profile claude --query 'computeNodeGroups[].id' --output text); do
  aws pcs delete-compute-node-group --cluster-identifier $CLUSTER_ID --compute-node-group-identifier $cng --region us-east-2 --profile claude
done
sleep 60
aws cloudformation delete-stack --stack-name <stack-name> --region us-east-2 --profile claude
```

---

## Production deploy (post-merge)

After PR merge, maintainer syncs to prod bucket:
```bash
aws s3 sync assets/ s3://awsome-distributed-ai/templates/
```

Users deploy from prod bucket (no S3BucketName override needed — it's the default):
```bash
aws cloudformation create-stack \
  --stack-name pcs-ml-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-1a \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```
