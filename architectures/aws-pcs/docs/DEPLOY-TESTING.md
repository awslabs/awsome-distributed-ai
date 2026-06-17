# Deploying updated templates before they are published

The one-click Quick Start in the [README](../README.md#3-quick-start) deploys from the
**public production bucket** (`awsome-distributed-ai`), which only has the templates that
have already been merged and published. This guide is for the other case: **you have
template/script changes that are not yet in the public bucket** (e.g. a fork, a feature
branch, or a PR under review) and you want to deploy and test them.

The approach is the same in every case: **host the templates + boot scripts in an S3
bucket you control, then point the deploy at that bucket** via the `S3BucketName` /
`S3KeyPrefix` parameters. The nested stacks and the first-boot scripts are all fetched
from `s3://<S3BucketName>/<S3KeyPrefix>...`, so overriding those two parameters redirects
the entire deploy to your copy.

## 1. Prerequisites

- AWS CLI configured with credentials for your account.
- An S3 bucket you own, in any Region (the bucket is reached by its global name, so it
  does not need to be in the deploy Region). It can be **private** — the templates are
  fetched by your CLI/CloudFormation, and the boot scripts are fetched by the node
  instance role, so no public access is required.
- A local checkout of the branch/fork with the changes you want to test.

This guide uses these placeholders — substitute your own:

```bash
BUCKET=my-pcs-templates       # an S3 bucket you control
PREFIX=templates/             # key prefix (keep the trailing slash)
REGION=us-east-1              # the Region to deploy the cluster into
AZ=us-east-1a                 # an Availability Zone in $REGION
```

## 2. Upload the templates + scripts to your bucket

Run from the repo's `architectures/aws-pcs` directory (so `assets/` is the source):

```bash
aws s3 sync assets/ "s3://${BUCKET}/${PREFIX}" \
  --exclude "*" --include "*.yaml" --include "*.sh"
```

This uploads both the CloudFormation templates (`*.yaml`) and the boot scripts (`*.sh`)
in one command. The scripts live under `assets/scripts/`, so they land at
`s3://${BUCKET}/${PREFIX}scripts/` — which is exactly where the default
`PostInstallScriptUrl` looks (`s3://<S3BucketName>/<S3KeyPrefix>scripts/install-enroot-pyxis.sh`).
Re-run this sync after every change you want to test.

## 3. Deploy pointing at your bucket

Pass `S3BucketName` (and `S3KeyPrefix` if you changed it from the `templates/` default).
That single override is what makes the nested stacks **and** the first-boot Enroot/Pyxis
installer come from your copy instead of the public bucket:

```bash
aws cloudformation create-stack \
  --stack-name pcs-test \
  --template-url "https://${BUCKET}.s3.amazonaws.com/${PREFIX}pcs-ml-cluster-deploy-all.yaml" \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=${AZ} \
    ParameterKey=S3BucketName,ParameterValue=${BUCKET} \
    ParameterKey=S3KeyPrefix,ParameterValue=${PREFIX} \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region ${REGION}
```

> **Why this matters.** If you deploy the updated top-level template but leave
> `S3BucketName` at its default, the nested stacks and boot scripts are still pulled from
> the **public** bucket — so you'd be testing your top-level change against the old
> published nested templates/scripts. Always override `S3BucketName` to your bucket when
> testing unpublished changes.

Add any parameters you're testing on top — for example multi-user:

```bash
    ParameterKey=DirectoryService,ParameterValue=OpenLDAP-LoginNode \
```

or an EFA-capable CPU queue:

```bash
    ParameterKey=OnDemandInstanceType,ParameterValue=hpc8a.96xlarge \
    ParameterKey=OnDemandEfaInterfaceCount,ParameterValue=2 \
```

See [PARAMETERS.md](./PARAMETERS.md) for the full list.

## 4. Monitor progress

```bash
aws cloudformation describe-stacks --stack-name pcs-test --region ${REGION} \
  --query 'Stacks[0].StackStatus' --output text

aws cloudformation describe-stack-events --stack-name pcs-test --region ${REGION} \
  --query 'StackEvents[?ResourceStatus!=`CREATE_IN_PROGRESS`].[Timestamp,LogicalResourceId,ResourceStatus]' \
  --output text | head -20
```

Typical timeline (~25-30 min): Prerequisites (VPC + FSx) ~10 min → Cluster ~5 min →
Login + compute node groups (parallel) ~5 min → node boot + cloud-init (post-install,
monitoring, directory) ~3-8 min.

## 5. Connect and verify

Connect to the login node over SSM (no public IP or SSH key needed):

```bash
CLUSTER_ID=$(aws cloudformation describe-stacks --stack-name pcs-test --region ${REGION} \
  --query 'Stacks[0].Outputs[?OutputKey==`ClusterId`].OutputValue' --output text)
LOGIN_ID=$(aws ec2 describe-instances --region ${REGION} \
  --filters "Name=tag:pcs-cluster-id,Values=$CLUSTER_ID" \
            "Name=tag:monitoring-role,Values=login" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
aws ssm start-session --target $LOGIN_ID --region ${REGION}
```

Then `sudo su - ubuntu` and run the checks relevant to your change. A few quick ones:

```bash
# Slurm sees the queues / nodes (adjust the version if you set SlurmVersion=25.05)
export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH; sinfo -N; squeue

# Container runtime came from your bucket (the log shows the s3:// it fetched)
grep s3:// /var/log/pcs-post-install.log

# Monitoring containers (when MonitoringStack=Prometheus-LoginNode)
sudo docker ps --format "table {{.Names}}\t{{.Status}}"
```

For the full reproducible test matrix (monitoring, container runtime, CPU/GPU, NCCL,
FSDP, multi-user, GPU health), see [tests/README.md](../tests/README.md).

## 6. Iterate

After editing a template or script, re-run the **sync** (step 2), then
`update-stack` (same parameters as `create-stack`) — or delete and recreate if the change
can't be applied in place (e.g. a subnet CIDR change):

```bash
aws cloudformation update-stack --stack-name pcs-test \
  --template-url "https://${BUCKET}.s3.amazonaws.com/${PREFIX}pcs-ml-cluster-deploy-all.yaml" \
  --parameters ParameterKey=PrimarySubnetAZ,UsePreviousValue=true \
               ParameterKey=S3BucketName,UsePreviousValue=true \
               ... \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM --region ${REGION}
```

> A stack update only re-runs first-boot scripts on **newly launched** nodes; existing
> nodes keep what they booted with. To re-test a boot-script change, let the node group
> scale a fresh node (or replace the affected nodes).

## 7. Cleanup

```bash
aws cloudformation delete-stack --stack-name pcs-test --region ${REGION}
```

Nested stacks (and FSx — back up data first) are deleted automatically. If a CNG stack
hits `DELETE_FAILED` (a PCS timing dependency), delete the PCS compute node groups first,
then retry:

```bash
for cng in $(aws pcs list-compute-node-groups --cluster-identifier $CLUSTER_ID --region ${REGION} \
  --query 'computeNodeGroups[].id' --output text); do
  aws pcs delete-compute-node-group --cluster-identifier $CLUSTER_ID \
    --compute-node-group-identifier $cng --region ${REGION}
done
sleep 60
aws cloudformation delete-stack --stack-name pcs-test --region ${REGION}
```

## After the change is published

Once the change is merged and the maintainer has synced `assets/` to the public bucket
(`awsome-distributed-ai`), no override is needed — the defaults already point there, and
the [README Quick Start](../README.md#3-quick-start) works as written.
