# AWS ParallelCluster Distributed Training Reference Architecture

This README provides a "vanilla" reference architectures and deployment guide for setting up distributed training clusters using [AWS ParallelCluster](https://github.com/aws/aws-parallelcluster). These architectures are optimized for machine learning workloads and include configurations for high-performance computing instances (P and Trn EC2 families) with shared filesystems (FSx for Lustre and OpenZFS). Key features are including:

- Pre-configured for distributed training workloads
- Integrated with FSx for Lustre for high-performance storage and OpenZFS for home directory
- Support for On-Demand Capacity Reservations (ODCR) and Capacity Blocks (CB)
- Optimized networking with Elastic Fabric Adapter (EFA)

## Architecture

![AWS ParallelCluster diagram](../../assets/core-infra-architecture.png)

The infrastructure consists of the two layers:

### `parallelcluster-prerequisites`

The core infrastructure for your training cluster that has 3 components.

* An [Amazon Virtual Private Cloud (VPC)](https://aws.amazon.com/vpc/) with one public and one private subnet.
* An [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/) high performance file system to store training and checkpointing data.
* An [Amazon FSx for OpenZFS](https://aws.amazon.com/fsx/openzfs/) file system to store home directories data.


### `parallelcluster`

The AWS ParallelCluster is an open-source cluster management tool that makes it easy to deploy and manage High Performance Computing (HPC) clusters on AWS. It automates the creation of the infrastructure needed for HPC workloads, including:

- **Head-node**: A login and controller node that users connect to for submitting jobs and managing the cluster. This node runs the scheduler (Slurm) and other management services.
- **Compute-nodes**: Worker nodes that execute the actual computational workloads. These are dynamically provisioned based on job requirements and can scale up or down automatically.

The architecture follows a traditional HPC design pattern where users interact with the head node to submit jobs, and the scheduler distributes those jobs to the compute nodes. For ML workloads, the compute nodes are typically equipped with GPUs (like P4d, P5, etc.) or AWS Trainium accelerators.


## Prerequisites

Before you begin, ensure you have the following tools installed on your local machine:

- **Git**: For cloning the repository and version control
  - Installation: [Git download page](https://git-scm.com/downloads)

- **Python 3.8 or later**: Required for AWS ParallelCluster CLI
  - Installation: [Python download page](https://www.python.org/downloads/)
  - Verify with: `python3 --version`

- **yq**: A lightweight command-line YAML processor
  - Installation:
    - On macOS: `brew install yq`
    - On Linux: `sudo snap install yq` or `sudo apt-get install yq`
    - On Windows: `choco install yq`
  - Verify with: `yq --version`

These tools are essential for following the deployment steps in this guide and managing your cluster configuration.

First, clone the repository and move to this directory:

```bash
git clone https://github.com/awslabs/awsome-distributed-ai.git
cd awsome-distributed-ai/architectures/aws-parallelcluster
```

Then create a directory under home directory to store cluster config files:

```bash
# For example
export AWS_REGION=ap-northeast-1
export CLUSTER_NAME=ml-cluster
export PCLUSTER_VERSION=3.14.2
export CONFIG_DIR="${HOME}/${CLUSTER_NAME}_${AWS_REGION}_${PCLUSTER_VERSION}"

mkdir -p ${CONFIG_DIR}
touch ${CONFIG_DIR}/config.yaml
yq -i ".CLUSTER_NAME = \"$CLUSTER_NAME\"" ${CONFIG_DIR}/config.yaml
yq -i ".AWS_REGION = \"$AWS_REGION\"" ${CONFIG_DIR}/config.yaml
yq -i ".PCLUSTER_VERSION = \"$PCLUSTER_VERSION\"" ${CONFIG_DIR}/config.yaml
```


The rest of this section describes following required/optional components.

- [AWS Account with administrator permissions](#aws-account-with-appropriate-permissions-to-create-and-manage-resources)
- [AWS ParallelCluster CLI for cluster deployment and management](#aws-parallelcluster-cli-for-cluster-deployment-and-management)
- [Reserved accelerated instance capacity (P/Trn instances) through  On-Demand Capacity Reservation (ODCR) or EC2 Capacity Blocks (CB) for ML](#reserved-accelerated-instance-capacity-ptrn-instances-through--on-demand-capacity-reservation-odcr-or-ec2-capacity-blocks-cb-for-ml)
- [EC2 Key Pair for SSH access](#ec2-key-pair-for-ssh-access)
- [(Optional) S3 bucket for data persistence](#optional-s3-bucket-for-data-persistence)

#### AWS Account with appropriate permissions to create and manage resources

To deploy AWS ParallelCluster, you need to be an Administrator user of the AWS account. See [this issue](https://github.com/aws/aws-parallelcluster/issues/2060) for the related discussion. You need to use the IAM user to create clusters using AWS ParallelCluster CLI from local console, for that you need to configure your AWS credentials by running `aws configure`.

#### AWS ParallelCluster CLI for cluster deployment and management
The AWS ParallelCluster CLI is a command-line tool that helps you deploy and manage HPC clusters on AWS. It provides commands for creating, updating, and deleting clusters, as well as managing cluster resources. The CLI is built on top of the AWS SDK and provides a simple interface for interacting with AWS ParallelCluster. For detailed information about the CLI and its commands, refer to the [AWS ParallelCluster Command Line Interface Reference](https://docs.aws.amazon.com/parallelcluster/latest/ug/commands-v3.html). The CLI requires Python 3.8 or later installed on your local environment.

You can install the AWS ParallelCluster CLI using pip in a Python virtual environment:


```bash
export VIRTUAL_ENV_PATH=~/pcluster_${PCLUSTER_VERSION}_env # change the path to your liking
# Update pip and the virtual env module
python3 -m pip install --upgrade pip
python3 -m pip install --user --upgrade virtualenv
python3 -m virtualenv ${VIRTUAL_ENV_PATH} # create the virtual env
source ${VIRTUAL_ENV_PATH}/bin/activate # activate the environment
pip3 install awscli # install the AWS CLI
pip3 install aws-parallelcluster==${PCLUSTER_VERSION} # then AWS ParallelCluster
```

#### Reserved accelerated instance capacity (P/Trn instances) through  On-Demand Capacity Reservation (ODCR) or EC2 Capacity Blocks (CB) for ML

Distributed training usually requires P or Trn instances, both of which are high demand and hard to launch from the on-demand pool. It is strongly recommended to reserve capacity using On-Demand Capacity Reservation (ODCR) or EC2 Capacity Blocks (CB) for ML.

On-Demand Capacity Reservation (ODCR) is a tool for reserving capacity without having to launch and run the EC2 instances. The CRs for P or Trn instances are typically *created by AWS*, not by users, which affects how to correctly configure the cluster networking.

[Amazon EC2 Capacity Blocks for ML](https://aws.amazon.com/ec2/capacityblocks/) is another way to reserve compute capacity for your ML workloads. Unlike ODCR, Capacity Blocks allows you to reserve capacity for a specific time window (from 1 to 14 days) with a specific start time. This is particularly useful for planned ML training jobs that require high-performance P/Trn instances. Pricing varies by region and instance type. You can find the complete pricing information and available instance types in each region on the [Amazon EC2 Capacity Blocks for ML pricing page](https://aws.amazon.com/ec2/capacityblocks/pricing/).

__You need the following information before proceeding__:

* An ODCR ID (usually for P or Trn instances) on the account: `CAPACITY_RESERVATION_ID`
* Availability Zone for the capacity: `AZ`
* Number of instances in the capacity reservation: `NUM_INSTANCES`
* Instance type: `INSTANCE` (e.g., p5.48xlarge)

Add them to your config file `${CONFIG_DIR}/config.yaml`:

```bash
# Export your capacity reservation details
export CAPACITY_RESERVATION_ID=<your-capacity-reservation-id>  # e.g. cr-0123456789abcdef0
export AZ=<your-availability-zone>  # e.g. ap-northeast-1a
export NUM_INSTANCES=<number-of-instances>  # e.g. 8
export INSTANCE=<instance-type>  # e.g. p5.48xlarge

yq -i ".CAPACITY_RESERVATION_ID = \"$CAPACITY_RESERVATION_ID\"" ${CONFIG_DIR}/config.yaml
yq -i ".AZ = \"$AZ\"" ${CONFIG_DIR}/config.yaml
yq -i ".NUM_INSTANCES = \"$NUM_INSTANCES\"" ${CONFIG_DIR}/config.yaml
yq -i ".INSTANCE = \"$INSTANCE\"" ${CONFIG_DIR}/config.yaml
```


#### EC2 Key Pair for SSH access

The EC2 key pair enables you to connect to your cluster's head node through SSH or [AWS Systems Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-sessions-start.html). We'll cover SSM below.

You can list your existing key pairs in the [AWS Console](https://console.aws.amazon.com/ec2/home?#KeyPairs:) or check your SSH directory for private keys (`~/.ssh` if using Linux or macOS).

First, export the name of the key pair you have or will create:

```bash
export KEYPAIR_NAME=<your-keypair-name>
yq -i ".KEYPAIR_NAME = \"$KEYPAIR_NAME\"" ${CONFIG_DIR}/config.yaml
```

If you don't have a key pair yet, create one using the AWS CLI:

```bash
# Create the key pair and save the private key
pushd ~/.ssh
aws ec2 create-key-pair \
    --key-name ${KEYPAIR_NAME} \
    --query KeyMaterial \
    --key-type ed25519 \
    --region ${AWS_REGION} \
    --output text > ${KEYPAIR_NAME}.pem

# Set appropriate permissions on the private key
chmod 600 ${KEYPAIR_NAME}.pem
popd
```
>[!TIP]
>You can verify the key pair was created successfully by listing your key pairs:
> ```bash
> aws ec2 describe-key-pairs --region ${AWS_REGION}
> ```

#### (Optional) S3 bucket for data persistence

An S3 bucket can be used to persist training data, model checkpoints, and other artifacts across cluster deployments. This is particularly useful for:
- Storing training datasets
- Saving model checkpoints
- Maintaining experiment logs
- Preserving output artifacts

The bucket can be integrated with FSx for Lustre through a Data Repository Association (DRA) later.

To deploy the S3 bucket using our CloudFormation template:

1. Click the button below to launch the CloudFormation stack:

[<kbd> <br> 1-Click Deploy 🚀 <br> </kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/0.private-bucket.yaml&stackName=cluster-data-bucket)

2. In the CloudFormation console:
   - Enter a stack name (e.g., `cluster-data-bucket`)
   - Specify the bucket name
   - Click "Create stack"

3. Once the stack creation is complete, note the bucket name from the Outputs tab. You'll need this for the DRA configuration:

```bash
# Get the bucket name from CloudFormation outputs
# Note: You need to change stack-name if you are using non-default name
export DATA_BUCKET_NAME=$(aws cloudformation describe-stacks \
    --stack-name cluster-data-bucket \
    --query 'Stacks[0].Outputs[?OutputKey==`S3BucketName`].OutputValue' \
    --region ${AWS_REGION} \
    --output text)

echo "Your data bucket name is: ${DATA_BUCKET_NAME}"
yq -i ".DATA_BUCKET_NAME = \"$DATA_BUCKET_NAME\"" ${CONFIG_DIR}/config.yaml
```

> [!NOTE]
> The bucket will be retained even if you delete the cluster, ensuring your data persists across cluster life cycles.

## Cluster deployment

This section guides you through deploying your AWS ParallelCluster environment. You'll set up the necessary infrastructure components including networking, storage, and compute resources to create a high-performance computing cluster optimized for machine learning workloads. The deployment process is streamlined using CloudFormation templates that handle the complex infrastructure provisioning automatically.

### Step1: Deploy parallelcluster-prerequisites

In this section, you deploy a custom [_Amazon Virtual Private Cloud_](https://aws.amazon.com/vpc/) (Amazon VPC) network and security groups, as well as supporting services such as FSx for Lustre using the CloudFormation template called `parallelcluster-prerequisites.yaml`. This template is region agnostic and enables you to create a VPC with the required network architecture to run your workloads.

Please follow the steps below to deploy your resources:

1. Click on this link to deploy to CloudFormation:

[<kbd> <br> 1-Click Deploy 🚀 <br> </kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/parallelcluster-prerequisites.yaml&stackName=parallelcluster-prerequisites)

> [!IMPORTANT]
> When opening the link, you must specify the region and availability zone where your compute resources are located. Be sure to select the correct region and fill out the "Availability Zone configuration for the subnets" field, when you create the stack.


> [!NOTE]
> The above CloudFormation stack uses FSx for Lustre `PERSISTENT_2` deployment type by default. If your selected availability zone doesn't support `PERSISTENT_2` or you specifically need to use `PERSISTENT_1` deployment type, please use the link below instead:
> [<kbd> <br> 1-Click Deploy 🚀 <br> </kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/parallelcluster-prerequisites-p1.yaml&stackName=parallelcluster-prerequisites)

![parallelcluster-prerequisites-cfn](../../assets/parallelcluster-prerequisites-cfn.png)

2. Once the CloudFormation stack deployment is complete, you'll need to export the stack name as an environment variable for future steps:

```bash
export STACK_ID_VPC=parallelcluster-prerequisites # Change here if you uses non-default stack name
yq -i ".STACK_ID_VPC = \"$STACK_ID_VPC\"" ${CONFIG_DIR}/config.yaml
```


#### (Optional) Associate Lustre storage with S3 bucket with data-repository-association (DRA)

If you have deployed your S3 data repository in the previous step In this step, you will create a [Data Repository Association (DRA)](https://docs.aws.amazon.com/fsx/latest/LustreGuide/create-dra-linked-data-repo.html) between the S3 bucket and FSx Lustre Filesystem.

```bash
export FSX_ID=`aws cloudformation describe-stacks \
    --stack-name ${STACK_ID_VPC} \
    --query 'Stacks[0].Outputs[?OutputKey==\`FSxLustreFilesystemId\`].OutputValue' \
    --region ${AWS_REGION} \
    --output text`
```

Note: `DATA_BUCKET_NAME`  is the bucket created in [Step0: Check resource info](https://quip-amazon.com/wDrEAxaBEI3A#temp:C:fdV996b34e8ad4e4dc3ac2ef128b).
Create DRA as follows:

```bash
aws fsx create-data-repository-association \
    --file-system-id ${FSX_ID} \
    --file-system-path "/data" \
    --data-repository-path s3://${DATA_BUCKET_NAME} \
    --s3 AutoImportPolicy='{Events=[NEW,CHANGED,DELETED]},AutoExportPolicy={Events=[NEW,CHANGED,DELETED]}' \
    --batch-import-meta-data-on-create \
    --region ${AWS_REGION}
```

You shall see output like below:

```
{
    "Association": {
        "AssociationId": "dra-0295ef8c2a0e78886",
        "ResourceARN": "arn:aws:fsx:ap-northeast-1:483026362307:association/fs-0160ebe1881498442/dra-0295ef8c2a0e78886",
        "FileSystemId": "fs-0160ebe1881498442",
        "Lifecycle": "CREATING",
        "FileSystemPath": "/data",
        "DataRepositoryPath": "s3://genica-cluster-data-483026362307",
        "BatchImportMetaDataOnCreate": true,
        "ImportedFileChunkSize": 1024,
        "S3": {
            "AutoImportPolicy": {
                "Events": [
                    "NEW",
                    "CHANGED",
                    "DELETED"
                ]
            },
            "AutoExportPolicy": {
                "Events": [
                    "NEW",
                    "CHANGED",
                    "DELETED"
                ]
            }
        },
        "Tags": [],
        "CreationTime": "2024-10-22T09:06:57.151000+09:00"
    }
}
```

[!TIPS]
> You can query the status of the DRA creation as below:
> ```bash
> aws fsx describe-data-repository-associations \
>     --filters "Name=file-system-id,Values=${FSX_ID}" --query "Associations[0].Lifecycle" --output text
>     --region ${AWS_REGION}
> ```
> Wait until the output becomes `AVAILABLE`. You also can check the status of DRA on [AWS console](https://console.aws.amazon.com/fsx/home#file-systems).

## Step2: Deploy ParallelCluster

This section guides you through deploying your distributed training cluster. Before proceeding, ensure you have completed all the [prerequisites](#prerequisites) steps.

### 1. Set Up Environment Variables

First, make sure that you are in virtual environment if you have `pcluster` installed:

```bash
source ${VIRTUAL_ENV_PATH}/bin/activate # activate the environment for pcluster CLI
```

You should have the following values set:

```bash
cat  ${CONFIG_DIR}/config.yaml
```

```yaml
# Example values - these will vary by environment
CLUSTER_NAME: ml-cluster
AWS_REGION: eu-west-2
PCLUSTER_VERSION: 3.14.2
CAPACITY_RESERVATION_ID: cr-XXXXXXXXXXXXXXXXX
AZ: eu-west-2c
NUM_INSTANCES: "16"
INSTANCE: p5.48xlarge
KEYPAIR_NAME: eu-west-2
DATA_BUCKET_NAME: "cluster-data-bucket-us-west-2-XXXXXXXXXXXX"
STACK_ID_VPC: parallelcluster-prerequisites
```

## 2. Generate Cluster Configuration

Retrieve additional environment variables from the `parallelcluster-prerequisites` stack and generate the cluster configuration using our template:

```bash
# Grab all the outputs from the CloudFormation stack in a single command and append to config.yaml
# First create a temporary file with the stack outputs
aws cloudformation describe-stacks \
    --stack-name $STACK_ID_VPC \
    --query 'Stacks[0].Outputs[?contains(@.OutputKey, ``)].{OutputKey:OutputKey, OutputValue:OutputValue}' \
    --region ${AWS_REGION} \
    --output json | yq e '.[] | .OutputKey + ": " + .OutputValue' - > ${CONFIG_DIR}/stack_outputs.yaml
# Merge the stack outputs with config.yaml without duplicating entries
yq eval-all 'select(fileIndex == 0) * select(fileIndex == 1)' ${CONFIG_DIR}/config.yaml ${CONFIG_DIR}/stack_outputs.yaml > ${CONFIG_DIR}/config_updated.yaml
mv ${CONFIG_DIR}/config_updated.yaml ${CONFIG_DIR}/config.yaml
rm ${CONFIG_DIR}/stack_outputs.yaml
# Now let's use the following snippet to read them as environment variables
eval $(yq e 'to_entries | .[] | "export " + .key + "=\"" + .value + "\""' ${CONFIG_DIR}/config.yaml)

# You can also view the updated config file
cat ${CONFIG_DIR}/config.yaml

# Read all the environment variables we have in config.yaml
eval $(yq e 'to_entries | .[] | "export " + .key + "=\"" + .value + "\""' ${CONFIG_DIR}/config.yaml)
# Generate the configuration file
cat cluster-templates/cluster-vanilla.yaml | envsubst > ${CONFIG_DIR}/cluster.yaml
```

> [!IMPORTANT]
> The default configuration has `PlacementGroup` set to `False`. This is recommended when using AWS-provided ODCR or Capacity Blocks, as enabling placement groups may conflict with the placement group assigned to your ODCR/CB and cause *Insufficient Capacity Error* (ICE).
>
> However, when using user-procured On-demand Capacity Reservations (ODCR) for distributed training workloads, you should set `PlacementGroup` to `True` and specify a cluster placement group name. This ensures optimal network performance between instances. Configure it like this:
>
> ```yaml
> PlacementGroup:
>   Enabled: True
>   Name: <Your Placement Group Name>
> ```
>
> It's recommended to create a cluster placement group first and use its name when reserving capacity for distributed training workloads.

> [!TIP]
> If using AWS CloudShell and `envsubst` is not available, install it with:
> ```bash
> sudo yum install gettext
> ```

### 3. Create the Cluster

Deploy your cluster using the generated configuration:

```bash
pcluster create-cluster \
    --cluster-name ${CLUSTER_NAME} \
    --cluster-configuration ${CONFIG_DIR}/cluster.yaml \
    --region ${AWS_REGION} \
    --rollback-on-failure false
```

You should see output similar to:
```json
{
  "cluster": {
    "clusterName": "ml-cluster",
    "cloudformationStackStatus": "CREATE_IN_PROGRESS",
    "cloudformationStackArn": "arn:aws:cloudformation:ap-northeast-1:123456789012:stack/ml-cluster/abcd1234-...",
    "region": "ap-northeast-1",
    "version": "3.14.2",
    "clusterStatus": "CREATE_IN_PROGRESS",
    "scheduler": {
      "type": "slurm"
    }
  }
}
```

### 4. Monitor Cluster Creation

You can monitor the cluster creation progress in several ways:

1. Using the AWS ParallelCluster CLI:
```bash
pcluster list-clusters --region ${AWS_REGION}
```

2. Through the CloudFormation console:
   - Navigate to the [CloudFormation console](https://console.aws.amazon.com/cloudformation)
   - Select your cluster's stack
   - Monitor the "Events" tab for real-time updates

The cluster creation typically takes 15-20 minutes. Wait for the status to show "CREATE_COMPLETE" before proceeding to connect to your cluster.

### Next Steps

Once your cluster is ready:
- [Connect to your cluster](#connect-to-the-cluster) using SSH or AWS Systems Manager
- Set up your development environment
- Start submitting training jobs

For advanced configurations and customizations, refer to our [deployment guides](./deployment-guides).

## Connect to the Cluster

Once the cluster goes into **CREATE COMPLETE**, we can connect to the head node in one of two ways, either through the SSM or SSH.

**SSM Session Manager** is ideal for quick terminal access to the head node, it doesn't require any ports to be open on the head node, however it does require you to authenticate with the AWS account the instance it running in.

**SSH** can be used to connect to the cluster from a standard SSH client. This can be configured to use your own key via adding the public key or a new key can be provisioned.

### SSM Connect
![ssm connect](../../assets/ssm-connect.png)
You'll need to be authenticated to the AWS account that instance is running in and have permission to launch a SSM session . Once you're connected you'll have access to a terminal on the head node:

Now change to `ubuntu` user:

```bash
sudo su - ubuntu
```

![ssm user connect](../../assets/ssm-connect-user.png)

### SSH access

Also, You can access to the headnode via SSH (if you set up keypair). You can retrieve IP address of the head node with the following command:

```bash
pcluster ssh --region ap-northeast-1 --cluster-name ml-cluster --identity_file ~/.ssh/ap-northeast-1.pem  --dryrun true
```

It will show output like follows:

```bash
{
  "command": "ssh ubuntu@18.183.235.248 --identity_file /Users/mlkeita/.ssh/ap-northeast-1.pem"
}
```


## 8. References

* [AWS ParallelCluster wiki](https://github.com/aws/aws-parallelcluster/wiki)

