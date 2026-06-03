# Deploy SageMaker HyperPod Slurm Infrastructure using Terraform

This directory contains Terraform modules to deploy a complete SageMaker HyperPod cluster with Slurm orchestration, including all necessary infrastructure components.

## Architecture Overview

The Terraform modules create:
- VPC with public and private subnets
- Security groups configured for EFA communication
- FSx for Lustre file system
- Optional (FSx OpenZFS file system)
- S3 bucket for lifecycle scripts
- IAM roles and policies
- SageMaker HyperPod cluster with Slurm orchestration

## Quick Start

1. **Clone and Navigate**
   ```bash
   git clone https://github.com/awslabs/awsome-distributed-ai.git
   cd awsome-distributed-ai/architectures/sagemaker-hyperpod-slurm/terraform-modules/hyperpod-slurm-tf
   ```

2. **Customize Configuration**
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your specific requirements
   ```

3. **Deploy**

    Initialize Terraform:

    ```bash
    terraform init
    ```

    Review the deployment plan:

    ```bash
    terraform plan
    ```

    Deploy the infrastructure:

    ```bash
    terraform apply
    ```

    When prompted, type `yes` to confirm the deployment.

    After successful deployment, extract the outputs:

    ```bash
    ./terraform_outputs.sh
    source env_vars.sh

## Configuration

### Basic Configuration Example

```hcl
# terraform.tfvars
resource_name_prefix = "hyperpod"
aws_region = "us-west-2"
availability_zone_id = "usw2-az2"

hyperpod_cluster_name = "ml-cluster"

instance_groups = {
  controller-machine = {
    instance_type = "ml.c5.2xlarge"
    instance_count = 1
    ebs_volume_size = 100
    threads_per_core = 1
    lifecycle_script = "on_create.sh"
  }
  login-nodes = {
    instance_type    = "ml.m5.4xlarge"
    instance_count   = 1
    ebs_volume_size  = 100
    threads_per_core = 1
    lifecycle_script = "on_create.sh"
  }
  compute-nodes = {
    instance_type = "ml.g5.4xlarge"
    instance_count = 2
    ebs_volume_size = 500
    threads_per_core = 1
    lifecycle_script = "on_create.sh"
  }
}
```

### Using Existing Resources

To reuse existing infrastructure, set the corresponding `create_*_module` to `false` and provide the existing resource ID:

```hcl
create_vpc_module = false
existing_vpc_id = "vpc-1234567890abcdef0"
existing_private_subnet_id = "subnet-1234567890abcdef0"
existing_security_group_id = "sg-1234567890abcdef0"
```
## OpenZFS
FSx OpenZFS provides NFS-based shared storage suitable for user home directories and general-purpose file sharing. FSx OpenZFS provides NFS-based shared storage suitable for user home directories and general-purpose file sharing.

To enable FSx OpenZFS (refer to `terraform.tfvars.example`), add to your `terraform.tfvars`:

```hcl
create_fsx_openzfs_module = true
fsx_openzfs_storage_capacity = 64
fsx_openzfs_throughput_capacity = 64
fsx_openzfs_compression_type = "ZSTD"
```
Then set ```enable_fsx_openzfs = True ``` in ```../../LifecycleScripts/base-config/config.py``` to mount the file system during the lifecycle scripts execution. 
    
## Modules

- **vpc**: Creates VPC with public/private subnets, IGW, NAT Gateway
- **security_group**: EFA-enabled security group for HyperPod
- **fsx_lustre**: High-performance Lustre file system
- **fsx_openzfs**: OpenZFS file system for home directory
- **s3_bucket**: Storage for lifecycle scripts
- **sagemaker_iam_role**: IAM role with required permissions
- **lifecycle_script**: Uploads and configures Slurm lifecycle scripts
- **hyperpod_cluster**: SageMaker HyperPod cluster with Slurm

## Lifecycle Scripts

The modules automatically upload the base Slurm configuration from `../../LifecycleScripts/base-config/` to your S3 bucket. These scripts:

- Configure Slurm scheduler
- Mount FSx Lustre file system
- Mount FSx OpenZFS file system if specified
- Install Docker, Enroot, and Pyxis
- Set up user accounts and permissions

## Accessing Your Cluster

After deployment, use the provided helper script:

```bash
./easy-ssh.sh <cluster-name> <region>
```

Or manually:
```bash
aws ssm start-session --target sagemaker-cluster:${CLUSTER_ID}_${CONTROLLER_GROUP}-${INSTANCE_ID}
```

## Clean Up

```bash
terraform destroy
```
