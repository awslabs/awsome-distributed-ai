# Deploy HyperPod Infrastructure using Terraform

The diagram below depicts the Terraform modules that have been bundled into a single project to enable you to deploy a full HyperPod cluster environment all at once. 

<img src="./smhp_tf_modules.png" width="75%"/>

---


## Get the Modules
Clone the AWSome Distributed Training repository and navigate to the terraform-modules directory:
```bash
git clone https://github.com/awslabs/awsome-distributed-training.git
cd awsome-distributed-training/1.architectures/7.sagemaker-hyperpod-eks/terraform-modules/hyperpod-eks-tf
```

---

## Deployment 
First, clone the [HyperPod Helm charts GitHub repository](https://github.com/aws/sagemaker-hyperpod-cli/tree/main/helm_chart) to locally stage the dependencies Helm chart.  
```bash
git clone https://github.com/aws/sagemaker-hyperpod-cli.git /tmp/helm-repo
```
Run `terraform init` to initialize the Terraform working directory, install necessary provider plugins, download modules, set up state storage, and configure the backend for managing infrastructure state: 

```bash 
terraform init
```
Run `terraform plan` to generate and display an execution plan that outlines the changes Terraform will make to your infrastructure, allowing you to review and validate the proposed updates before applying them.

```bash 
terraform plan
```
If you created a `custom.tfvars` file, plan using the `-var-file` flag: 
```bash 
terraform plan -var-file=custom.tfvars
```
Or for RIG deployments:
```bash
terraform plan -var-file=rig_custom.tfvars
```
Run `terraform apply` to execute the proposed changes outlined in the Terraform plan, creating, updating, or deleting infrastructure resources according to your configuration, and updating the state to reflect the new infrastructure setup.

```bash 
terraform apply 
```
If you created a `custom.tfvars` file, apply using the `-var-file` flag: 
```bash
terraform apply  -var-file=custom.tfvars
```
Or for RIG deployments: 
```bash
terraform apply -var-file=rig_custom.tfvars
```
When prompted to confirm, type `yes` and press enter.

You can also run `terraform apply` with the `-auto-approve` flag to avoid being prompted for confirmation, but use with caution to avoid unintended changes to your infrastructure. 

---

## Environment Variables
Run the `terraform_outputs.sh` script, which populates the `env_vars.sh` script with your environment variables for future reference: 
```bash 
cd ..
chmod +x terraform_outputs.sh
./terraform_outputs.sh
cat env_vars.sh 
```
Source the `env_vars.sh` script to set your environment variables: 
```bash 
source env_vars.sh
```
Verify that your environment variables are set: 
```bash
echo $EKS_CLUSTER_NAME
echo $PRIVATE_SUBNET_ID
echo $SECURITY_GROUP_ID
```

---

## Clean Up

Before cleaning up, validate the changes by running a speculative destroy plan: 

```bash
cd hyperpod-eks-tf
terraform plan -destroy
```

Before destroying resources, list state to exclude any resources you wish to retain from deletion:
```bash
terraform state list
terraform state rm < resource_to_preserve >
```

If you created a `custom.tfvars` file, plan using the `-var-file` flag: 
```bash
terraform plan -destroy -var-file=custom.tfvars
```
Or for RIG deployments:
```bash
terraform plan -destroy -var-file=rig_custom.tfvars
```
Once you've validated the changes, you can proceed to destroy the resources: 
```bash 
terraform destroy
```
If you created a `custom.tfvars` file, destroy using the `-var-file` flag: 
```bash
terraform destroy -var-file=custom.tfvars
```
Or for RIG deployments: 
```bash
terraform destroy -var-file=rig_custom.tfvars
```

---

## Customize Deployment Configuration
Start by reviewing the default configurations in the `terraform.tfvars` file and make modifications to customize your deployment as needed.

If you wish to reuse any cloud resources rather than creating new ones, set the associated `create_*_module` variable to `false` and provide the id for the corresponding resource as the value of the `existing_*` variable. 

For example, if you want to reuse an existing VPC, set `create_vpc_module ` to `false`, then set `existing_vpc_id` to your VPC ID, like `vpc-1234567890abcdef0`. 

---

### Using a `custom.tfvars` File 
To modify your deployment details without having to open and edit the `terraform.tfvars` file directly, create a `custom.tfvars` file with your parameter overrides. 

For example, the following `custom.tfvars` file would enable the creation of all new resources including a new EKS Cluster and a HyperPod instance group of 5 `ml.p5en.48xlarge` instances in `us-west-2` using a [training plan](https://docs.aws.amazon.com/sagemaker/latest/dg/reserve-capacity-with-training-plans.html):

```bash
cat > custom.tfvars << EOL 
kubernetes_version = "1.32"
eks_cluster_name = "my-eks-cluster"
hyperpod_cluster_name = "my-hp-cluster"
resource_name_prefix = "hp-eks-test"
aws_region = "us-west-2"
instance_groups = [
    {
        name = "accelerated-instance-group-1"
        instance_type = "ml.p5en.48xlarge",
        instance_count = 5,
        availability_zone_id  = "usw2-az2",
        ebs_volume_size_in_gb = 100,
        threads_per_core = 2,
        enable_stress_check = true,
        enable_connectivity_check = true,
        lifecycle_script = "on_create.sh"
        training_plan_arn = arn:aws:sagemaker:us-west-2:123456789012:training-plan/training-plan-example
    }
]
EOL
```
---

### Configuring Kubernetes Labels and Taints

You can add custom Kubernetes labels and taints to your HyperPod instance groups to control pod scheduling and node organization. This is useful for:
- Organizing nodes by workload type, team, or environment
- Dedicating GPU nodes to specific workloads
- Implementing multi-tenant cluster configurations
- Controlling pod placement with node selectors and tolerations

**Adding Labels:**

Labels are key-value pairs for node identification and selection:

```hcl
instance_groups = [
  {
    name                      = "gpu-training-group"
    instance_type             = "ml.g5.12xlarge"
    instance_count            = 4
    availability_zone_id      = "usw2-az2"
    ebs_volume_size_in_gb     = 500
    threads_per_core          = 1
    enable_stress_check       = false
    enable_connectivity_check = false
    lifecycle_script          = "on_create.sh"
    
    # Custom labels
    labels = {
      "workload-type" = "training"
      "gpu-type"      = "a10g"
      "team"          = "ml-research"
    }
  }
]
```

**Adding Taints:**

Taints prevent pods without matching tolerations from scheduling on nodes:

```hcl
instance_groups = [
  {
    name                      = "gpu-inference-group"
    instance_type             = "ml.g5.12xlarge"
    instance_count            = 2
    availability_zone_id      = "usw2-az2"
    ebs_volume_size_in_gb     = 200
    threads_per_core          = 1
    enable_stress_check       = false
    enable_connectivity_check = false
    lifecycle_script          = "on_create.sh"
    
    # Custom taints (max 50 per instance group)
    taints = [
      {
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NoSchedule"  # NoSchedule, PreferNoSchedule, or NoExecute
      },
      {
        key    = "dedicated"
        value  = "inference"
        effect = "NoSchedule"
      }
    ]
  }
]
```

**Taint Effects:**
- `NoSchedule`: Pods without matching tolerations won't be scheduled
- `PreferNoSchedule`: Kubernetes tries to avoid scheduling non-matching pods
- `NoExecute`: Existing pods without matching tolerations will be evicted

---

### Using an Existing EKS Cluster with HyperPod

The following `custom.tfvars` file uses an existing EKS Cluster (referenced by name) along with an existing Security Group, VPC, and NAT Gateway (referenced by ID):
```bash
cat > custom.tfvars << EOL 
create_eks_module = false
existing_eks_cluster_name = "my-eks-cluster"
existing_security_group_id = "sg-1234567890abcdef0"
create_vpc_module = false
existing_vpc_id = "vpc-1234567890abcdef0"
existing_nat_gateway_id = "nat-1234567890abcdef0"
hyperpod_cluster_name = "my-hp-cluster"
resource_name_prefix = "hp-eks-test"
aws_region = "us-west-2"
instance_groups = [
    {
        name = "accelerated-instance-group-1"
        instance_type = "ml.p5en.48xlarge",
        instance_count = 5,
        availability_zone_id  = "usw2-az2",
        ebs_volume_size_in_gb = 100,
        threads_per_core = 2,
        enable_stress_check = true,
        enable_connectivity_check = true,
        lifecycle_script = "on_create.sh"
        training_plan_arn = arn:aws:sagemaker:us-west-2:123456789012:training-plan/training-plan-example
    }
]
EOL
```
---

### Closed Network Deployment

For air-gapped or closed network environments without internet access:

#### Prerequisites: Copy Images to ECR and Prepare Helm Chart

**BEFORE running Terraform**, copy container images to your private ECR and update the Helm chart:

```bash
# Navigate to the terraform modules directory
cd awsome-distributed-training/1.architectures/7.sagemaker-hyperpod-eks/terraform-modules/hyperpod-eks-tf

# 1. Set your AWS account and region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=$(aws configure get region)  # Or set explicitly: export AWS_REGION="us-west-2"

echo "Account ID: $AWS_ACCOUNT_ID"
echo "Region: $AWS_REGION"

# 2. Copy container images to your private ECR
# This copies all required images from public registries (NVIDIA, Kubeflow, etc.) to your ECR
cd tools
./copy-images-to-ecr.sh $AWS_REGION $AWS_ACCOUNT_ID
cd ..

# 3. Clone sagemaker-hyperpod-cli (if not already done)
git clone https://github.com/aws/sagemaker-hyperpod-cli.git

# 4. Update Helm chart to reference your private ECR images
# IMPORTANT: Run from hyperpod-eks-tf directory so script finds tools/ecr-images.conf
python3 tools/update-values-with-ecr.py $AWS_REGION $AWS_ACCOUNT_ID

# 5. Commit changes locally
cd sagemaker-hyperpod-cli
git add -A && git commit -m "Update to private ECR images"

# 6. Get commit hash for terraform.tfvars
git rev-parse HEAD
# Copy this hash - you'll use it in helm_repo_revision variable

# 7. Copy entire git repo to /tmp/helm-repo for Terraform
# IMPORTANT: Copy the entire repo (with .git) so Terraform can checkout the commit
cd ..
rm -rf /tmp/helm-repo  # Remove if exists
cp -r sagemaker-hyperpod-cli /tmp/helm-repo
```

**What these steps do:**
- **Step 2**: Copies images from public registries to your private ECR (creates ECR repos automatically)
- **Step 4**: Updates Helm chart `values.yaml` to use your ECR instead of public registries
- **Step 7**: Stages the updated Helm chart where Terraform expects it

**Important**: All commands should be run from the `hyperpod-eks-tf` directory so the scripts can find the configuration files.

#### Deployment Configuration

The `closed-network.tfvars` file provides a complete example for deploying in a closed network environment.

**Option 1: Create New Closed Network VPC**

Use the provided example as-is to create a brand new closed network:
```bash
terraform plan -var-file=closed-network.tfvars
terraform apply -var-file=closed-network.tfvars
```

**Option 2: Use Existing Resources**

To use existing VPC, subnets, or other resources, modify `closed-network.tfvars`:

```hcl
# Use existing VPC instead of creating new one
create_vpc_module = false
existing_vpc_id   = "vpc-xxxxx"

# Use existing private subnets
create_private_subnet_module = false
existing_private_subnet_ids  = ["subnet-xxxxx", "subnet-yyyyy", "subnet-zzzzz"]

# Use existing security group
create_security_group_module = false
existing_security_group_id   = "sg-xxxxx"

# Use existing EKS cluster
create_eks_module         = false
existing_eks_cluster_name = "my-existing-cluster"

# Use existing S3 bucket
create_s3_bucket_module = false
existing_s3_bucket_name = "my-existing-bucket"

# Reuse existing VPC endpoints (if VPC already has them)
create_vpc_endpoints_module      = false
existing_private_route_table_ids = ["rtb-xxxxx"]
```

**IMPORTANT: When Reusing VPC Endpoints with a New Security Group**

If you're creating a new security group (`create_security_group_module = true`) but reusing existing VPC endpoints (`create_vpc_endpoints_module = false`), the deployment will fail because pods can't access the VPC endpoints. Follow this workflow:

**Step 1: Initial deployment (will fail)**
```bash
terraform apply -var-file=your-config.tfvars
```

The deployment will fail with this error after ~5 minutes:
```
Error: local-exec provisioner error
  with module.hyperpod_cluster[0].null_resource.wait_for_hyperpod_nodes[0],
  on modules/hyperpod_cluster/main.tf line 133, in resource "null_resource" "wait_for_hyperpod_nodes":
  
Error running command: exit status 1. Output:
Waiting for EKS Pod Identity Agent to be ready...
error: timed out waiting for the condition on pods/eks-pod-identity-agent-xxxxx
```

This is expected! The security group was created but pods can't reach VPC endpoints yet.

**Step 2: Add new security group to VPC endpoints**
```bash
# Get your new security group ID from the partial deployment
NEW_SG_ID=$(terraform output -raw security_group_id)
VPC_ID=$(terraform output -raw vpc_id)  
REGION=$(terraform output -raw aws_region)  

# Get all interface VPC endpoint IDs
ENDPOINT_IDS=$(aws ec2 describe-vpc-endpoints \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=vpc-endpoint-type,Values=Interface" \
  --query 'VpcEndpoints[].VpcEndpointId' \
  --output text --region $REGION)

# Add new security group to each VPC endpoint (convert tabs to newlines for proper iteration)
echo "$ENDPOINT_IDS" | tr '\t' '\n' | while read VPCE_ID; do
  if [ -n "$VPCE_ID" ]; then
    echo "Adding security group to $VPCE_ID..."
    aws ec2 modify-vpc-endpoint \
      --vpc-endpoint-id $VPCE_ID \
      --add-security-group-ids $NEW_SG_ID \
      --region $REGION
  fi
done

echo "All VPC endpoints updated!"
```

**Step 3: Delete failing pods and complete deployment**
```bash
# Delete pods that failed to pull images
kubectl delete pods -n kube-system -l app.kubernetes.io/name=eks-pod-identity-agent

# Complete the deployment
terraform apply -var-file=your-config.tfvars
```

**Alternative: Reuse existing security group** to avoid this issue entirely:
```hcl
create_security_group_module = false
existing_security_group_id   = "sg-xxxxx"  # Same SG used by VPC endpoints
```

**Note on Cleanup:** When destroying resources, the security group deletion may take 5-10 minutes because AWS automatically detaches it from all VPC endpoints. This is normal. If the destroy times out or gets stuck, you may need to manually remove the security group from VPC endpoints before retrying:
```bash
# Set your values
NEW_SG_ID="sg-029efc6343bdb7d05"  
VPC_ID="vpc-09ab7b104c4c92266"    
REGION="us-west-2"    

# Get all interface VPC endpoint IDs
ENDPOINT_IDS=$(aws ec2 describe-vpc-endpoints \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=vpc-endpoint-type,Values=Interface" \
  --query 'VpcEndpoints[].VpcEndpointId' \
  --output text --region $REGION)

# Remove security group from each endpoint
echo "$ENDPOINT_IDS" | tr '\t' '\n' | while read VPCE_ID; do
  if [ -n "$VPCE_ID" ]; then
    echo "Removing security group from $VPCE_ID..."
    aws ec2 modify-vpc-endpoint \
      --vpc-endpoint-id $VPCE_ID \
      --remove-security-group-ids $NEW_SG_ID \
      --region $REGION
  fi
done
```

**Key Configuration Details:**

**VPC Endpoints (Required for Closed Networks):**
- **S3** (Gateway) - Free - Container images and data access
- **EC2** (Interface) - CRITICAL - AWS CNI plugin needs this to assign IPs to pods
- **ECR API/DKR** (Interface) - Pull container images from ECR
- **STS** (Interface) - IAM role assumption (IRSA)
- **EKS Auth** (Interface) - CRITICAL - EKS Pod Identity authentication
- **CloudWatch Logs/Monitoring** (Interface) - Observability
- **SSM/SSM Messages/EC2 Messages** (Interface) - Systems Manager access

**EKS API Access:**
```hcl
eks_endpoint_private_access = true   # Required for nodes to join
eks_endpoint_public_access  = true   # Disable after deployment for full isolation
```

#### Deployment Steps

Once prerequisites are complete and your Helm chart is prepared:

```bash
# Navigate to terraform directory
cd awsome-distributed-training/1.architectures/7.sagemaker-hyperpod-eks/terraform-modules/hyperpod-eks-tf

# Initialize Terraform
terraform init

# Plan deployment (review changes)
terraform plan -var-file=closed-network.tfvars

# Apply deployment
terraform apply -var-file=closed-network.tfvars
```

#### Verification

After deployment, verify connectivity to AWS services:
```bash
python3 tools/verify-aws-connectivity.py
```

---

### Local Zone Deployment

You can place HyperPod worker instance groups in an [AWS Local Zone](https://docs.aws.amazon.com/local-zones/latest/ug/what-is-aws-local-zones.html) to run compute closer to a specific metro area. The EKS control plane stays in the parent Region: EKS cannot create control-plane ENIs in a Local Zone, so only the HyperPod worker subnet lives in the Local Zone while the control-plane subnets remain in standard parent Availability Zones.

Local Zone support is opt-in and additive. All of the variables below default to standard-AZ behavior when unset, so existing deployments are unaffected.

> **Note:** Not every Local Zone is supported by HyperPod, and there is no API to enumerate the supported zones. Confirm your target zone with AWS before deploying. The instance type must also be both offered in the Local Zone and present in HyperPod's `ClusterInstanceType` enum.

#### Local Zone Variables

| Variable | Usage |
|----------|-------|
| `private_subnet_availability_zone_ids` | Pins the HyperPod private (worker) subnets to explicit Availability Zone IDs, 1:1 with `private_subnet_cidrs`. This bypasses the `opt-in-status = "opt-in-not-required"` AZ-discovery filter, which excludes opt-in Local Zones. Default `[]` = discover standard AZs automatically. |
| `local_zone_egress_zone_ids` | List of Local Zone AZ IDs that should get a Local-Zone-local NAT gateway. Default `[]` = worker subnets route through the regional NAT gateway. |
| `local_zone_public_subnet_cidrs` | Local Zone public subnet CIDRs, 1:1 with `local_zone_egress_zone_ids`. Typically carved from the VPC primary CIDR (secondary CIDRs are usually consumed by the worker subnet). |
| `local_zone_network_border_groups` | `NetworkBorderGroup` names for the Local Zone NAT Elastic IPs, 1:1 with `local_zone_egress_zone_ids`. Required: a plain VPC-scoped EIP cannot attach to a NAT gateway in a Local Zone subnet. The border group is the Local Zone name minus the trailing zone letter (e.g. `us-west-2-phx-2a` -> `us-west-2-phx-2`). |

#### Local Zone egress (NAT placement)

By default the `vpc` module creates a single regional NAT gateway in a standard-AZ public subnet. A worker subnet in a Local Zone routes `0.0.0.0/0` to that regional NAT, so egress traffic hairpins back to the parent Region and pays an added round trip per packet.

Setting the three `local_zone_*` variables creates one Local-Zone-local NAT gateway per listed zone (with a border-group-scoped EIP) and routes matching worker subnets to it via the `vpc` module's `nat_gateway_ids_by_zone_id` output. Keeping egress in-zone significantly improves first-hop latency and internet throughput for Local Zone workers. Unmapped AZs continue to use the regional NAT. This has been validated with an end-to-end Local Zone HyperPod deployment.

#### Example `custom.tfvars`

```hcl
resource_name_prefix = "hp-eks"
aws_region           = "us-west-2"

# VPC
create_vpc_module    = true
vpc_cidr             = "10.192.0.0/16"
public_subnet_1_cidr = "10.192.10.0/24"
public_subnet_2_cidr = "10.192.11.0/24"

# Private (worker) subnet pinned to the Local Zone AZ ID, 1:1 with the CIDR.
# This bypasses the opt-in-not-required discovery filter that excludes Local Zones.
create_private_subnet_module         = true
private_subnet_cidrs                 = ["10.1.0.0/16"]
private_subnet_availability_zone_ids = ["usw2-phx2-az1"]

# Optional: Local-Zone-local NAT gateway (all three lists non-empty and 1:1).
# Uncomment to keep worker egress in-zone instead of hairpinning to the Region.
# local_zone_egress_zone_ids       = ["usw2-phx2-az1"]
# local_zone_public_subnet_cidrs   = ["10.192.20.0/24"]
# local_zone_network_border_groups = ["us-west-2-phx-2"]

# EKS control-plane subnets stay in parent AZs (cannot live in a Local Zone).
create_eks_module         = true
create_eks_subnets        = true
eks_private_subnet_1_cidr = "10.192.7.0/28"
eks_private_subnet_2_cidr = "10.192.8.0/28"

# FSx placement: default co-locates FSx with the instance group's subnet (in-Local-Zone).
create_fsx_module         = true
create_new_fsx_filesystem = true
fsx_storage_capacity      = 1200
fsx_throughput            = 250
# fsx_availability_zone_id = ""  # set to a parent-AZ ID for a cross-zone mount
                                 # if the Local Zone does not offer FSx (or the tier).

instance_groups = [
  {
    name                      = "instance-group-1"
    instance_type             = "ml.c6i.2xlarge"
    instance_count            = 1
    availability_zone_id      = "usw2-phx2-az1" # land workers in the Local Zone
    ebs_volume_size_in_gb     = 100
    threads_per_core          = 2
    enable_stress_check       = false
    enable_connectivity_check = false
    lifecycle_script          = "on_create.sh"
  }
]
```

#### FSx for Lustre in a Local Zone

FSx placement is already configurable through `fsx_availability_zone_id` (see the [FSx for Lustre Module](#fsx-for-lustre-module) section). When empty (default), FSx is created in the first instance group's subnet, which co-locates it with compute in the Local Zone. FSx for Lustre availability and per-tier support vary by Local Zone; if your target zone does not offer FSx (or the tier you need), set `fsx_availability_zone_id` to a parent-AZ ID for a cross-zone mount, or set `create_new_fsx_filesystem = false`.

#### Prerequisite: opt in to the Local Zone

The target Local Zone must be opted in before you deploy (a not-yet-opted-in zone makes the private subnet fail to create):

```bash
# Look up your Local Zone's AZ ID and parent zone
aws ec2 describe-availability-zones --all-availability-zones \
  --query "AvailabilityZones[?ZoneType=='local-zone'].[ZoneName,ZoneId,ParentZoneName]" \
  --output table

# Opt in (opt-in is asynchronous - verify it reports opted-in before deploying)
aws ec2 modify-availability-zone-group \
  --group-name us-west-2-phx-2a --opt-in-status opted-in
```

For a complete, ready-to-run Local Zone example, see the [HyperPod Local Zone quickstart](https://github.com/aravneelaws/hyperpod-local-zone-quickstart/tree/main/terraform/eks). That repository ships only a `local-zone.tfvars` file and applies it against this reference stack (no forked Terraform), so the variable file lives there while the modules live here.

---
### Enabling Optional Addons 
Set the following parameters to `true` in your `custom.tfvars` file to enable optional addons for your HyperPod cluster (e.g. `create_task_governance_module = true`):
| Parameter | Usage |
|-----------|-------|
| `create_task_governance_module`    | Installs the [HyperPod task governance addon](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-operate-console-ui-governance.html) for  job queuing, prioritization, and scheduling on multi-team compute clusters |
| `create_hyperpod_training_operator_module`   | Installs the [HyperPod training operator addon](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-eks-operator.html) for intelligent fault recovery, hang job detection, and process-level management capabilities (required for [Checkpointless](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-eks-checkpointless.html) and [Elastic](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-eks-elastic-training.html) training)|
| `create_hyperpod_inference_operator_module`  | Installs the [HyperPod inference operator addon](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment-setup.html) for deployment and management of machine learning inference endpoints |
| `create_observability_module` | Installs the [HyperPod Observability addon](https://docs.aws.amazon.com/sagemaker/latest/dg/hyperpod-observability-addon-setup.html) to publish key metrics to Amazon Managed Service for Prometheus and displays them in Amazon Managed Grafana dashboards | 

#### Task governance compute allocations

When `create_task_governance_module = true`, you can also manage SageMaker HyperPod task governance compute allocations with Terraform by setting `task_governance_compute_quotas`. The Terraform providers do not currently expose a first-class SageMaker compute quota resource, so this module uses the AWS CLI from a Terraform `local-exec` provisioner to call the SageMaker `create-compute-quota`, `update-compute-quota`, and `delete-compute-quota` APIs. The machine running Terraform must have the AWS CLI, `kubectl`, and `python3` installed and authenticated for the target account.

The `local-exec` wrapper is a temporary compatibility layer and should be migrated to a native AWS provider resource when one becomes available. Compute quota descriptions are managed as Terraform desired state: if `description` is omitted, it defaults to an empty string and will clear an existing remote description on update.

```hcl
create_task_governance_module = true

task_governance_compute_quotas = [
  {
    name        = "team-a-quota"
    description = "Team A compute allocation"

    compute_quota_resources = [
      {
        instance_type = "ml.g5.8xlarge"
        count         = 2
      }
    ]

    resource_sharing_config = {
      strategy = "DontLend"
    }

    preempt_team_tasks = "LowerPriority"

    target = {
      team_name         = "team-a"
      fair_share_weight = 0
    }
  }
]
```

For teams that lend and borrow idle capacity, use `strategy = "LendAndBorrow"` and optionally set `borrow_limit` or `absolute_borrow_limits`.

```hcl
task_governance_compute_quotas = [
  {
    name = "team-b-quota"

    compute_quota_resources = [
      {
        instance_type = "ml.g5.8xlarge"
        count         = 4
      }
    ]

    resource_sharing_config = {
      strategy     = "LendAndBorrow"
      borrow_limit = 50
    }

    preempt_team_tasks = "LowerPriority"

    target = {
      team_name         = "team-b"
      fair_share_weight = 50
    }
  }
]
```

The HyperPod training and inference operators both require the [cert-manager](https://cert-manager.io/) EKS addon to be installed as a prerequisite. The variable `enable_cert_manager` is set to `true` by default, so that when `create_hyperpod_training_operator_module` or `create_hyperpod_inference_operator_module` are also set to `true`, cert-manager will be installed as a dependency of the operators. In other words, this stack will not install cert-manager as a standalone component, but it can be disabled if you already have it installed on an existing EKS cluster and wish to use one of the HyperPod operators. 

The HyperPod inference operator also has the following additional dependencies: 
  - The [Amazon FSx for Lustre CSI driver](https://github.com/kubernetes-sigs/aws-fsx-csi-driver): This EKS addon is installed by default as part of the FSx for Lustre module. Set `create_fsx_module = false` if you already have it installed on an existing EKS cluster. 
  - The [Mountpoint for Amazon S3 CSI Driver](https://github.com/awslabs/mountpoint-s3-csi-driver): This EKS addon is bundled with the Hyperpod inference operator module and is enabled by default. Set `enable_s3_csi_driver = false` if you already have it installed on an existing EKS cluster.
  - The [AWS Load Balancer Controller](https://github.com/kubernetes-sigs/aws-load-balancer-controller): This is bundled with the HyperPod inference operator EKS addon and is enabled by default. Set `enable_alb_controller = false` if you already have it installed on an existing EKS cluster.
  - The [KEDA (Kubernetes Event-driven Autoscaling) Operator](https://keda.sh/): This is bundled with the HyperPod inference operator EKS addon and is enabled by default. Set `enable_keda = false` if you already have it installed on an existing EKS cluster.

---
### Advanced Observability Metrics Configuration
In addition to enabling the [HyperPod Observability addon](https://docs.aws.amazon.com/sagemaker/latest/dg/hyperpod-observability-addon-setup.html) by setting `create_observability_module = true`, you can also configure the following metrics that you wish to collect on your cluster: 
| Parameter | Default | Options | Usage |
|-----------|---------|---------|-------|
| `training_metric_level` | `BASIC` | `BASIC, ADVANCED` | Task duration, type, fault data (Advanced: Event-based task performance), [Learn More Here](https://docs.aws.amazon.com/sagemaker/latest/dg/hyperpod-observability-cluster-metrics.html#hyperpod-observability-training-metrics)
| `task_governance_metric_level` | `DISABLED` | `DISABLED, ADVANCED` | Team-level resource allocation, [Learn More Here](https://docs.aws.amazon.com/sagemaker/latest/dg/hyperpod-observability-cluster-metrics.html#hyperpod-observability-task-governance-metrics)
| `scaling_metric_level` | `DISABLED` | `DISABLED, ADVANCED` | KEDA auto-scaling metrics, [Learn More Here](https://docs.aws.amazon.com/sagemaker/latest/dg/hyperpod-observability-cluster-metrics.html#hyperpod-observability-scaling-metrics)
| `cluster_metric_level` | `BASIC` | `BASIC, ADVANCED` | Cluster health, instance count (Advanced: Detailed Kube-state cluster metrics), [Learn More Here](https://docs.aws.amazon.com/sagemaker/latest/dg/hyperpod-observability-cluster-metrics.html#hyperpod-observability-cluster-health-metrics)
| `node_metric_level` | `BASIC` | `BASIC, ADVANCED` | CPU, disk, OS-level usage (Advanced: Full node exporter suite), [Learn More Here](https://docs.aws.amazon.com/sagemaker/latest/dg/hyperpod-observability-cluster-metrics.html#hyperpod-observability-instance-metrics)
| `network_metric_level` | `DISABLED` | `DISABLED, ADVANCED` | Elastic Fabric Adapter metrics, [Learn More Here](https://docs.aws.amazon.com/sagemaker/latest/dg/hyperpod-observability-cluster-metrics.html#hyperpod-observability-network-metrics)
| `accelerated_compute_metric_level` | `BASIC` | `BASIC, ADVANCED` | GPU utilization, temperature (Advanced: All NVIDIA GPU DCGM, Neuron metrics), [Learn More Here](https://docs.aws.amazon.com/sagemaker/latest/dg/hyperpod-observability-cluster-metrics.html#hyperpod-observability-accelerated-compute-metrics)
| `logging_enabled` | `false` | `true, false` | When enabled, this will automatically create the required log groups in Amazon CloudWatch and start recording all container and pod logs as log streams
---
### FSx for Lustre Module

By default, the FSx for Lustre module installs the Amazon FSx for Lustre Container Storage Interface (CSI) Driver, but does not dynamically provision a new filesystem. For existing filesystems, you can follow [these steps in the AI on SageMaker HyperPod Workshop](https://awslabs.github.io/ai-on-sagemaker-hyperpod/docs/getting-started/orchestrated-by-eks/Set%20up%20your%20shared%20file%20system#option-3-bring-your-own-fsx-static-provisioning) for static provisioning. If you wish to create a new filesystem using Terraform, add the parameter `create_new_fsx_filesystem = true` to your `custom.tfvars` file, and review the `fsx_storage_capacity` (default 1200 GiB) and `fsx_throughput` (default 250 MBps/TiB) parameters to ensure they are set according to your requirements. When `create_new_fsx_filesystem = true` the FSx for Lustre module will statically create a new filesystem along with a StorageClass, PersistentVolume, and PersistentVolumeClaim (PVC). By default the PVC will be mapped to the default namespace. If you wish to use another namespace, use the `fsx_pvc_namespace` parameter to specify it. By default, specifying a non-default namespace will trigger the creation of that namespace. If you are using an existing EKS cluster where the target namespace already exists, set `create_fsx_pvc_namespace = false` to skip creation. 

---

## Cilium CNI (Optional)

You can replace the default AWS VPC CNI with [Cilium](https://cilium.io) by setting `enable_cilium = true`. This supports two pre-configured modes plus a fully custom option:

| Mode | Description | VPC CNI |
|------|-------------|---------|
| `overlay` | VXLAN tunnel, non-VPC-routable pod IPs, highest pod density | Removed |
| `chaining` | VPC CNI handles networking, Cilium adds eBPF policy/LB | Kept |
| `custom` | User provides all Helm values, no defaults applied | Removed |

### New EKS Cluster with Cilium

```hcl
enable_cilium  = true
cilium_mode    = "overlay"
cilium_version = "1.19.4"

# Optional: override specific Helm values on top of mode defaults
cilium_helm_values = {
  hubble = {
    enabled = true
  }
}
```

### Existing EKS Cluster with Cilium Already Installed

If you are integrating HyperPod with an existing EKS cluster that already has Cilium running:

```hcl
create_eks_module = false
existing_eks_cluster_name = "my-cilium-cluster"
enable_cilium = true
cilium_mode   = "overlay"  # Match your existing Cilium configuration
```

Setting `enable_cilium = true` with `create_eks_module = false` will:
- Skip Cilium deployment (it's already on your cluster)
- Add appropriate security group rules (e.g., VXLAN UDP 8472 for overlay mode)
- Skip VPC CNI addon creation

### Custom Mode

For full control over the Cilium Helm chart configuration:

```hcl
enable_cilium = true
cilium_mode   = "custom"
cilium_helm_values = {
  routingMode = "tunnel"
  tunnelProtocol = "vxlan"
  ipam = {
    mode = "cluster-pool"
  }
  hubble = {
    enabled = true
    relay = {
      enabled = true
    }
  }
}
```

### Closed-Network Chart Source

By default the Cilium Helm chart is pulled from the public repository `https://helm.cilium.io/`. In closed-network (air-gapped) deployments without internet access, pre-stage the chart to a private mirror and redirect the source with `cilium_helm_repository` / `cilium_helm_chart`. The Helm provider (>= 3.0) supports both classic HTTP repositories and OCI registries such as Amazon ECR:

```hcl
# Classic HTTP mirror
cilium_helm_repository = "https://my-mirror.internal/charts/"
cilium_helm_chart      = "cilium"

# OCI registry (repository + chart name)
cilium_helm_repository = "oci://<account>.dkr.ecr.<region>.amazonaws.com"
cilium_helm_chart      = "cilium"

# OCI registry (fully qualified reference in the chart, empty repository)
cilium_helm_repository = ""
cilium_helm_chart      = "oci://<account>.dkr.ecr.<region>.amazonaws.com/cilium"
```

> **Note:** These variables redirect the **chart** source only. Cilium's **container images** are configured separately through `cilium_helm_values` (e.g. `image.repository`, `operator.image.repository`, `preflight.image.repository`). A fully closed-network deployment must pre-stage both the chart and the images to your private registry.

### Limitations

- **Closed network:** The Cilium Helm chart and container images must be pre-staged to a private registry (e.g. ECR) in closed-network deployments. Redirect the chart with `cilium_helm_repository` / `cilium_helm_chart` (see [Closed-Network Chart Source](#closed-network-chart-source)) and the images via `cilium_helm_values`.
- **ENI mode not supported:** Cilium's native ENI mode is incompatible with HyperPod because SageMaker-managed instances are not visible in the EC2 API. Use `overlay` or `chaining` instead.
- **Overlay mode:** Pod-to-VPC traffic is SNATed. Webhooks must be host-networked or exposed via Service/Ingress.
- **Chaining mode:** Some Cilium features limited (L7 policy, IPsec encryption).

---

### Amazon GuardDuty EKS Runtime Monitoring
If your target account has [Amazon GuardDuty EKS Runtime Monitoring](https://docs.aws.amazon.com/guardduty/latest/ug/runtime-monitoring.html) enabled, an interface VPC endpoint is automatically created to allow the security agent to deliver events to GuardDuty while event data remains within the AWS network. Because this VPC endpoint is not managed by Terraform, the associated Elastic Network Interfaces (ENIs) and Security Group that are automatically deployed by GuardDuty can block destruction when you are ready to clean up. To mitigate this, we've included an optional GuardDuty cleanup script [guardduty-cleanup.sh](./hyperpod-eks-tf/scripts/guardduty-cleanup.sh) that is invoked only at destruction time using a Terraform `null_resource`. This script finds the GuardDuty VPC endpoint associated with your HyperPod VPC and deletes it, waits for the associated ENIs to be cleaned up, then deletes the associated Security Group. To enable this script at plan and apply time, simply add the parameter `enable_guardduty_cleanup = true` to your `custom.tfvars` file. This script won't run when you issue a `terraform apply` command, but will run when you issue a `terraform destroy` command. 

---

### Creating a Restricted Instance Group (RIG) for Nova Model Customization

As a prerequisite, you will need to identify or create input and output S3 buckets to reference in your deployment (represented as `my-tf-rig-test-input-bucket` and `my-tf-rig-test-output-bucket` in the following examples). 

To create new S3 buckets, you can execute commands like the following example using the AWS CLI: 
```bash
aws s3 mb s3://my-tf-rig-test-input-bucket --region us-east-1 # adjust region as needed

aws s3 mb s3://my-tf-rig-test-output-bucket --region us-east-1 # adjust region as needed
```
S3 bucket names must be globally unique. 

You will also need to have [yq](https://pypi.org/project/yq/) installed so that a bash script that modifies CoreDNS and VPC CNI deployments can execute properly. 

For Nova model customization using Restricted Instance Groups (RIG), you can use the example configuration in [`rig_custom.tfvars`](./hyperpod-eks-tf/rig_custom.tfvars). This file demonstrates how to configure restricted instance groups with the necessary S3 buckets and instance specifications.

If you wish to create a new `rig_custom.tfvars` file, you execute a command like the following example with your specific configuration: 

```bash 
cat > rig_custom.tfvars << EOL 
kubernetes_version = "1.32"
eks_cluster_name = "tf-eks-cluster-rig"
hyperpod_cluster_name = "tf-hp-cluster-rig"
resource_name_prefix = "tf-eks-test-rig"
aws_region = "us-east-1"
rig_input_s3_bucket = "my-tf-rig-test-input-bucket"
rig_output_s3_bucket = "my-tf-rig-test-output-bucket"
restricted_instance_groups = [
    {
        name = "rig-1" 
        instance_type = "ml.p5.48xlarge",
        instance_count = 2, 
        availability_zone_id  = "use1-az6"
        ebs_volume_size_in_gb = 850,
        threads_per_core = 2, 
        enable_stress_check = false,
        enable_connectivity_check = false,
        fsxl_per_unit_storage_throughput = 250,
        fsxl_size_in_gi_b = 4800
        training_plan_arn = arn:aws:sagemaker:us-west-2:123456789012:training-plan/training-plan-example
    }
]
EOL
```
RIG mode (`local.rig_mode = true` set in [main.tf](./hyperpod-eks-tf/main.tf)) is automatic when `restricted_instance_groups` are defined, enabling Nova model customization with the following changes: 
- **VPC Endpoints**: Lambda and SQS interface endpoints are added for reinforcement fine-tuning (RFT) with integrations for your custom reward service hosted outside of the RIG. These endpoints are enabled in RIG mode by default so that you can easily transition from continuous pre-training (CPT) or supervised fine-tuning (SFT) to RFT without making infrastructure changes, but they can be disabled by setting `rig_rft_lambda_access` and `rig_rft_sqs_access` to false. 
- **IAM Execution Role Permissions**: The execution role associated with the HyperPod nodes is expanded to include read permission to your input S3 bucket and write permissions to your output S3 bucket. Access to SQS and Lambda resources with ARN patterns `arn:aws:lambda:*:*:function:*SageMaker*` and `arn:aws:sqs:*:*:*SageMaker*` are also conditionally added if `rig_rft_lambda_access` and `rig_rft_sqs_access` are true (default). 
- **Helm Charts**: A specific Helm revision is checked out and used for RIG support. After Helm chart instillation, a bash script is used to modify CoreDNS and VPC NCI deployments (be sure to have [yq](https://pypi.org/project/yq/) installed for this). 
- **HyperPod Cluster**: Continuous provisioning mode and Karpenter autoscaling are disabled automatically for RIG compatibility. Deploying a HyperPod cluster with a combination of standard instance groups and RIGs is also not currently supported, so `instance_groups` definitions are ignored when `restricted_instance_groups` are defined.
- **FSx for Lustre**: For RIGs a service managed FSx for Lustre filesystem is created based on the specifications you provide in `fsxl_per_unit_storage_throughput` and `fsxl_size_in_gi_b`. 
    - Valid values for `fsxl_per_unit_storage_throughput` are 125, 250, 500, or 1000 MBps/TiB. 
    - Valid values for `fsxl_size_in_gi_b` start at 1200 GiB and go up in increments of 2400 GiB. 
- **S3 Lifecycle Scripts**: Because RIGs do not leverage lifecycle scripts, the `s3_bucket` and `lifecycle_script` modules are also disabled in RIG mode. 

Please note that the following addons are NOT currently supported on HyperPod with RIGs: 
- HyperPod Task Governance 
- HyperPod Observability
- HyperPod Training Operator
- HyperPod Inference Operator

Do not attempt to install these addons later using the console. 

Once you have your `rig_custom.tfvars` file is created, you can proceed to deployment. 

---
