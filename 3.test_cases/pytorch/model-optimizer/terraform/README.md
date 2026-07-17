# Terraform — Single GPU EC2 Instance for ModelOpt

Provisions a single GPU EC2 instance for evaluating NVIDIA Model Optimizer, connected via
**SSM Session Manager** (no inbound SSH, no key pair). Creates:

- An IAM role + instance profile with `AmazonSSMManagedInstanceCore`
- An **egress-only** security group (SSM needs no inbound)
- One GPU instance from the GPU Deep Learning AMI (resolved via SSM Parameter Store), with a
  200 GiB gp3 root volume and IMDSv2 enforced, in your default VPC

## Prerequisites

- Terraform `>= 1.14.0`
- AWS credentials with permission to create EC2/IAM/SG resources
- The [Session Manager plugin][ssm-plugin] for `aws ssm start-session`
- Sufficient EC2 quota for the chosen instance family (G or P on-demand vCPUs)

[ssm-plugin]: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html

## Usage

```bash
terraform init
terraform plan
terraform apply

# Connect (use the emitted output):
$(terraform output -raw ssm_start_session_command)
```

Verify the GPU once connected:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Then follow the [recipe README](../README.md) to install ModelOpt and run FP8 quantization.

## Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `aws_region` | `us-west-2` | Region to deploy into |
| `instance_type` | `g6e.xlarge` | 1x L40S (Ada), FP8/INT8. Override to `p5` (H100) or `p6-b200` (NVFP4) |
| `root_volume_size_gb` | `200` | Root gp3 volume |
| `project_tag` | `modelopt-runbook` | Applied to all resources via `default_tags` |
| `name_prefix` | `modelopt-runbook` | Resource name prefix |
| `dlami_ssm_parameter` | base OSS NVIDIA-driver AL2023 | SSM path to the AMI ID |

## Teardown (cost-critical)

```bash
terraform destroy
```

This terminates the instance (root volume has `delete_on_termination = true`) and removes the
security group and IAM role/instance profile. Confirm `destroy` completes with no errors.

> [!WARNING]
> A `g6e.xlarge` is ~$1.86/hr on-demand in us-west-2. Run `terraform destroy` as soon as you
> are done — the instance bills until destroyed.
