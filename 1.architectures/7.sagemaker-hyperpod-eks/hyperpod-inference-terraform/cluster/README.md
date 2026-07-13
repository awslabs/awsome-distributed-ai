# Stage 1 — Create a HyperPod EKS Cluster From Scratch

This stack provisions a complete SageMaker HyperPod environment so you can go
from an empty account to a GPU cluster ready to host the HyperPod **inference
operator** — the prerequisite for the endpoint deployment in the parent
directory.

It is a thin orchestration layer: every piece of infrastructure is built from
the canonical leaf modules in
[`../../terraform-modules/hyperpod-eks-tf/modules`](../../terraform-modules/hyperpod-eks-tf/modules)
(referenced via relative `source` paths), so the underlying resources stay in
lockstep with the upstream reference stack.

## What it creates

| Module | Purpose |
|--------|---------|
| `vpc` | VPC with public subnets, internet gateway, and a NAT gateway |
| `private_subnet` | One large `/16` private subnet per AZ (NAT-routed) for the GPU nodes |
| `security_group` | Cluster security group (intra-SG traffic, FSx ports) |
| `eks_cluster` | EKS control plane + core add-ons (vpc-cni, kube-proxy, pod-identity, coredns) and its own `/28` control-plane subnets |
| `s3_bucket` | Bucket for lifecycle scripts + access-logs bucket |
| `lifecycle_script` | Uploads `on_create.sh` / `on_create_main.sh` to S3 |
| `sagemaker_iam_role` | Execution role for the HyperPod nodes |
| `vpc_endpoints` | S3 gateway endpoint |
| `null_resource.helm_repo` | Idempotently clones the HyperPod dependencies Helm chart repo to `/tmp/helm-repo` (runs before `helm_chart`) |
| `helm_chart` | HyperPod dependencies chart (device plugins, training operators, health monitoring) |
| `hyperpod_cluster` | The SageMaker HyperPod cluster + GPU instance group + cert-manager |
| `fsx_lustre` | Installs the FSx Lustre CSI driver (required by the inference operator's init container) |
| `hyperpod_inference_operator` | The inference operator EKS add-on + ALB controller, KEDA, S3 CSI, metrics-server, and all IRSA roles |
| `subnet_tags.tf` | Tags the HyperPod private subnets for internal-ALB discovery in the GPU AZ (see below) |

By default it provisions **1× `ml.p5.48xlarge` in `use1-az6` (us-east-1c)**.

## Prerequisites

- Terraform/OpenTofu >= 1.5, AWS CLI, `kubectl`, `helm`, and `git` on PATH
- GPU capacity in the target AZ. On-Demand `p5.48xlarge` usually requires an
  On-Demand Capacity Reservation (ODCR) or a SageMaker training plan — set
  `training_plan_arn` on the instance group if you have one.

> The HyperPod dependencies Helm chart repo is **cloned automatically** to
> `/tmp/helm-repo` by `null_resource.helm_repo` (idempotent: it clones if
> missing, otherwise fetches and checks out the pinned `helm_repo_revision`).
> Override `helm_repo_url` / `helm_repo_local_path` / `helm_repo_revision` if
> needed. No manual `git clone` step is required.

## Usage

```bash
cp terraform.tfvars.example terraform.tfvars   # edit region / instance group / AZ
tofu init
tofu plan -out=tfplan
tofu apply tfplan                               # ~30-60 min

# Emit the exact inputs for the endpoint stage:
tofu output -raw endpoint_stage_tfvars
```

> **If the inference operator add-on times out:** the
> `amazon-sagemaker-hyperpod-inference` add-on only reports `ACTIVE` once its
> controller-manager pod is healthy (it waits for a GPU node and several large
> image pulls). The create timeout defaults to **40m**
> (`inference_operator_create_timeout`). If a slow P5 bring-up still exceeds
> it, **re-run `tofu apply`** — it is idempotent and resumes.

## Why `subnet_tags.tf` exists

The AWS Load Balancer Controller discovers subnets for the operator's internal
ALB via the `kubernetes.io/role/internal-elb=1` tag. By default only the small
`/28` EKS control-plane subnets carry that tag, and they live in the first two
AZs. If the GPU instance group runs in a different AZ (e.g. `use1-az6`), the
ALB lands in the wrong AZs and the inference pod targets are stuck `unused`
(`Target.NotInUse`).

`subnet_tags.tf` fixes this declaratively: it tags the large HyperPod private
subnets (one per AZ, including the GPU AZ) for internal-ELB discovery and
removes the tag from the `/28` EKS subnets, leaving exactly one eligible subnet
per AZ so the ALB is placed in the same AZ as the GPU pods.

## Cleanup

```bash
tofu destroy
```
