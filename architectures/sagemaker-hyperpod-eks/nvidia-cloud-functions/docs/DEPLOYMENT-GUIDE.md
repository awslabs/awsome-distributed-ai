# Deployment Guide: NVIDIA Cloud Functions on an Existing SageMaker HyperPod EKS Cluster

This guide walks through installing NVIDIA Cloud Functions (NVCF) on an
**existing** SageMaker HyperPod EKS cluster using the BYOC model, from
cluster discovery through function invocation.

**Time estimate**: 30-60 minutes (depending on GPU node scale-up time)

**Prerequisites**:
- An existing SageMaker HyperPod EKS cluster in `InService` state
- AWS CLI configured with permissions for EKS, SageMaker, and EC2
- NVIDIA NGC account with Cloud Functions access
- NGC Personal API Key with Cloud Functions + Private Registry scopes
- Cloud Functions Admin role in your NGC organization

## Overview of Steps

| Step | Script | What It Does |
|------|--------|--------------|
| -- | `nvcf-config.env` | Centralised configuration (fill in once) |
| 0 | `00-discover-cluster.sh` | Discover and validate existing SageMaker HyperPod cluster |
| 1 | `01-prepare-cluster.sh` | Check tools, configure kubeconfig, scale GPU nodes |
| 2 | `02-install-gpu-operator.sh` | Install NVIDIA GPU Operator (driver-skip mode) |
| 3 | `03-register-nvca.sh` | Register cluster with NVIDIA Cloud Functions |
| 4 | `04-validate-setup.sh` | Validate the full setup |
| 5 | `nvcf/sample-function/deploy.sh` | Deploy a sample echo function |

---

## Configuration: `nvcf-config.env`

All user-supplied values live in a single file at the repository root.
Every script auto-sources this file when it exists, so you never need to
export variables manually.

```bash
# 1. Create your config from the template
cp nvcf-config.env.template nvcf-config.env

# 2. Open it and fill in your values
vi nvcf-config.env

# 3. (Optional) Source it in your current shell so you can run ad-hoc commands
source nvcf-config.env
```

The template has detailed comments for every variable. At a minimum you need
to fill in:

| Variable | When Needed | Where to Get It |
|----------|-------------|-----------------|
| `AWS_REGION` | Step 0+ | Your AWS region (e.g., `us-west-2`) |
| `HYPERPOD_CLUSTER_NAME` | Step 0 | `aws sagemaker list-clusters --region <region>` |
| `NGC_CLUSTER_KEY` | Step 3 | NVCF UI > Register Cluster > Helm command |
| `NVCA_HELM_URL` | Step 3 | Same Helm command |
| `NVCA_NCA_ID` | Step 3 | Same Helm command (`--set ncaID`) |
| `NVCA_CLUSTER_ID` | Step 3 | Same Helm command (`--set clusterID`) |
| `NGC_API_KEY` | Step 5 | https://org.ngc.nvidia.com/setup/personal-keys |
| `NGC_ORG_NAME` | Step 5 | https://org.ngc.nvidia.com/profile |
| `CLUSTER_GROUP` | Step 5 | NVCF UI > Clusters (backend name) |
| `GPU_TYPE` | Step 5 | NVCF cluster group API (see template) |
| `INSTANCE_TYPE` | Step 5 | NVCF cluster group API (see template) |

> **Tip**: You can fill in the Step 0 variables first, run the discovery
> script, then come back and fill in the remaining variables as you progress
> through the steps.

> **Security**: `nvcf-config.env` is in `.gitignore` so your secrets are
> never committed. Only the `.template` file is tracked in git.

---

## Step 0: Discover Your Existing Cluster

```bash
./infra/scripts/00-discover-cluster.sh
```

The script reads `HYPERPOD_CLUSTER_NAME` from `nvcf-config.env`. You can also
override it on the command line:
```bash
./infra/scripts/00-discover-cluster.sh --cluster-name my-hyperpod-cluster
```

> Reads `AWS_REGION`, `AWS_PROFILE`, and `HYPERPOD_CLUSTER_NAME` from
> `nvcf-config.env`.

**What this does**:
- Lists all SageMaker HyperPod clusters in the region
- Auto-selects the first `InService` cluster (or uses the one you specified)
- Shows instance groups, GPU quotas, and networking details
- Checks for NAT Gateway (required for NVCF outbound connectivity)
- Warns if Kubernetes version is above NVCF's documented max (v1.32.x)
- Saves cluster config to `infra/.cluster-config` for subsequent scripts

**What to check in the output**:
- Cluster status is `InService`
- At least one GPU instance group exists (e.g., `ml.g5.8xlarge`, `ml.g5.2xlarge`)
- NAT Gateway is present in the VPC
- Kubernetes version is noted (v1.33 is above NVCF docs but may work)

---

## Step 1: Prepare the Cluster

```bash
./infra/scripts/01-prepare-cluster.sh \
    --instance-group accelerated-worker-group-1 \
    --target-count 1
```

**What this does**:
1. Loads cluster config from Step 0
2. Checks that CLI tools are installed (aws, kubectl, helm, docker, ngc)
3. Configures `kubeconfig` for the EKS cluster
4. Scales up the specified GPU instance group (if currently at 0)
5. Verifies outbound network connectivity to NVIDIA endpoints

> **EKS access entry**: If `kubectl` fails to connect after kubeconfig is
> configured, your IAM identity may not have an EKS access entry. Create one:
> ```bash
> # Find your IAM role ARN
> aws sts get-caller-identity --query 'Arn'
>
> # Create access entry (use the role ARN, not the assumed-role ARN)
> aws eks create-access-entry \
>     --cluster-name "${EKS_CLUSTER_NAME}" \
>     --principal-arn <your-iam-role-arn> \
>     --type STANDARD --region "${AWS_REGION}"
>
> # Associate cluster-admin policy
> aws eks associate-access-policy \
>     --cluster-name "${EKS_CLUSTER_NAME}" \
>     --principal-arn <your-iam-role-arn> \
>     --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy \
>     --access-scope type=cluster --region "${AWS_REGION}"
> ```
> (Both `EKS_CLUSTER_NAME` and `AWS_REGION` are available after sourcing
> `nvcf-config.env` and running Step 0.)

> **Instance type guidance**: Use `ml.g5.2xlarge` (8 vCPU, 32 GiB) or
> larger to meet NVCA's 6 CPU + 8 GiB overhead per GPU node. The
> `ml.g5.xlarge` (4 vCPU, 16 GiB) is too tight. If your cluster has
> `ml.g5.8xlarge` (32 vCPU, 128 GiB), that works well too.

**Wait**: If GPU nodes are being scaled up, they take 5-15 minutes to join
the EKS cluster. Monitor with:

```bash
kubectl get nodes -w
```

---

## Step 2: Install NVIDIA GPU Operator

```bash
./infra/scripts/02-install-gpu-operator.sh
```

This installs the GPU Operator with **driver installation disabled** since
SageMaker HyperPod nodes have pre-installed NVIDIA drivers. It enables:
- Device Plugin (GPU scheduling in Kubernetes)
- DCGM Exporter (GPU metrics)
- GPU Feature Discovery (required by NVCA Dynamic GPU Discovery)
- Node Feature Discovery (hardware labels)

**Environment variables** (optional):
- `GPU_OPERATOR_VERSION`: Helm chart version (default: `v24.9.2`)

**Verify**:
```bash
kubectl get pods -n gpu-operator
# All pods should be Running

kubectl get nodes -l nvidia.com/gpu.present=true
# Should show GPU nodes with labels
```

---

## Step 3: Register with NVIDIA Cloud Functions

### 3a. Register the cluster in the NVCF UI

1. Go to https://nvcf.ngc.nvidia.com
2. Navigate to **Settings** > **Register Cluster**
3. Fill in:
   - **Cluster Name**: a name for your cluster (e.g., `hyperpod-nvcf`)
   - **Cluster Group**: a group name (e.g., `hyperpod-nvcf`)
   - **Compute Platform**: `AWS`
   - **Region**: your AWS region (the same `AWS_REGION` from Step 0)
4. In **Cluster Features**:
   - Enable: **Dynamic GPU Discovery** (should be on by default)
   - **Disable**: **Caching Support** (not supported for AWS EKS)
   - **Disable**: **Collect Function Logs** (not supported for BYOC)
5. Click **Save and Continue**
6. The NVCF UI will show a Helm install command. Copy **all four values** from it:
   - **NGC Cluster Key** (`NGC_CLUSTER_KEY`) -- the `--password` value
   - **Helm chart URL** (`NVCA_HELM_URL`) -- the chart URL
   - **NCA ID** (`NVCA_NCA_ID`) -- the `--set ncaID` value
   - **Cluster ID** (`NVCA_CLUSTER_ID`) -- the `--set clusterID` value

### 3b. Install the NVCA Operator

After filling in the four NVCA values in `nvcf-config.env` (from Step 3a):

```bash
./infra/scripts/03-register-nvca.sh
```

**What this does**:
1. Validates environment variables
2. Checks Kubernetes connectivity, GPU Operator presence, and cluster-admin RBAC
3. Installs the NVCA Operator via Helm
4. Labels GPU nodes for NVCF scheduling
5. Shows post-install notes on caching, network policies, logs, and autoscaling

**Wait**: The NVCA operator deploys the agent. This takes 2-5 minutes.

**Verify**:
```bash
kubectl get nvcfbackend -n nvca-operator
# NAME                    AGE   VERSION   HEALTH
# hyperpod-nvcf           1m    2.50.0    healthy
```

Also check the NVCF UI: **Settings** > **Clusters** -- the cluster should show as **Ready**.

---

## Step 4: Validate the Full Setup

```bash
./infra/scripts/04-validate-setup.sh
```

This checks:
- Kubernetes connectivity and version (warns if > 1.32)
- GPU Operator installation and GPU discovery
- NVCA operator and backend health
- SageMaker HyperPod node resources (CPU/memory vs. NVCA overhead)
- VPC CNI and NetworkPolicy status

---

## Step 5: Deploy a Sample Function

### 5a. Configure container runtime and NGC Registry access

The deploy script supports both Docker and [Finch](https://runfinch.com/) as
container runtimes.  It auto-detects which one is available (prefers Docker,
falls back to Finch).

```bash
# Log in to NGC Private Registry (choose your runtime)
# Docker:
docker login nvcr.io -u '$oauthtoken' -p <your-ngc-api-key>

# Finch:
echo "<your-ngc-api-key>" | finch login nvcr.io -u '$oauthtoken' --password-stdin
```

### 5b. Build, push, create, and deploy the echo function

Ensure you have filled in the Step 5 variables in `nvcf-config.env`
(`NGC_API_KEY`, `NGC_ORG_NAME`, `CLUSTER_GROUP`, `GPU_TYPE`, `INSTANCE_TYPE`).

```bash
./nvcf/sample-function/deploy.sh
```

> **`GPU_TYPE` and `INSTANCE_TYPE` values**: These must match what the NVCA
> operator registered for your cluster. Common values for SageMaker HyperPod:
>
> | SageMaker Instance | GPU | `GPU_TYPE` | `INSTANCE_TYPE` (1 GPU) |
> |--------------------|-----|------------|-------------------------|
> | ml.g5.xlarge-8xlarge | NVIDIA A10G | `A10G` | `AWS.GPU.A10G_1x` |
> | ml.g6.xlarge-8xlarge | NVIDIA L4 | `L4` | `AWS.GPU.L4_1x` |
> | ml.g6e.xlarge-8xlarge | NVIDIA L40S | `L40S` | `AWS.GPU.L40S_1x` |
> | ml.p4d.24xlarge | NVIDIA A100 80GB | `A100` | `AWS.GPU.A100_1x` |
> | ml.p5.48xlarge | NVIDIA H100 80GB | `H100` | `AWS.GPU.H100_1x` |
>
> The `_1x` suffix indicates 1 GPU per instance. For multi-GPU deployments,
> use `_2x`, `_4x`, `_8x`, etc. To see the exact values registered for your
> cluster, query the NVCF API:
> ```bash
> curl -s https://api.ngc.nvidia.com/v2/nvcf/clusterGroups \
>   -H "Authorization: Bearer ${NGC_API_KEY}" | python3 -m json.tool
> ```
> Look for your cluster group name and note the `gpus[].name` and
> `gpus[].instanceTypes[].name` values.

This will:
1. Build the container image for `linux/amd64` (Docker or Finch)
2. Push it to NGC Private Registry at `nvcr.io/${NGC_ORG_NAME}/nvcf-echo:1.0.0`
3. Create a function in NVCF with `/echo` inference and `/health` health endpoints
4. Deploy it to the specified cluster group

### 5c. Invoke the function

```bash
FUNCTION_ID="<function-id-from-deploy-output>"  # printed by deploy.sh after function creation

curl --location "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/${FUNCTION_ID}" \
    --header 'Content-Type: application/json' \
    --header "Authorization: Bearer ${NGC_API_KEY}" \
    --data '{"message": "hello from SageMaker HyperPod!"}'
```

Expected response:
```json
{"echo": "hello from SageMaker HyperPod!"}
```

Streaming test:
```bash
curl --location "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/${FUNCTION_ID}" \
    --header 'Content-Type: application/json' \
    --header "Authorization: Bearer ${NGC_API_KEY}" \
    --data '{"message": "streaming test", "stream": true, "repeats": 3}'
```

---

## Cleanup

To tear down the NVCF components (this does **not** destroy your SageMaker HyperPod cluster):

```bash
# 1. Undeploy and delete the function in NVCF (via UI or API)

# 2. Deregister the cluster from NVCF
kubectl delete nvcfbackends -A --all
kubectl delete ns nvca-system
helm uninstall -n nvca-operator nvca-operator
kubectl delete ns nvca-operator

# 3. Remove GPU Operator
helm uninstall gpu-operator -n gpu-operator
kubectl delete ns gpu-operator

# 4. (Optional) Scale GPU instance group back to 0
aws sagemaker update-cluster \
    --cluster-name <your-hyperpod-cluster-name> \
    --instance-groups '[...]' \
    --region ${AWS_REGION}
```

> **Note**: Scaling GPU instance groups to 0 stops the associated compute
> costs while keeping the SageMaker HyperPod cluster intact.
