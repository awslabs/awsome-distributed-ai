# HyperPod Inference: Public Endpoint with Terraform

Deploy a Hugging Face model on SageMaker HyperPod with a publicly accessible HTTPS endpoint — fully managed by Terraform/OpenTofu. No AWS credentials required for callers.

This project is organized into **two stages** so you can start from absolutely nothing (no VPC, no EKS, no HyperPod cluster) and end with a public, authenticated inference endpoint:

| Stage | Directory | What it creates |
|-------|-----------|-----------------|
| **Stage 1 — Cluster** | [`cluster/`](./cluster) | A complete HyperPod environment from scratch: VPC, private subnets, security group, EKS cluster, S3 bucket + lifecycle scripts, SageMaker execution role, the HyperPod cluster (GPU instance group), and the **HyperPod inference operator** add-on (with its ALB controller, KEDA, S3 CSI driver, cert-manager, and metrics-server dependencies). |
| **Stage 2 — Endpoint** | `.` (this directory) | The model deployment: ACM certificate, namespace, HuggingFace token secret, `InferenceEndpointConfig`, and the public API Gateway (auth + rate limiting + custom domain). |

If you **already have** a HyperPod EKS cluster with the inference operator installed, skip Stage 1 and go straight to [Stage 2](#stage-2-phase-1-deploy-the-inference-endpoint).

The Stage 1 code reuses the canonical leaf modules from the sibling
[`terraform-modules/hyperpod-eks-tf`](../terraform-modules/hyperpod-eks-tf)
stack (via relative `source` paths) so the underlying infrastructure stays in
lockstep with the upstream reference.



## Architecture

```
Internet Client (curl, Python, OpenAI SDK)
    │
    │  HTTPS (TLSv1.3, HTTP/2, port 443)
    │  Header: x-api-key: sk-hyperpod-...
    ▼
┌─────────────────────────────────────────────────┐
│  API Gateway (HTTP API)                         │  ← Terraform-managed
│  Custom domain: inference-htx.sa.do.wwso.aws.dev│
│  ├── Lambda Authorizer (validates x-api-key)    │
│  ├── Rate limiting: 10 req/s, burst 20         │
│  └── CloudWatch metrics + access logs           │
└─────────────────────┬───────────────────────────┘
                      │ VPC Link (private connectivity)
                      ▼
┌─────────────────────────────────────────────────┐
│  Internal ALB (TLS with custom ACM cert)        │  ← Operator-managed
│  Routes: /v1/chat/completions → pods            │
└─────────────────────┬───────────────────────────┘
                      │ HTTP → port 8081 (nginx sidecar)
                      ▼
┌─────────────────────────────────────────────────┐
│  vLLM Pod (Qwen2.5-32B-Instruct, TP=2)         │  ← Operator-managed
│  2x H100 GPUs, 120Gi RAM                       │
│  + nginx sidecar + otel collector               │
└─────────────────────────────────────────────────┘

Route 53:→ API Gateway custom domain
```

### How it works (end-to-end request flow)

When a client sends a request to your custom domain:

1. **DNS Resolution** — Route 53 resolves `inference.yourdomain.com` to the API Gateway regional endpoint.

2. **TLS Termination at API Gateway** — API Gateway terminates TLS using your ACM certificate (TLS 1.3, HTTP/2). The client sees a valid, publicly trusted Amazon-issued certificate.

3. **Authentication** — API Gateway invokes a Lambda authorizer that validates the `x-api-key` header (or `Authorization: Bearer` token). Invalid or missing keys are rejected immediately with 401/403 — the request never reaches the model.

4. **Rate Limiting** — API Gateway enforces configurable throttling (e.g., 10 requests/second sustained, 20 burst). Excess requests receive 429 Too Many Requests.

5. **VPC Link** — Authenticated requests are forwarded through a VPC Link, which provides private connectivity from API Gateway to the internal ALB without exposing it to the internet. The VPC Link creates ENIs in your private subnets.

6. **Internal ALB (Operator-managed)** — The HyperPod inference operator manages this ALB. It has a second ACM certificate attached (same domain) and routes requests by path (`/v1/chat/completions`) to the correct target group. API Gateway validates this cert via TLS ServerNameToVerify.

7. **Nginx Sidecar** — Each inference pod has an nginx reverse proxy that handles request buffering, connection management, and exposes Prometheus metrics for the OpenTelemetry collector.

8. **vLLM Inference** — The request reaches the vLLM server, which processes it using the loaded model weights across the allocated GPUs (tensor parallelism). The response streams back through the same path.

9. **Response** — The JSON response (OpenAI-compatible format) returns through ALB → VPC Link → API Gateway → client. Total added latency from API Gateway + Lambda authorizer is typically <10ms.

**Key security properties:**
- The internal ALB is never directly accessible from the internet
- GPU inference pods run in private subnets with no public IPs
- All traffic is encrypted (TLS 1.3 client→APIGW, TLS 1.2 APIGW→ALB)
- API keys are stored in Lambda environment variables (encrypted at rest)
- The HuggingFace token is stored as a Kubernetes secret (encrypted by EKS envelope encryption)

## Prerequisites

1. **Terraform/OpenTofu** >= 1.5
2. **AWS CLI** configured with admin credentials
3. **kubectl**, **helm**, and **git** installed (Stage 1 uses helm + git to bootstrap the dependencies chart)
4. **A public Route 53 hosted zone** for your domain (Stage 2)
5. **HuggingFace token** (for gated models or higher download rate limits)
6. **GPU capacity** — for `ml.p5.48xlarge` On-Demand you typically need an
   On-Demand Capacity Reservation (ODCR) or a SageMaker training plan in the
   target AZ. Check your `... for cluster usage` quota:
   ```bash
   aws service-quotas list-service-quotas --service-code sagemaker \
     --query 'Quotas[?contains(QuotaName,`p5.48xlarge for cluster usage`)].[QuotaName,Value]' \
     --output table --region us-east-1
   ```
7. **A SageMaker HyperPod EKS cluster in service** — *only if you skip Stage 1*.
   Stage 1 creates this for you.

## File Structure

```
hyperpod-inference-terraform/
├── cluster/                    # STAGE 1 — create the cluster from scratch
│   ├── providers.tf            #   AWS + awscc + kubernetes + helm (exec auth)
│   ├── variables.tf            #   region, instance_groups (P5 in use1-az6), etc.
│   ├── main.tf                 #   wires the reference leaf modules together
│   ├── outputs.tf              #   emits the exact tfvars for Stage 2
│   └── terraform.tfvars.example
│
├── providers.tf            # STAGE 2 — AWS + Kubernetes + Archive providers
├── variables.tf            # All configurable inputs
├── terraform.tfvars.example # Template — copy to terraform.tfvars and fill in
├── .gitignore              # Prevents committing secrets and state
├── acm.tf                  # ACM certificate + S3 bucket for TLS
├── iam.tf                  # Operator IAM permissions (ACM, Route53, S3)
├── apigw.tf                # API Gateway + VPC Link + Lambda authorizer + Route 53
├── inference.tf            # Namespace, HF token secret, InferenceEndpointConfig
└── outputs.tf              # Endpoint URL, API Gateway URL, next steps
```

---

## Stage 1: Create the HyperPod Cluster From Scratch

> Skip this stage if you already have a HyperPod EKS cluster with the inference
> operator installed.

Stage 1 lives in [`cluster/`](./cluster). It builds the full environment and,
by default, provisions **1× `ml.p5.48xlarge` in `use1-az6` (us-east-1c)** as the
inference instance group.

```bash
cd cluster

# 1. Configure the cluster
cp terraform.tfvars.example terraform.tfvars
```

### Configure region, AZ, instance type, and instance count

Open `terraform.tfvars` and set the following. These four values control
*where* and *what* GPU capacity the HyperPod cluster provisions.

```hcl
# Region the whole cluster is deployed into.
region = "us-east-1"

instance_groups = [
  {
    name                      = "inference-p5"
    instance_type             = "ml.p5.48xlarge"   # GPU instance type
    instance_count            = 1                  # number of nodes
    availability_zone_id      = "use1-az6"         # AZ *zone-id* (not "us-east-1c")
    ebs_volume_size_in_gb     = 500
    threads_per_core          = 2
    enable_stress_check       = false
    enable_connectivity_check = false
    lifecycle_script          = "on_create.sh"
    # training_plan_arn       = "arn:aws:sagemaker:...:training-plan/NAME"  # for reserved capacity
  }
]
```

| Field | What to set it to | Notes |
|-------|-------------------|-------|
| `region` | Your AWS region, e.g. `us-east-1`, `us-west-2` | Must match where you have GPU quota/capacity. |
| `instance_type` | A HyperPod GPU type, e.g. `ml.p5.48xlarge`, `ml.p4d.24xlarge`, `ml.g5.12xlarge` | Must match `instance_type` in the **Stage 2** `terraform.tfvars`. |
| `instance_count` | Number of nodes, e.g. `1`, `2` | Must be ≤ your `... for cluster usage` quota for that type. |
| `availability_zone_id` | The AZ **zone-id** (e.g. `use1-az6`), **not** the AZ name (`us-east-1c`) | This is where the GPU nodes — and therefore the inference pods/ALB — land. |

**Important:** `availability_zone_id` uses the *zone-id* format, which is stable
across accounts (e.g. `use1-az6`), not the friendly name (`us-east-1c`) which
maps to a different physical AZ per account. Look up the mapping for your region:

```bash
aws ec2 describe-availability-zones --region us-east-1 \
  --query 'AvailabilityZones[].[ZoneName,ZoneId]' --output table
# Pick the ZoneId of the AZ where your P5/GPU capacity (or ODCR) lives.
```

Verify you have quota for the type and count you chose:

```bash
aws service-quotas list-service-quotas --service-code sagemaker --region us-east-1 \
  --query "Quotas[?contains(QuotaName,'p5.48xlarge for cluster usage')].[QuotaName,Value]" \
  --output table
```

> On-Demand `ml.p5.48xlarge` typically requires an On-Demand Capacity
> Reservation (ODCR) or a SageMaker training plan in the chosen AZ. If you have
> a training plan, set `training_plan_arn` on the instance group.

### Deploy

```bash
# Deploy (VPC + EKS + HyperPod GPU nodes + inference operator).
# The HyperPod dependencies Helm chart repo is cloned automatically to
# /tmp/helm-repo by the stack (null_resource.helm_repo) — no manual clone
# needed. Requires `git` and `helm` on PATH.
tofu init
tofu plan -out=tfplan
tofu apply tfplan        # ~30-60 min (EKS + node provisioning + operator addon)
```

> **If the operator add-on times out:** the
> `amazon-sagemaker-hyperpod-inference` EKS add-on only reports `ACTIVE` once
> its controller-manager is healthy, which waits on a GPU node being ready and
> several large images being pulled. The create timeout defaults to **40m**
> (`inference_operator_create_timeout`). On unusually slow P5 bring-up it can
> still time out — if so, **just re-run `tofu apply`**; the operation is
> idempotent and resumes where it left off.

When it finishes, copy the generated Stage 2 inputs straight out of the outputs:

```bash
tofu output -raw endpoint_stage_tfvars
# Paste the result into ../terraform.tfvars (then add domain_name,
# hosted_zone_id, hf_token, api_keys, and model settings).
```

Verify the cluster and operator are healthy:

```bash
EKS_CLUSTER=$(tofu output -raw eks_cluster_name)
aws eks update-kubeconfig --name "$EKS_CLUSTER" --region $(tofu output -raw region)
kubectl get nodes
kubectl get pods -n hyperpod-inference-system   # controller-manager should be 1/1 Running
cd ..
```

---

## Step 0: EKS Access for Your Identity

> If you ran Stage 1, the EKS cluster was created with your identity as a
> cluster admin (`bootstrap_cluster_creator_admin_permissions = true`) and
> `kubectl` already works — you can skip this step. The steps below are for
> attaching access to a **pre-existing** cluster.

Before `kubectl` works, your IAM identity needs access to the EKS cluster.

```bash
# Identify who you are
aws sts get-caller-identity

# Get the EKS cluster name from the HyperPod cluster
EKS_CLUSTER=$(aws sagemaker describe-cluster \
  --cluster-name <hyperpod-cluster-name-or-arn> \
  --region <region> \
  --query 'Orchestrator.Eks.ClusterArn' --output text | cut -d'/' -f2)

# Add EKS permissions to your identity
# For IAM Role:
aws iam put-role-policy --role-name <your-role> --policy-name EKSAccess --policy-document '{
  "Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["eks:AccessKubernetesApi","eks:DescribeCluster"],"Resource":"*"}]}'

# For IAM User:
aws iam put-user-policy --user-name <your-user> --policy-name EKSAccess --policy-document '{
  "Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["eks:AccessKubernetesApi","eks:DescribeCluster"],"Resource":"*"}]}'

# Create EKS access entry (use your role or user ARN)
aws eks create-access-entry \
  --cluster-name $EKS_CLUSTER \
  --principal-arn <your-iam-arn> \
  --type STANDARD --region <region>

aws eks associate-access-policy \
  --cluster-name $EKS_CLUSTER \
  --principal-arn <your-iam-arn> \
  --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy \
  --access-scope type=cluster --region <region>

# Configure kubectl
aws eks update-kubeconfig --name $EKS_CLUSTER --region <region>

# Verify
kubectl get nodes
```

---

## Step 1: One-Time Cluster Setup (Inference Operator)

> **Stage 1 users:** the inference operator (and its ALB controller, KEDA, S3
> CSI driver, cert-manager, and metrics-server dependencies) is already
> installed by the `cluster/` stack — skip this step.

Skip this if the inference operator is already installed (`kubectl get pods -n hyperpod-inference-system` shows running pods).

```bash
REGION="<your-region>"                        # e.g., us-east-2
EKS_CLUSTER="<your-eks-cluster-name>"         # e.g., sagemaker-mycluster-abc123-eks
CLUSTER_SHORT="<short-name>"                  # e.g., mycluster (used in role names)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# 1. Get the OIDC provider ID
OIDC_ID=$(aws eks describe-cluster --name $EKS_CLUSTER --region $REGION \
  --query 'cluster.identity.oidc.issuer' --output text | rev | cut -d'/' -f1 | rev)

# 2. Create trust policy for IRSA
cat > /tmp/trust.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "oidc.eks.${REGION}.amazonaws.com/id/${OIDC_ID}:aud": "sts.amazonaws.com"
      }
    }
  }]
}
EOF

# 3. Create IRSA role for inference operator
aws iam create-role --role-name SageMakerHyperPodInference-${CLUSTER_SHORT} \
  --assume-role-policy-document file:///tmp/trust.json
aws iam attach-role-policy --role-name SageMakerHyperPodInference-${CLUSTER_SHORT} \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerHyperPodInferenceAccess

# 4. Create IRSA role for ALB controller
aws iam create-role --role-name HyperPodALBController-${CLUSTER_SHORT} \
  --assume-role-policy-document file:///tmp/trust.json
aws iam attach-role-policy --role-name HyperPodALBController-${CLUSTER_SHORT} \
  --policy-arn arn:aws:iam::aws:policy/ElasticLoadBalancingFullAccess
aws iam attach-role-policy --role-name HyperPodALBController-${CLUSTER_SHORT} \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess
aws iam attach-role-policy --role-name HyperPodALBController-${CLUSTER_SHORT} \
  --policy-arn arn:aws:iam::aws:policy/AmazonVPCFullAccess

# 5. Create S3 bucket for TLS certificates
aws s3 mb s3://hyperpod-tls-${CLUSTER_SHORT}-${ACCOUNT_ID} --region $REGION

# 6. Install S3 CSI driver (required by inference operator)
aws eks create-addon \
  --cluster-name $EKS_CLUSTER \
  --addon-name aws-mountpoint-s3-csi-driver \
  --region $REGION

# Wait for S3 CSI to be active
aws eks wait addon-active --cluster-name $EKS_CLUSTER \
  --addon-name aws-mountpoint-s3-csi-driver --region $REGION

# 7. Install inference operator addon
aws eks create-addon \
  --cluster-name $EKS_CLUSTER \
  --addon-name amazon-sagemaker-hyperpod-inference \
  --configuration-values "{\"executionRoleArn\":\"arn:aws:iam::${ACCOUNT_ID}:role/SageMakerHyperPodInference-${CLUSTER_SHORT}\",\"tlsCertificateS3Bucket\":\"hyperpod-tls-${CLUSTER_SHORT}-${ACCOUNT_ID}\"}" \
  --region $REGION

# 8. Wait for addon (takes 3-5 minutes)
echo "Waiting for inference operator..."
sleep 180

# 9. Fix ALB controller IRSA (the addon installs with a placeholder role)
kubectl annotate sa aws-load-balancer-controller -n hyperpod-inference-system \
  eks.amazonaws.com/role-arn=arn:aws:iam::${ACCOUNT_ID}:role/HyperPodALBController-${CLUSTER_SHORT} --overwrite

kubectl rollout restart deployment hyperpod-inference-alb -n hyperpod-inference-system

# 10. Verify all pods are ready
kubectl get pods -n hyperpod-inference-system -w
# Wait until controller-manager shows 1/1 Running
```

---

## Stage 2: Deploy the Inference Endpoint + Public API Gateway

This stage deploys everything in a **single `tofu apply`**: the ACM
certificate, namespace, HF token secret, IAM permissions, the
`InferenceEndpointConfig`, and the public API Gateway (auth + throttling +
custom domain). The operator downloads the model and creates the internal ALB;
Terraform waits for that ALB and **auto-discovers** it (by its ingress tags) to
wire up the API Gateway VPC Link — no manual ARN paste, no two-phase apply.

> If you ran Stage 1, populate `terraform.tfvars` from its outputs first:
> ```bash
> ( cd cluster && tofu output -raw endpoint_stage_tfvars ) >> terraform.tfvars
> ```
> then add `domain_name`, `hosted_zone_id`, `hf_token`, `api_keys`, and the
> model settings.

```bash
# 1. Edit terraform.tfvars with your values (see reference below)
# 2. Leave internal_alb_arn = "" so the ALB is auto-discovered
# 3. Set your api_keys list (generate with: openssl rand -hex 24)

tofu init
tofu plan -out=tfplan
tofu apply tfplan
```

A single apply takes ~10-15 minutes end to end:
- ACM certificate issued (~1 min)
- Model downloaded from Hugging Face (~64GB for 32B model, 5-8 min)
- vLLM loads the model into the GPUs (~2 min)
- Operator creates the internal ALB (~2 min) — Terraform polls until it has a
  443 listener, then auto-discovers it
- VPC Link + API Gateway + Lambda authorizer + custom domain (~2-3 min)

Monitor progress in another terminal:
```bash
# Watch pod status
kubectl get pods -n hyperpod-ns-inference -w

# Check model download progress
kubectl logs <pod-name> -c hf-model-downloader -n hyperpod-ns-inference -f

# Check deployment status
kubectl describe inferenceendpointconfigs.inference.sagemaker.aws.amazon.com \
  <endpoint-name> -n hyperpod-ns-inference | grep -A5 "State:"
```

> **Overriding auto-discovery:** if you want API Gateway to front a specific
> pre-existing ALB instead, set `internal_alb_arn` to that ARN in
> `terraform.tfvars`. When it is empty (the default), the operator-managed ALB
> is discovered automatically.

---

## Step 3: Test the Endpoint

```bash

# With valid API key → 200 OK with inference response
curl -s https://<your-domain>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: <your-api-key>" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}], "model": "Qwen/Qwen2.5-32B-Instruct", "max_tokens": 20}'
```

## How to Find Your Configuration Values

Before filling in `terraform.tfvars`, use these commands to discover the required values:

```bash
# Set your base variables
REGION="us-east-2"
HYPERPOD_CLUSTER_ARN="arn:aws:sagemaker:us-east-2:ACCOUNT_ID:cluster/CLUSTER_ID"

# --- EKS cluster name ---
EKS_CLUSTER=$(aws sagemaker describe-cluster \
  --cluster-name $HYPERPOD_CLUSTER_ARN --region $REGION \
  --query 'Orchestrator.Eks.ClusterArn' --output text | cut -d'/' -f2)
echo "eks_cluster_name = \"$EKS_CLUSTER\""

# --- VPC ID ---
VPC_ID=$(aws eks describe-cluster --name $EKS_CLUSTER --region $REGION \
  --query 'cluster.resourcesVpcConfig.vpcId' --output text)
echo "vpc_id = \"$VPC_ID\""

# --- Public subnets (for API Gateway VPC Link — must have internet gateway route) ---
aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=map-public-ip-on-launch,Values=true" \
  --query 'Subnets[*].[SubnetId,AvailabilityZone]' \
  --region $REGION --output table
# Copy the subnet IDs into public_subnet_ids

# --- Private subnets (where the internal ALB and pods live) ---
aws eks describe-cluster --name $EKS_CLUSTER --region $REGION \
  --query 'cluster.resourcesVpcConfig.subnetIds' --output json
# These are typically private subnets — copy into private_subnet_ids

# --- Inference operator role name (created in Step 1) ---
echo "inference_operator_role_name = \"SageMakerHyperPodInference-<your-cluster-short-name>\""

# --- Internal ALB ARN ---
# Not required: leave internal_alb_arn = "" and Terraform auto-discovers the
# operator-managed ALB during apply. (Only look this up if you intend to front
# a specific pre-existing ALB.)
aws elbv2 describe-load-balancers --region $REGION \
  --query 'LoadBalancers[?Scheme==`internal` && Type==`application` && contains(LoadBalancerName,`hyperpod`)].[LoadBalancerArn,State.Code]' \
  --output table
# Only needed if you set internal_alb_arn to override auto-discovery
```

### Domain and DNS setup

Terraform **automatically creates** the ACM certificate and DNS validation records. You only need to provide:
- `domain_name` — the subdomain you want (e.g., `inference.mycompany.com`)
- `hosted_zone_id` — the Route 53 hosted zone that owns the parent domain

Terraform handles:
- Requesting the ACM certificate for your domain
- Creating the DNS CNAME record for certificate validation
- Waiting for the certificate to be issued
- Attaching the cert to the ALB (via the operator) and API Gateway custom domain
- Creating the Route 53 A record alias pointing your domain to API Gateway

**You do NOT need to manually create any certificates or DNS records.**

To find your hosted zone ID:
```bash
aws route53 list-hosted-zones --query 'HostedZones[?Config.PrivateZone==`false`].[Id,Name]' --output table
# Use the zone that matches your domain (e.g., zone "mycompany.com." for domain "inference.mycompany.com")
# The ID format is "/hostedzone/ZXXXXXXXXXXXXX" — use only the ZXXXXXXXXXXXXX part
```

If you don't have a Route 53 hosted zone, create one:
```bash
aws route53 create-hosted-zone --name mycompany.com --caller-reference $(date +%s)
# Then update your domain registrar's NS records to point to the Route 53 name servers
```

---

## terraform.tfvars Reference

```hcl
# === REQUIRED: Cluster Configuration ===
region               = "us-east-2"
hyperpod_cluster_arn = "arn:aws:sagemaker:us-east-2:ACCOUNT_ID:cluster/CLUSTER_ID"
eks_cluster_name     = "sagemaker-CLUSTERNAME-SUFFIX-eks"
vpc_id               = "vpc-XXXXXXXXXXXXXXXXX"

public_subnet_ids = [
  "subnet-XXXXXXXXXXXXXXXXX",
  "subnet-XXXXXXXXXXXXXXXXX",
]

private_subnet_ids = [
  "subnet-XXXXXXXXXXXXXXXXX",
  "subnet-XXXXXXXXXXXXXXXXX",
]

# === REQUIRED: Domain + DNS ===
domain_name    = "inference.yourdomain.com"
hosted_zone_id = "ZXXXXXXXXXXXXX"

# === REQUIRED: IAM ===
inference_operator_role_name = "SageMakerHyperPodInference-YOURCLUSTER"

# === REQUIRED: Model Configuration ===
model_id             = "Qwen/Qwen2.5-32B-Instruct"
instance_type        = "ml.p5.48xlarge"
gpu_count            = 2
memory               = "120Gi"
cpu                  = "24"
tensor_parallel_size = 2
max_model_len        = 4096

# === REQUIRED: Secrets ===
hf_token = "hf_YOUR_TOKEN_HERE"
api_keys = ["sk-GENERATE_WITH_openssl_rand_-hex_24"]

# === Internal ALB (optional override) ===
internal_alb_arn = ""  # Leave empty to auto-discover the operator's ALB (single apply)

# === OPTIONAL ===
endpoint_name      = "qwen32b-public"
namespace          = "hyperpod-ns-inference"
vllm_image         = "vllm/vllm-openai:v0.10.1"
tls_cert_s3_bucket = ""                              # Auto-generated if empty
enable_waf         = false
throttle_rate_limit  = 10
throttle_burst_limit = 20
```

## Resource Sizing Guide

| Model Size | GPU Count | Memory | CPU | Tensor Parallel | Instance Type |
|-----------|-----------|--------|-----|-----------------|---------------|
| 1.5B-7B | 1 | 32Gi | 8 | 1 | ml.g6e.12xlarge or ml.p5.48xlarge |
| 13B | 1 | 64Gi | 16 | 1 | ml.p5.48xlarge |
| 32B | 2 | 120Gi | 24 | 2 | ml.p5.48xlarge |
| 70B | 4 | 256Gi | 48 | 4 | ml.p5.48xlarge |
| 70B (max throughput) | 8 | 512Gi | 96 | 8 | ml.p5.48xlarge |

## Validation Results

```

$ curl -v https://inference.yourdomain.com/v1/chat/completions \
    -H "x-api-key: <your-valid-key>" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"model":"Qwen/Qwen2.5-32B-Instruct","max_tokens":20}'

* SSL connection using TLSv1.3 / TLS_AES_128_GCM_SHA256
*  subject: CN=inference.yourdomain.com
*  issuer: C=US; O=Amazon; CN=Amazon RSA 2048 M01
*  SSL certificate verify ok.
> POST /v1/chat/completions HTTP/2
< HTTP/2 200

{"model":"Qwen/Qwen2.5-32B-Instruct",
 "choices":[{"message":{"content":"2+2 equals 4."}}],
 "usage":{"prompt_tokens":36,"completion_tokens":8}}
```

### Security features validated

| Feature | Status |
|---------|--------|
| TLS 1.3 with Amazon-issued cert | Verified |
| API key via `x-api-key` header | Verified |
| API key via `Authorization: Bearer` (OpenAI SDK) | Verified |
| Invalid/missing key rejected (401/403) | Verified |
| Rate limiting (configurable) | Configured |
| VPC Link (internal ALB not publicly exposed) | Verified |
| HTTP/2 | Verified |
| Health check endpoint `/health` (no auth) | Configured |

## Cleanup

Destroy in reverse order — Stage 2 (endpoint) first, then Stage 1 (cluster):

```bash
# 1. Stage 2 — endpoint resources (API Gateway, ACM, InferenceEndpointConfig)
tofu destroy

# 2. Stage 1 — cluster (HyperPod nodes, EKS, VPC, inference operator)
#    Only if you created the cluster with the cluster/ stack.
cd cluster
tofu destroy
cd ..
```

If you used a pre-existing cluster (skipped Stage 1), you can optionally remove
just the inference operator addon instead:

```bash
aws eks delete-addon --cluster-name <eks-cluster-name> \
  --addon-name amazon-sagemaker-hyperpod-inference --region <region>
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| ALB ingress has no address | ALB controller IRSA broken | Check SA annotation, fix role trust policy, restart ALB pods |
| `amazon-sagemaker-hyperpod-inference` add-on `timeout while waiting for state to become 'ACTIVE'` | Controller-manager not healthy yet (slow GPU node bring-up / large image pulls); FSx Lustre CSI driver missing | Stage 1 installs the FSx driver and sets a 40m create timeout (`inference_operator_create_timeout`). If it still times out, **re-run `tofu apply`** — it is idempotent and resumes. |
| Target stuck `unused` / `Target.NotInUse` | Internal ALB placed in different AZ than the GPU pod (only the /28 EKS subnets carried `kubernetes.io/role/internal-elb`) | Stage 1 fixes this automatically by tagging the HyperPod private subnets for internal-elb and untagging the /28 EKS subnets (see `cluster/subnet_tags.tf`). For a pre-existing cluster, tag the subnet in the GPU AZ with `kubernetes.io/role/internal-elb=1`, delete the ALB ingress, and let the operator recreate it. |
| `FailedBuildModel: ec2:CreateSecurityGroup` | ALB controller role missing EC2 perms | Attach `AmazonEC2FullAccess` + `AmazonVPCFullAccess` to ALB role |
| `S3 CSI driver not installed` | Missing prerequisite addon | Install `aws-mountpoint-s3-csi-driver` addon |
| 504 Gateway Timeout | Response took >30s (large generation) | Reduce `max_tokens` or use streaming via SageMaker endpoint |
| DNS resolves to private IPs | You're inside the VPC | Test from external network or use `--resolve` with API GW public IPs |
| Init container stuck | 32B model downloading (~64GB) | Wait 5-10 min, check: `kubectl logs <pod> -c hf-model-downloader` |
| `tokenSecretRef` error | HF token secret not found | Verify: `kubectl get secret hf-token-secret -n <namespace>` |
| `Unauthorized` with valid key | Lambda env vars not updated | Run `tofu apply` after changing `api_keys` |

---

## Using with OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://inference.yourdomain.com/v1",
    api_key="sk-your-api-key-here"  # Sent as Authorization: Bearer <key>
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-32B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms."}],
    max_tokens=200
)

print(response.choices[0].message.content)
```

The Lambda authorizer accepts both:
- `x-api-key: <key>` header (curl, httpie)
- `Authorization: Bearer <key>` header (OpenAI SDK, LangChain, LiteLLM)

### Using with curl

```bash
curl -X POST https://inference.yourdomain.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-your-api-key-here" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "model": "Qwen/Qwen2.5-32B-Instruct", "max_tokens": 100}'
```

### Health check (no auth required)

```bash
curl https://inference.yourdomain.com/health
```

---

## API Key Management

### Generate a new API key

```bash
openssl rand -hex 24
# Example output: a94f7c2b1e8d3f6a0c5b9e7d4f2a1c8b6e3d9f0a7b5c
# Prefix it: sk-prod-a94f7c2b1e8d3f6a0c5b9e7d4f2a1c8b6e3d9f0a7b5c
```

### Rotate keys (zero-downtime)

1. Generate new key
2. Add new key to `api_keys` list → `tofu apply` (both old and new work)
3. Update all clients to use new key
4. Remove old key from `api_keys` list → `tofu apply` (old key revoked)

---

## Known Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| API Gateway HTTP API max timeout: 30s | Long generations (>500 tokens) may 504 | Keep `max_tokens` < 500, or use SageMaker endpoint for streaming |
| Lambda authorizer adds ~5ms latency | Negligible for inference | N/A |
| No streaming (SSE) support | Can't stream token-by-token | Use SageMaker `invoke-endpoint-with-response-stream` for streaming |

---
