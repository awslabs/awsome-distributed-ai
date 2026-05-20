# HyperPod Inference: Public Endpoint with Terraform

Deploy a Hugging Face model on SageMaker HyperPod with a publicly accessible HTTPS endpoint — fully managed by Terraform/OpenTofu. No AWS credentials required for callers.



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
3. **kubectl** installed
4. **A public Route 53 hosted zone** for your domain
5. **A SageMaker HyperPod EKS cluster** in service
6. **HuggingFace token** (for gated models or higher download rate limits)

## File Structure

```
├── providers.tf            # AWS + Kubernetes + Archive providers
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

## Step 0: EKS Access for Your Identity

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

## Step 2 (Phase 1): Deploy the Inference Endpoint

This creates the ACM certificate, namespace, HF token secret, IAM permissions, and the `InferenceEndpointConfig`. The operator then downloads the model and creates the internal ALB.

```bash
# 1. Edit terraform.tfvars with your values (see reference below)
# 2. Ensure internal_alb_arn = "" for Phase 1
# 3. Set your api_keys list (generate with: openssl rand -hex 24)

tofu init
tofu plan -out=tfplan
tofu apply tfplan
```

Wait 5-10 minutes for:
- ACM certificate to be issued (~1 min)
- Model to download from Hugging Face (~64GB for 32B model, 5-8 min)
- vLLM to load model into GPUs (~2 min)
- Operator to create the internal ALB (~2 min)

Monitor progress:
```bash
# Watch pod status
kubectl get pods -n hyperpod-ns-inference -w

# Check model download progress
kubectl logs <pod-name> -c hf-model-downloader -n hyperpod-ns-inference -f

# Check deployment status
kubectl describe inferenceendpointconfigs.inference.sagemaker.aws.amazon.com \
  qwen32b-public -n hyperpod-ns-inference | grep -A5 "State:"
```

---

## Step 3 (Phase 2): Create API Gateway (Public Access with Auth)

Once the internal ALB is active (pods show 3/3 Running and ingress has an ADDRESS):

```bash
# 1. Discover the internal ALB ARN
aws elbv2 describe-load-balancers --region us-east-2 \
  --query 'LoadBalancers[?Scheme==`internal` && Type==`application` && contains(LoadBalancerName,`hyperpod`)].[LoadBalancerArn]' --output text

# 2. Update terraform.tfvars:
#    internal_alb_arn = "arn:aws:elasticloadbalancing:us-east-2:..."

# 3. Apply Phase 2
tofu plan -out=tfplan
tofu apply tfplan
```

Wait 2-3 minutes for VPC Link + API Gateway + Lambda + custom domain provisioning.

---

## Step 4: Test the Endpoint

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

# --- Internal ALB ARN (only available AFTER Phase 1 deploy) ---
aws elbv2 describe-load-balancers --region $REGION \
  --query 'LoadBalancers[?Scheme==`internal` && Type==`application` && contains(LoadBalancerName,`hyperpod`)].[LoadBalancerArn,State.Code]' \
  --output table
# Copy the ARN into internal_alb_arn for Phase 2
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

# === Phase 2 (set after ALB is created) ===
internal_alb_arn = ""  # Set to ALB ARN after Phase 1, then re-apply

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

```bash
# Destroy all Terraform resources
tofu destroy

# Optionally remove the inference operator addon
aws eks delete-addon --cluster-name <eks-cluster-name> \
  --addon-name amazon-sagemaker-hyperpod-inference --region <region>
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| ALB ingress has no address | ALB controller IRSA broken | Check SA annotation, fix role trust policy, restart ALB pods |
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
