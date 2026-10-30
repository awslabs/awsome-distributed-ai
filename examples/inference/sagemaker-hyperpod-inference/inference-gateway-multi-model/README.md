# Multi-Model Inference with the HyperPod Inference Gateway

Deploy two models behind a single [HyperPod Inference Gateway](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment-inference-gateway.html) endpoint and route requests by the `model` field in the request body.

## What This Example Shows

The Inference Gateway uses [llm-d](https://llm-d.ai/) components to route and schedule inference requests:

```
Client
  │
  ▼
┌─────────────────────────────────────────────────┐
│  ALB (internal, one per InferenceGatewayConfig)  │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  Body-Based Router (BBR)                         │
│  Reads "model" from the request JSON body and    │
│  selects the matching InferencePool              │
└──────────┬──────────────────────┬───────────────┘
           │                      │
           ▼                      ▼
┌────────────────────┐  ┌────────────────────┐
│  Endpoint Picker   │  │  Endpoint Picker   │
│  (Qwen2.5-7B)     │  │  (Qwen2.5-3B)     │
│  Scores pods by    │  │  Scores pods by    │
│  queue, KV-cache,  │  │  queue, KV-cache,  │
│  prefix-cache      │  │  prefix-cache      │
└────────┬───────────┘  └────────┬───────────┘
         │                       │
         ▼                       ▼
┌────────────────────┐  ┌────────────────────┐
│  vLLM 0.9.2        │  │  vLLM 0.9.2        │
│  Qwen2.5-7B-Inst.  │  │  Qwen2.5-3B-Inst.  │
│  (1 replica)       │  │  (1 replica)       │
└────────────────────┘  └────────────────────┘
```

With one replica per model, this example demonstrates **model-level routing**: the Body-Based Router reads the `model` field and sends the request to the correct InferencePool. To see llm-d's **pod-level scheduling** (where the Endpoint Picker chooses among multiple pods using queue depth, KV-cache utilization, prefix-cache and least-recently-used scores), set `replicas: 2` on one of the models in its manifest before the first `kubectl apply`, and enable `logLevel: 4` on the Endpoint Picker as described in the [troubleshooting guide](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment-ts-inference-gateway.html).

## Prerequisites

| Requirement | Details |
|-------------|---------|
| HyperPod EKS cluster | Built with [`architectures/sagemaker-hyperpod-eks/`](../../../../architectures/sagemaker-hyperpod-eks/) with `create_hyperpod_inference_operator_module = true`, which installs the inference add-on and cert-manager |
| Inference Operator add-on | `amazon-sagemaker-hyperpod-inference` v2.0.0-eksbuild.2 or later |
| GPU instance groups | `ml.g5.4xlarge` (7B model, 64 GiB) + `ml.g5.2xlarge` (3B model, 32 GiB) — see [Instance sizing](#instance-sizing) |
| kubectl | Configured for the HyperPod EKS cluster |
| Tools | `jq`, `curl`, `python3` |

### Inference Gateway setup

The Inference Operator add-on enables the gateway controller by default (`inferenceGateway.enabled: true` in the add-on schema). However, **TLS requires the certificate-issuer IRSA role**. Without it, the gateway serves HTTP only and `.status.tls` is empty.

Create the role and add it to the add-on configuration:

```bash
CLUSTER=<eks-cluster> REGION=<region> ACCOUNT=<account-id> ROLE_NAME=<cert-issuer-role-name>
OIDC_ID=$(aws eks describe-cluster --name $CLUSTER --region $REGION \
  --query 'cluster.identity.oidc.issuer' --output text | sed 's|https://||')

# 1. Create the role that the gateway controller's service account assumes
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "arn:aws:iam::${ACCOUNT}:oidc-provider/${OIDC_ID}"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {"StringEquals": {
      "${OIDC_ID}:sub": "system:serviceaccount:hyperpod-inference-system:inference-gateway-controller",
      "${OIDC_ID}:aud": "sts.amazonaws.com"
    }}
  }]
}
EOF
aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file://trust-policy.json
aws iam attach-role-policy --role-name $ROLE_NAME \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerHyperPodInferenceGatewayAccess

# 2. Add the role to the configuration that the module set on the add-on
aws eks describe-addon --cluster-name $CLUSTER --region $REGION \
  --addon-name amazon-sagemaker-hyperpod-inference \
  --query addon.configurationValues --output text \
  | jq --arg role "arn:aws:iam::${ACCOUNT}:role/${ROLE_NAME}" \
      '.inferenceGateway.serviceAccount.roleArn = $role' > addon-config.json
aws eks update-addon --cluster-name $CLUSTER --region $REGION \
  --addon-name amazon-sagemaker-hyperpod-inference \
  --configuration-values file://addon-config.json --resolve-conflicts OVERWRITE
aws eks wait addon-active --cluster-name $CLUSTER --region $REGION \
  --addon-name amazon-sagemaker-hyperpod-inference

# 3. Restart the controller so that it starts with the role
kubectl rollout restart deploy/inference-gateway-controller -n hyperpod-inference-system
```

> **Note:** The Terraform module manages this add-on's `configuration_values`, so the next `terraform apply` sets the configuration back without the role. Repeat step 2 after each apply.

### Instance sizing

The operator mounts the model-weights volume as `emptyDir: {medium: Memory}` at `/opt/ml/model` and sets `HF_HOME=/opt/ml/model`, so the Hugging Face download and its xet chunk cache are stored in **tmpfs and count against the container's memory limit**.

Observed memory usage (operator v3.7.0, add-on v2.1.0-eksbuild.1):

| Model | Weights | Xet cache | Total pod memory | Minimum instance |
|-------|---------|-----------|--------------------|------------------|
| Qwen2.5-7B-Instruct | ~15 GB | ~10 GB | ~29.0 GiB (29,691 MiB) | `ml.g5.4xlarge` (64 GiB) |
| Qwen2.5-3B-Instruct | ~5.8 GB | ~5.8 GB | ~14.5 GiB (14,889 MiB) | `ml.g5.2xlarge` (32 GiB) |

The 3B model was OOMKilled five times at `12Gi` memory limit and ran successfully at `24Gi`. Both models use ungated Hugging Face repositories, so no access token is needed.

> **Tip:** To pin model weights to a specific revision, add `commitSHA` to `modelSourceConfig.huggingFaceModel` in the manifests. Without it, the download follows the repository's default branch.

## File Structure

```
inference-gateway-multi-model/
├── README.md
├── k8s/
│   ├── 01-qwen2.5-7b.yaml         # 7B InferenceEndpointConfig
│   ├── 02-qwen2.5-3b.yaml         # 3B InferenceEndpointConfig
│   └── 03-test-pod.yaml           # Test pod with RBAC for validation
└── scripts/
    └── 0.test-gateway.sh           # End-to-end gateway validation (non-streaming + streaming)
```

## Step 1: Deploy the Models

Apply both manifests. The shared `inferenceGateway.name: qwen-gateway` attaches both models to a single gateway with one ALB, one Body-Based Router, and two schedulers:

```bash
kubectl apply -f k8s/01-qwen2.5-7b.yaml
kubectl apply -f k8s/02-qwen2.5-3b.yaml
```

## Step 2: Wait for Models to Be Ready

```bash
kubectl wait inferenceendpointconfig qwen2-5-7b-instruct \
  --for=condition=DeploymentComplete --timeout=600s

kubectl wait inferenceendpointconfig qwen2-5-3b-instruct \
  --for=condition=DeploymentComplete --timeout=600s
```

Verify the pods are running (each model runs 3 containers: vLLM + sidecar-reverse-proxy + otel-collector):

```bash
kubectl get pods -n default
# NAME                                    READY   STATUS    AGE
# qwen2-5-7b-instruct-xxxxx-yyyyy        3/3     Running   5m
# qwen2-5-3b-instruct-xxxxx-yyyyy        3/3     Running   5m
```

## Step 3: Verify the Gateway

Check that the InferenceGatewayConfig has both schedulers:

```bash
kubectl get igwc qwen-gateway -n default \
  -o jsonpath='{range .spec.schedulers[*]}{.name}{" "}{.scheduler}{"\n"}{end}'
# Expected output:
# qwen2-5-3b-instruct llm-d
# qwen2-5-7b-instruct llm-d
```

Check the gateway endpoint:

```bash
kubectl get ingress qwen-gateway-alb -n hyperpod-inference-system \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

## Step 4: Access and Security

Both the gateway ALB and the per-model ALBs are **internal** load balancers. Requests must originate from inside the VPC (e.g., from a pod, a bastion host, or a VPN-connected workstation).

Key security characteristics:

- **No request authentication by default.** Without `spec.auth.jwt` on the InferenceGatewayConfig, the gateway accepts any request that reaches it. For production use, configure [JWT authentication](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment-inference-gateway.html) on the gateway.
- **Private TLS certificate.** The gateway's certificate is issued by a private issuer (`sagemaker-inference-gateway`), not a public CA. Clients must use `-k` (or `--insecure`) with curl, or trust the private CA. With the certificate-issuer IRSA role configured, the gateway serves HTTPS; without it, only HTTP is available.

## Step 5: Run the Validation Script

The script must run from **inside the VPC**. Deploy the test pod:

```bash
kubectl create configmap gateway-test-script --from-file=scripts/0.test-gateway.sh
kubectl apply -f k8s/03-test-pod.yaml
kubectl wait pod/gateway-test --for=condition=Ready --timeout=120s
kubectl exec gateway-test -- bash /scripts/0.test-gateway.sh
```

### Expected output

The script runs 12 checks: 4 non-streaming (Steps 2-5) and 8 streaming (Steps 6-9, four checks per model).

```
═══════════════════════════════════════════════════════════
  Inference Gateway — End-to-End Validation
═══════════════════════════════════════════════════════════

Step 1: Discovering gateway endpoint...
  Gateway endpoint: internal-k8s-hyperpod-...elb.amazonaws.com
  Protocol: https
  URL: https://internal-k8s-hyperpod-...elb.amazonaws.com/v1/chat/completions

Step 2: Checking InferenceGatewayConfig...
  ✅ PASS: IGWC 'qwen-gateway' has 2 schedulers (2 Available)

Step 3: Testing Qwen/Qwen2.5-7B-Instruct...
  ✅ PASS: Qwen/Qwen2.5-7B-Instruct — model matched, id: chatcmpl-<hex>

Step 4: Testing Qwen/Qwen2.5-3B-Instruct...
  ✅ PASS: Qwen/Qwen2.5-3B-Instruct — model matched, id: chatcmpl-<hex>

Step 5: Testing unknown model rejection...
  ✅ PASS: Unknown model 'does-not-exist/fake-model' rejected with HTTP 404

Step 6: Streaming test for Qwen/Qwen2.5-7B-Instruct...
  ✅ PASS: Qwen/Qwen2.5-7B-Instruct streaming — HTTP 200
  ✅ PASS: Qwen/Qwen2.5-7B-Instruct streaming — model field matches
  ✅ PASS: Qwen/Qwen2.5-7B-Instruct streaming — ends with data: [DONE]
  ✅ PASS: Qwen/Qwen2.5-7B-Instruct streaming — time spread N.NNN s ≥ 0.5 s (first→last content chunk)
  ℹ️  Content chunks: NN, first: NNNNNNNNNN.NNNNNN, last: NNNNNNNNNN.NNNNNN, spread: N.NNN s

Step 8: Streaming test for Qwen/Qwen2.5-3B-Instruct...
  ✅ PASS: Qwen/Qwen2.5-3B-Instruct streaming — HTTP 200
  ✅ PASS: Qwen/Qwen2.5-3B-Instruct streaming — model field matches
  ✅ PASS: Qwen/Qwen2.5-3B-Instruct streaming — ends with data: [DONE]
  ✅ PASS: Qwen/Qwen2.5-3B-Instruct streaming — time spread N.NNN s ≥ 0.5 s (first→last content chunk)
  ℹ️  Content chunks: NN, first: NNNNNNNNNN.NNNNNN, last: NNNNNNNNNN.NNNNNN, spread: N.NNN s

═══════════════════════════════════════════════════════════
  ✅ All 12 checks passed!

  Gateway : https://internal-k8s-hyperpod-...elb.amazonaws.com
  IGWC    : qwen-gateway
  Path    : /v1/chat/completions
  Tests   : 12/12
            non-streaming: Steps 2-5 (4 checks)
            streaming:     Steps 6-9 (8 checks)
═══════════════════════════════════════════════════════════
```

The `chatcmpl-<hex>` id does not identify the path: vLLM generates that form for any request without an `X-Request-Id` header. The requests go through the gateway because the script sends them to the gateway's own load balancer (Ingress `hyperpod-inference-system/qwen-gateway-alb`). For a model the gateway does not route, the gateway's 404 body reads `no matching model: ...`, while vLLM's own 404 for an unknown model reads `The model ... does not exist.`

### Streaming checks (Steps 6-9)

Steps 6-9 send a streaming chat request (`"stream": true`, `max_tokens: 128`) through the gateway for each model and verify:

1. **HTTP 200** — the gateway accepts the streaming request.
2. **Model field matches** — the `model` field in SSE chunks matches the requested model, confirming correct routing.
3. **`data: [DONE]`** — the stream ends with the OpenAI SSE sentinel, confirming the full response was delivered.
4. **Arrival-time spread** — the first content chunk arrives well before the last. A proxy that buffers the response delivers all chunks in a burst with near-zero spread, even though the SSE framing is valid. The script records a timestamp on each chunk and checks that the spread exceeds 0.5 s (conservative; real runs typically show 2-4 s for 128 tokens).

## Cleanup

```bash
kubectl delete -f k8s/03-test-pod.yaml
kubectl delete configmap gateway-test-script
kubectl delete -f k8s/02-qwen2.5-3b.yaml
kubectl delete -f k8s/01-qwen2.5-7b.yaml
kubectl delete inferencegatewayconfig qwen-gateway -n default
```

> **Note:** Deleting the models removes their scheduler entries but does **not** delete the InferenceGatewayConfig or its load balancers. The explicit `delete igwc` is required.

## References

- [Inference Gateway documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment-inference-gateway.html)
- [Inference Gateway troubleshooting](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment-ts-inference-gateway.html)
- [Deploy models from S3, FSx or Hugging Face](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-model-deployment-deploy-ftm.html)
- [llm-d architecture](https://llm-d.ai/docs/architecture)
- [InferenceEndpointConfig CRD](https://github.com/aws/sagemaker-hyperpod-cli/blob/main/helm_chart/HyperPodHelmChart/charts/inference-operator/config/crd/inference.sagemaker.aws.amazon.com_inferenceendpointconfigs.yaml)