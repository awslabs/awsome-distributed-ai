# ============================================================================
# Kubernetes: Namespace, HF token, InferenceEndpointConfig
# ============================================================================

# --- Namespace ---
resource "kubernetes_namespace" "inference" {
  metadata {
    name = var.namespace
  }
}

# --- HuggingFace token secret ---
resource "kubernetes_secret" "hf_token" {
  count = var.hf_token != "" ? 1 : 0

  metadata {
    name      = "hf-token-secret"
    namespace = var.namespace
  }

  data = {
    token = var.hf_token
  }

  type = "Opaque"

  depends_on = [kubernetes_namespace.inference]
}

# --- InferenceEndpointConfig ---
resource "kubernetes_manifest" "inference_endpoint" {
  manifest = {
    apiVersion = "inference.sagemaker.aws.amazon.com/v1"
    kind       = "InferenceEndpointConfig"
    metadata = {
      name      = var.endpoint_name
      namespace = var.namespace
      labels = var.queue_name != "" ? {
        "kueue.x-k8s.io/queue-name" = var.queue_name
      } : {}
    }
    spec = {
      modelName          = replace(var.model_id, "/", "-")
      endpointName       = var.endpoint_name
      instanceType       = var.instance_type
      invocationEndpoint = "v1/chat/completions"
      replicas           = 1

      modelSourceConfig = {
        modelSourceType = "huggingface"
        prefetchEnabled = true
        huggingFaceModel = merge(
          { modelId = var.model_id },
          var.hf_token != "" ? {
            tokenSecretRef = {
              name = "hf-token-secret"
              key  = "token"
            }
          } : {}
        )
      }

      tlsConfig = {
        customCertificateConfig = {
          acmArn     = aws_acm_certificate_validation.inference.certificate_arn
          domainName = var.domain_name
        }
        tlsCertificateOutputS3Uri = "s3://${aws_s3_bucket.tls_certs.id}"
      }

      dnsConfig = {
        hostedZoneId = var.hosted_zone_id
      }

      loadBalancer = {
        healthCheckPath = "/health"
      }

      worker = {
        image = var.vllm_image
        modelInvocationPort = {
          containerPort = 8000
          name          = "http"
        }
        modelVolumeMount = {
          name      = "model-weights"
          mountPath = "/opt/ml/model"
        }
        resources = {
          requests = {
            "nvidia.com/gpu" = tostring(var.gpu_count)
            memory           = var.memory
            cpu              = var.cpu
          }
          limits = {
            "nvidia.com/gpu" = tostring(var.gpu_count)
            memory           = var.memory
            cpu              = var.cpu
          }
        }
        args = [
          "--model", "/opt/ml/model",
          "--port", "8000",
          "--tensor-parallel-size", tostring(var.tensor_parallel_size),
          "--max-model-len", tostring(var.max_model_len),
          "--served-model-name", var.model_id
        ]
        environmentVariables = [
          {
            name  = "VLLM_REQUEST_TIMEOUT"
            value = "600"
          }
        ]
      }
    }
  }

  depends_on = [
    aws_acm_certificate_validation.inference,
    aws_iam_role_policy.operator_custom_cert,
    aws_s3_bucket.tls_certs,
    kubernetes_secret.hf_token,
    kubernetes_namespace.inference,
  ]
}

# ============================================================================
# Wait for the operator-managed internal ALB to be provisioned
# ============================================================================
# After the InferenceEndpointConfig is applied, the HyperPod inference operator
# (via the AWS Load Balancer Controller) creates an internal ALB tagged:
#   ingress.k8s.aws/stack = <namespace>/alb-<endpoint_name>
#   elbv2.k8s.aws/cluster = <eks_cluster_name>
# This poller blocks until that ALB exists and has an active 443 listener, so
# the API Gateway can auto-discover it in a single `tofu apply` (no manual ARN
# paste, no two-phase apply). Set var.internal_alb_arn to skip this and use an
# explicit ARN instead.
resource "null_resource" "wait_for_alb" {
  count = var.internal_alb_arn == "" ? 1 : 0

  triggers = {
    endpoint_name = var.endpoint_name
    namespace     = var.namespace
    cluster       = var.eks_cluster_name
  }

  provisioner "local-exec" {
    interpreter = ["/bin/bash", "-c"]
    command     = <<-EOT
      set -euo pipefail
      STACK_TAG="${var.namespace}/alb-${var.endpoint_name}"
      echo "Waiting for operator-managed ALB (tag ingress.k8s.aws/stack=$STACK_TAG)..."
      for i in $(seq 1 60); do
        ALB_ARN=$(aws elbv2 describe-load-balancers --region ${var.region} \
          --query "LoadBalancers[?Type=='application' && Scheme=='internal'].LoadBalancerArn" \
          --output text 2>/dev/null | tr '\t' '\n' | while read arn; do
            [ -z "$arn" ] && continue
            match=$(aws elbv2 describe-tags --resource-arns "$arn" --region ${var.region} \
              --query "TagDescriptions[0].Tags[?Key=='ingress.k8s.aws/stack' && Value=='$STACK_TAG'].Value" \
              --output text 2>/dev/null)
            if [ -n "$match" ]; then echo "$arn"; break; fi
          done)
        if [ -n "$ALB_ARN" ]; then
          LISTENER=$(aws elbv2 describe-listeners --load-balancer-arn "$ALB_ARN" --region ${var.region} \
            --query "Listeners[?Port==\`443\`].ListenerArn" --output text 2>/dev/null || true)
          if [ -n "$LISTENER" ]; then
            echo "Found ALB $ALB_ARN with 443 listener."
            exit 0
          fi
          echo "ALB found but 443 listener not ready yet... ($i/60)"
        else
          echo "ALB not found yet... ($i/60)"
        fi
        sleep 20
      done
      echo "Timeout: operator-managed ALB for $STACK_TAG not ready after 20 minutes" >&2
      exit 1
    EOT
  }

  depends_on = [kubernetes_manifest.inference_endpoint]
}
