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
