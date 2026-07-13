# ============================================================================
# CloudFront front door (replaces API Gateway HTTP API)
# ============================================================================
# Why: API Gateway HTTP API has a HARD 30s integration timeout that cannot be
# raised. Long / vision-model generations exceed it, so the gateway returns 504
# while the pod keeps generating in the background (and clients retry, doubling
# GPU cost). CloudFront removes that ceiling:
#   - CloudFront Function (viewer-request) validates x-api-key / Bearer token
#     => callers still need NO AWS credentials, same as before.
#   - VPC origin reaches the operator-managed INTERNAL ALB privately (never
#     exposed to the internet), same security posture as the old VPC Link.
#   - Responses stream (CloudFront does not buffer), and the origin
#     read timeout is raised well beyond 30s for long generations.
#   - AWS WAF rate-based rule replaces API Gateway throttling.
# The HyperPod operator, ALB, nginx sidecar, and pods are UNCHANGED.

locals {
  # Resolved ARN of the internal ALB to front (auto-discovered unless overridden).
  internal_alb_arn = var.internal_alb_arn != "" ? var.internal_alb_arn : data.aws_lb.discovered[0].arn
}

# --- Auto-discover the operator-managed internal ALB by its ingress tags ---
data "aws_lb" "discovered" {
  count = var.internal_alb_arn == "" ? 1 : 0

  tags = {
    "ingress.k8s.aws/stack" = "${var.namespace}/alb-${var.endpoint_name}"
    "elbv2.k8s.aws/cluster" = var.eks_cluster_name
  }

  depends_on = [null_resource.wait_for_alb]
}

data "aws_lb" "internal_alb" {
  arn = local.internal_alb_arn
}

# ============================================================================
# API key validation at the edge (CloudFront Function, viewer-request)
# ============================================================================
# Runs on every request before caching/origin. Accepts either:
#   x-api-key: <key>           (curl, httpie)
#   authorization: Bearer <key> (OpenAI SDK, LangChain, LiteLLM)
# Unauthenticated requests are rejected at the edge with 401 (never reach the
# model). /health is left open for health checks. No AWS credentials required.
resource "aws_cloudfront_key_value_store" "api_keys" {
  name    = "hyperpod-inference-api-keys"
  comment = "Valid API keys for the inference endpoint edge authorizer"
}

# Seed the KVS with the valid keys (rotate by editing var.api_keys + apply).
# nonsensitive() is required because for_each keys cannot be sensitive; the key
# strings become CloudFront KVS entries by design.
resource "aws_cloudfrontkeyvaluestore_key" "api_key" {
  for_each = nonsensitive(toset(var.api_keys))

  key_value_store_arn = aws_cloudfront_key_value_store.api_keys.arn
  key                 = each.value
  value               = "valid"
}

resource "aws_cloudfront_function" "api_key_auth" {
  name    = "hyperpod-inference-api-key-auth"
  runtime = "cloudfront-js-2.0"
  comment = "Validates x-api-key / Bearer token against a CloudFront KeyValueStore"
  publish = true

  key_value_store_associations = [aws_cloudfront_key_value_store.api_keys.arn]

  code = <<-JS
    import cf from 'cloudfront';

    // Bind to the single associated KeyValueStore (no-arg form).
    const kvs = cf.kvs();

    async function handler(event) {
        var request = event.request;

        // Health check is public (no auth).
        if (request.uri === '/health') {
            return request;
        }

        var headers = request.headers;
        var apiKey = '';

        if (headers['x-api-key'] && headers['x-api-key'].value) {
            apiKey = headers['x-api-key'].value;
        } else if (headers['authorization'] && headers['authorization'].value) {
            var auth = headers['authorization'].value;
            if (auth.startsWith('Bearer ')) {
                apiKey = auth.substring(7);
            }
        }

        if (apiKey) {
            try {
                await kvs.get(apiKey);   // throws if the key is not present
                return request;          // authorized -> continue to origin
            } catch (e) {
                // fall through to 401
            }
        }

        return {
            statusCode: 401,
            statusDescription: 'Unauthorized',
            headers: { 'content-type': { value: 'application/json' } },
            body: '{"error":"invalid or missing api key"}'
        };
    }
  JS
}

# ============================================================================
# AWS WAF (rate limiting) - must be in us-east-1 for CLOUDFRONT scope
# ============================================================================
resource "aws_wafv2_web_acl" "inference" {
  provider = aws.us_east_1
  name     = "hyperpod-inference-waf"
  scope    = "CLOUDFRONT"

  default_action {
    allow {}
  }

  rule {
    name     = "rate-limit"
    priority = 1

    action {
      block {}
    }

    statement {
      rate_based_statement {
        # WAF rate limit is per 5-minute window. Convert the requests/second
        # intent into a 5-minute budget (rate_limit * 300).
        limit              = var.throttle_rate_limit * 300
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "hyperpod-inference-rate-limit"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "hyperpod-inference-waf"
    sampled_requests_enabled   = true
  }

  tags = {
    Name      = "hyperpod-inference-waf"
    ManagedBy = "terraform"
  }
}

# ============================================================================
# CloudFront VPC origin -> private internal ALB (never internet-exposed)
# ============================================================================
resource "aws_cloudfront_vpc_origin" "internal_alb" {
  vpc_origin_endpoint_config {
    name                   = "hyperpod-inference-alb-origin-${substr(sha1(local.internal_alb_arn), 0, 8)}"
    arn                    = local.internal_alb_arn
    http_port              = 80
    https_port             = 443
    origin_protocol_policy = "https-only"

    origin_ssl_protocols {
      items    = ["TLSv1.2"]
      quantity = 1
    }
  }

  # A VPC origin cannot be updated in place while attached to a distribution
  # (e.g. if the operator recreates the ALB with a new ARN). create_before_destroy
  # lets Terraform stand up a replacement origin and swap the distribution to it
  # before deleting the old one. The name carries the ALB suffix so the two
  # origins never collide on name during the swap.
  lifecycle {
    create_before_destroy = true
  }
}

# ============================================================================
# CloudFront distribution
# ============================================================================
resource "aws_cloudfront_distribution" "inference" {
  enabled         = true
  comment         = "HyperPod inference public endpoint (long-running / streaming)"
  aliases         = [var.domain_name]
  is_ipv6_enabled = true
  http_version    = "http2and3"
  web_acl_id      = aws_wafv2_web_acl.inference.arn
  price_class     = "PriceClass_100"

  origin {
    origin_id           = "internal-alb"
    domain_name         = var.domain_name
    connection_attempts = 3
    connection_timeout  = 10

    vpc_origin_config {
      vpc_origin_id            = aws_cloudfront_vpc_origin.internal_alb.id
      origin_keepalive_timeout = 60
      # Origin read timeout raised well beyond API Gateway's hard 30s so long
      # generations complete on a single connection. Values > 60s require an
      # approved CloudFront quota increase for "Response timeout per origin".
      origin_read_timeout = var.cloudfront_origin_read_timeout
    }
  }

  default_cache_behavior {
    target_origin_id       = "internal-alb"
    viewer_protocol_policy = "https-only"
    allowed_methods        = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods         = ["GET", "HEAD"]
    compress               = false

    # Inference responses are dynamic and must never be cached.
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id

    function_association {
      event_type   = "viewer-request"
      function_arn = aws_cloudfront_function.api_key_auth.arn
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate_validation.cloudfront.certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name      = "hyperpod-inference"
    ManagedBy = "terraform"
  }
}

# AWS-managed policies: disable caching, forward everything to the origin.
data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_origin_request_policy" "all_viewer" {
  name = "Managed-AllViewer"
}

# ============================================================================
# Route 53 alias -> CloudFront
# ============================================================================
resource "aws_route53_record" "inference" {
  zone_id = var.hosted_zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.inference.domain_name
    zone_id                = aws_cloudfront_distribution.inference.hosted_zone_id
    evaluate_target_health = false
  }

  allow_overwrite = true
}
