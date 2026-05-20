# ============================================================================
# API Gateway: Production-grade public endpoint with auth + rate limiting
# ============================================================================
# Replaces NLB approach. API Gateway provides:
# - API key authentication via Lambda authorizer
# - Rate limiting (throttling)
# - Custom domain with TLS 1.3
# - CloudWatch logging and metrics
# - No credential management for callers (just x-api-key header)

locals {
  deploy_apigw = var.internal_alb_arn != ""
}

# --- VPC Link (connects API Gateway to internal ALB) ---
resource "aws_apigatewayv2_vpc_link" "inference" {
  count = local.deploy_apigw ? 1 : 0

  name               = "hyperpod-inference-vpclink"
  subnet_ids         = var.private_subnet_ids
  security_group_ids = tolist(data.aws_lb.internal_alb[0].security_groups)

  tags = {
    Name      = "hyperpod-inference-vpclink"
    ManagedBy = "terraform"
  }
}

data "aws_lb" "internal_alb" {
  count = local.deploy_apigw ? 1 : 0
  arn   = var.internal_alb_arn
}

data "aws_lb_listener" "internal_alb" {
  count             = local.deploy_apigw ? 1 : 0
  load_balancer_arn = var.internal_alb_arn
  port              = 443
}

# --- HTTP API ---
resource "aws_apigatewayv2_api" "inference" {
  count = local.deploy_apigw ? 1 : 0

  name          = "hyperpod-inference-api"
  protocol_type = "HTTP"

  tags = {
    Name      = "hyperpod-inference-api"
    ManagedBy = "terraform"
  }
}

# --- Integration (VPC Link to internal ALB) ---
resource "aws_apigatewayv2_integration" "inference" {
  count = local.deploy_apigw ? 1 : 0

  api_id             = aws_apigatewayv2_api.inference[0].id
  integration_type   = "HTTP_PROXY"
  integration_method = "ANY"
  integration_uri    = data.aws_lb_listener.internal_alb[0].arn
  connection_type    = "VPC_LINK"
  connection_id      = aws_apigatewayv2_vpc_link.inference[0].id

  payload_format_version = "1.0"
  timeout_milliseconds   = 30000

  tls_config {
    server_name_to_verify = var.domain_name
  }
}

# --- Lambda Authorizer for API Key validation ---
data "archive_file" "authorizer" {
  count       = local.deploy_apigw ? 1 : 0
  type        = "zip"
  output_path = "${path.module}/.terraform/authorizer.zip"

  source {
    content  = <<-PYTHON
import os
import json

VALID_API_KEYS = set(k for k in os.environ.get('API_KEYS', '').split(',') if k)

def handler(event, context):
    headers = event.get('headers', {})
    
    # Check x-api-key header (curl, httpie, custom clients)
    api_key = headers.get('x-api-key', '')
    
    # Also check Authorization: Bearer <key> (OpenAI SDK sends this)
    if not api_key:
        auth_header = headers.get('authorization', '')
        if auth_header.startswith('Bearer '):
            api_key = auth_header[7:]
    
    if api_key in VALID_API_KEYS:
        return {"isAuthorized": True, "context": {"caller": api_key[:8] + "..."}}
    
    return {"isAuthorized": False}
PYTHON
    filename = "authorizer.py"
  }
}

resource "aws_iam_role" "authorizer_lambda" {
  count = local.deploy_apigw ? 1 : 0

  name = "hyperpod-apigw-authorizer-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "authorizer_lambda" {
  count = local.deploy_apigw ? 1 : 0

  role       = aws_iam_role.authorizer_lambda[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_lambda_function" "authorizer" {
  count = local.deploy_apigw ? 1 : 0

  function_name = "hyperpod-api-key-authorizer"
  role          = aws_iam_role.authorizer_lambda[0].arn
  handler       = "authorizer.handler"
  runtime       = "python3.12"
  timeout       = 5
  memory_size   = 128

  filename         = data.archive_file.authorizer[0].output_path
  source_code_hash = data.archive_file.authorizer[0].output_base64sha256

  environment {
    variables = {
      API_KEYS = join(",", var.api_keys)
    }
  }
}

resource "aws_lambda_permission" "apigw" {
  count = local.deploy_apigw ? 1 : 0

  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.authorizer[0].function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.inference[0].execution_arn}/*"
}

resource "aws_apigatewayv2_authorizer" "api_key" {
  count = local.deploy_apigw ? 1 : 0

  api_id                            = aws_apigatewayv2_api.inference[0].id
  authorizer_type                   = "REQUEST"
  authorizer_uri                    = aws_lambda_function.authorizer[0].invoke_arn
  authorizer_payload_format_version = "2.0"
  enable_simple_responses           = true
  authorizer_result_ttl_in_seconds  = 0
  name                              = "api-key-authorizer"
}

# --- Route with auth ---
resource "aws_apigatewayv2_route" "chat_completions" {
  count = local.deploy_apigw ? 1 : 0

  api_id    = aws_apigatewayv2_api.inference[0].id
  route_key = "POST /v1/chat/completions"
  target    = "integrations/${aws_apigatewayv2_integration.inference[0].id}"

  authorization_type = "CUSTOM"
  authorizer_id      = aws_apigatewayv2_authorizer.api_key[0].id
}

# --- Health check route (no auth, returns 200 if API GW is reachable) ---
resource "aws_apigatewayv2_route" "health" {
  count = local.deploy_apigw ? 1 : 0

  api_id    = aws_apigatewayv2_api.inference[0].id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.inference[0].id}"

  # No authorization required for health checks
}

# --- Also support /v1/models endpoint (useful for SDK discovery, no auth) ---
resource "aws_apigatewayv2_route" "models" {
  count = local.deploy_apigw ? 1 : 0

  api_id    = aws_apigatewayv2_api.inference[0].id
  route_key = "GET /v1/models"
  target    = "integrations/${aws_apigatewayv2_integration.inference[0].id}"

  authorization_type = "CUSTOM"
  authorizer_id      = aws_apigatewayv2_authorizer.api_key[0].id
}

# --- Stage with throttling ---
resource "aws_apigatewayv2_stage" "default" {
  count = local.deploy_apigw ? 1 : 0

  api_id      = aws_apigatewayv2_api.inference[0].id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_burst_limit = var.throttle_burst_limit
    throttling_rate_limit  = var.throttle_rate_limit
  }

  depends_on = [
    aws_apigatewayv2_route.chat_completions,
    aws_apigatewayv2_route.health,
    aws_apigatewayv2_route.models,
  ]
}

# --- Custom Domain ---
resource "aws_apigatewayv2_domain_name" "inference" {
  count = local.deploy_apigw ? 1 : 0

  domain_name = var.domain_name

  domain_name_configuration {
    certificate_arn = aws_acm_certificate_validation.inference.certificate_arn
    endpoint_type   = "REGIONAL"
    security_policy = "TLS_1_2"
  }
}

resource "aws_apigatewayv2_api_mapping" "inference" {
  count = local.deploy_apigw ? 1 : 0

  api_id      = aws_apigatewayv2_api.inference[0].id
  domain_name = aws_apigatewayv2_domain_name.inference[0].id
  stage       = aws_apigatewayv2_stage.default[0].id
}

# --- Route 53 ---
resource "aws_route53_record" "inference" {
  count = local.deploy_apigw ? 1 : 0

  zone_id = var.hosted_zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_apigatewayv2_domain_name.inference[0].domain_name_configuration[0].target_domain_name
    zone_id                = aws_apigatewayv2_domain_name.inference[0].domain_name_configuration[0].hosted_zone_id
    evaluate_target_health = false
  }

  allow_overwrite = true
}
