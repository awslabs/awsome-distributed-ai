
# Grafana VPC Endpoint
resource "aws_vpc_endpoint" "grafana" {
  count = local.is_amg_allowed ? 1 : 0

  vpc_id              = var.vpc_id
  service_name        = "com.amazonaws.${var.aws_region}.grafana-workspace"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = var.private_subnet_ids
  security_group_ids  = [var.security_group_id]
  private_dns_enabled = true

  tags = {
    Name      = "SageMaker"
    SageMaker = "true"
  }
}

# Prometheus VPC Endpoint
resource "aws_vpc_endpoint" "prometheus" {
  vpc_id              = var.vpc_id
  service_name        = "com.amazonaws.${var.aws_region}.aps-workspaces"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = var.private_subnet_ids
  security_group_ids  = [var.security_group_id]
  private_dns_enabled = true

  tags = {
    Name      = "SageMaker"
    SageMaker = "true"
  }
}
