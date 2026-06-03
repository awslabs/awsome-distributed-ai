# HyperPod Observability AddOn Role
resource "aws_iam_role" "hyperpod_observability_addon" {
  name                 = "${var.resource_name_prefix}-AddOn"
  path                 = "/service-role/"
  max_session_duration = 3600

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowEksAuthToAssumeRoleForPodIdentity"
        Effect = "Allow"
        Principal = {
          Service = "pods.eks.amazonaws.com"
        }
        Action = [
          "sts:AssumeRole",
          "sts:TagSession"
        ]
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
            "aws:SourceArn"     = "${var.eks_cluster_arn}"
          }
        }
      }
    ]
  })

  lifecycle {
    ignore_changes = [assume_role_policy]
  }
}

resource "aws_iam_role_policy" "hyperpod_observability_addon" {
  name = "${var.resource_name_prefix}-OnPol"
  role = aws_iam_role.hyperpod_observability_addon.id

  lifecycle {
    ignore_changes = [policy]
  }

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "PrometheusAccess"
        Effect = "Allow"
        Action = [
          "aps:RemoteWrite"
        ]
        Resource = "arn:aws:aps:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:workspace/*"
      },
      {
        Sid    = "CloudwatchLogsAccess"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams",
          "logs:PutLogEvents",
          "logs:GetLogEvents",
          "logs:FilterLogEvents",
          "logs:GetLogRecord",
          "logs:StartQuery",
          "logs:StopQuery",
          "logs:GetQueryResults"
        ]
        Resource = [
          "arn:aws:logs:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/Clusters/*",
          "arn:aws:logs:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/Clusters/*:log-stream:*"
        ]
      }
    ]
  })
}

# Grafana Workspace Role
resource "aws_iam_role" "grafana_workspace" {
  count = local.is_amg_allowed && var.create_grafana_workspace ? 1 : 0

  name = "${var.resource_name_prefix}-GraAcc"
  path = "/service-role/"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "GrafanaAssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "grafana.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })

  lifecycle {
    ignore_changes = [assume_role_policy]
  }
}

resource "aws_iam_role_policy" "grafana_workspace" {
  count = local.is_amg_allowed && var.create_grafana_workspace ? 1 : 0

  name = "${var.resource_name_prefix}-GraPol"
  role = aws_iam_role.grafana_workspace[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowReadingMetricsFromCloudWatch"
        Effect = "Allow"
        Action = [
          "cloudwatch:ListMetrics",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:GetMetricData",
          "cloudwatch:DescribeAlarmsForMetric",
          "cloudwatch:GetInsightRuleReport"
        ]
        Resource = "*"
      },
      {
        Sid    = "AllowReadingAlarmsFromCloudWatch"
        Effect = "Allow"
        Action = [
          "cloudwatch:DescribeAlarmHistory",
          "cloudwatch:DescribeAlarms"
        ]
        Resource = "arn:aws:cloudwatch:*:*:alarm:*"
      },
      {
        Sid    = "AllowReadingLogsFromCloudWatch"
        Effect = "Allow"
        Action = [
          "logs:DescribeLogGroups",
          "logs:StopQuery",
          "logs:GetLogEvents"
        ]
        Resource = "*"
      },
      {
        Sid    = "AllowQueryLogsFromCloudWatch"
        Effect = "Allow"
        Action = [
          "logs:GetLogGroupFields",
          "logs:StartQuery",
          "logs:GetQueryResults",
          "logs:GetLogEvents"
        ]
        Resource = [
          "arn:aws:logs:*:*:log-group:*",
          "arn:aws:logs:*:*:log-group:*:log-stream:*"
        ]
      },
      {
        Sid    = "AllowReadingTagsInstancesRegionsFromEC2"
        Effect = "Allow"
        Action = [
          "ec2:DescribeTags",
          "ec2:DescribeInstances",
          "ec2:DescribeRegions"
        ]
        Resource = "*"
      },
      {
        Sid      = "AllowReadingResourcesForTags"
        Effect   = "Allow"
        Action   = "tag:GetResources"
        Resource = "*"
      },
      {
        Sid    = "AllowListWorkspacesFromPrometheus"
        Effect = "Allow"
        Action = [
          "aps:ListWorkspaces"
        ]
        Resource = "*"
      },
      {
        Sid    = "AllowReadingMetricsFromPrometheus"
        Effect = "Allow"
        Action = [
          "aps:DescribeWorkspace",
          "aps:QueryMetrics",
          "aps:GetLabels",
          "aps:GetSeries",
          "aps:GetMetricMetadata"
        ]
        Resource = "arn:aws:aps:*:*:workspace/*"
      },
      {
        Sid    = "AllowReadingAlertsFromPrometheus"
        Effect = "Allow"
        Action = [
          "aps:ListRules",
          "aps:ListAlertManagerSilences",
          "aps:ListAlertManagerAlerts",
          "aps:GetAlertManagerStatus",
          "aps:ListAlertManagerAlertGroups",
          "aps:PutAlertManagerSilences",
          "aps:DeleteAlertManagerSilence"
        ]
        Resource = "arn:aws:aps:*:*:workspace/*"
      }
    ]
  })
}
