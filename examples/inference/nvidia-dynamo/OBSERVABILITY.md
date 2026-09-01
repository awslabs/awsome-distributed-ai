# Observability Guide

Dynamo frontend exposes Prometheus metrics at `/metrics`. This guide covers how to get those metrics into the HyperPod Grafana dashboard.

## Setup

Run after installing the platform (step 01):

```bash
./scripts/01a-setup-observability.sh
```

This creates:
1. A stable Kubernetes Service (`dynamo-metrics`) that always points to the active frontend pod via label selector
2. A `customServiceScrapeTarget` in the HyperPod `ObservabilityConfig` CR, which tells the OTEL collector to scrape Dynamo metrics and send them to AMP

Metrics flow: **Dynamo pods → dynamo-metrics Service → OTEL collector → AMP → Grafana**

## Accessing Grafana

The HyperPod cluster creates an Amazon Managed Grafana workspace automatically.

### 1. Find the Grafana URL

```bash
aws grafana list-workspaces --region <REGION> --no-cli-pager \
  --query 'workspaces[*].{name:name,endpoint:endpoint,status:status}' --output table
```

Open the `endpoint` URL in your browser. Login is via AWS IAM Identity Center (SSO).

### 2. Get Admin access

By default, users are added as **Viewer** — which doesn't have access to Explore or dashboard creation. To promote to Admin:

```bash
# List current permissions
aws grafana list-permissions --workspace-id <WORKSPACE_ID> --region <REGION> --no-cli-pager

# Promote your user to Admin (use the user ID from the list above)
aws grafana update-permissions --workspace-id <WORKSPACE_ID> --region <REGION> --no-cli-pager \
  --update-instruction-batch '[{"action":"ADD","role":"ADMIN","users":[{"id":"<USER_ID>","type":"SSO_USER"}]}]'
```

After updating, **logout and login again** for the new role to take effect.

## Viewing Metrics

### Explore (ad-hoc queries)

1. Go to **Explore** (compass icon in the sidebar — only visible to Editor/Admin)
2. Select the **prometheus** data source
3. Try these queries:

| Query | What it shows |
|---|---|
| `dynamo_frontend_inflight_requests` | Current in-flight requests |
| `rate(dynamo_frontend_input_sequence_tokens_count[5m])` | Request rate (req/s) |
| `histogram_quantile(0.50, rate(dynamo_frontend_time_to_first_token_seconds_bucket[5m]))` | TTFT p50 |
| `histogram_quantile(0.99, rate(dynamo_frontend_time_to_first_token_seconds_bucket[5m]))` | TTFT p99 |
| `histogram_quantile(0.50, rate(dynamo_frontend_inter_token_latency_seconds_bucket[5m]))` | ITL/TPOT p50 |
| `histogram_quantile(0.50, rate(dynamo_frontend_request_duration_seconds_bucket[5m]))` | Request duration p50 |

### Available Metrics

The Dynamo frontend exposes these metric families:

| Metric | Type | Description |
|---|---|---|
| `dynamo_frontend_inflight_requests` | Gauge | Number of in-flight requests |
| `dynamo_frontend_disconnected_clients` | Gauge | Disconnected client count |
| `dynamo_frontend_input_sequence_tokens` | Histogram | Input sequence length distribution |
| `dynamo_frontend_output_sequence_tokens` | Histogram | Output sequence length distribution |
| `dynamo_frontend_time_to_first_token_seconds` | Histogram | TTFT distribution |
| `dynamo_frontend_inter_token_latency_seconds` | Histogram | Inter-token latency (TPOT) distribution |
| `dynamo_frontend_request_duration_seconds` | Histogram | Total request duration distribution |

### Tips

- Use `rate(...[5m])` for counters and histograms to see per-second rates
- Use `histogram_quantile(0.50, ...)` for p50, `0.99` for p99
- Filter by model: add `{model="openai/gpt-oss-20b"}` to any query
- Run a benchmark (`./scripts/08-benchmark.sh <scenario>`) to generate traffic and see metrics in action
