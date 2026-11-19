<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Monitoring

[`dashboard.json`](./dashboard.json) (uid `llmd-dsv4flash`) is a Grafana dashboard for the [EKS managed observability](../../../../../observability/prometheus-grafana/eks-managed-observability) stack (ADOT collector → Amazon Managed Service for Prometheus → Amazon Managed Grafana). Setup steps are in [Monitoring](../README.md#monitoring-optional).

## Scrape jobs

The dashboard reads three of that stack's [scrape jobs](../../../../../observability/prometheus-grafana/eks-managed-observability/README.md#scrape-configs). This sample adds nothing to the cluster for them beyond the pod labels the chart sets.

| Job | File | What it finds |
| --- | --- | --- |
| `dcgm-exporter` | `scrape-configs/dcgm-exporter.yaml` | GPU metrics |
| `sglang` | `scrape-configs/sglang.yaml` | The replicas, through the `sglang-metrics=true` label and `prometheus.io/*` annotations set in [`../modelserver/templates/deployment.yaml`](../modelserver/templates/deployment.yaml) |
| `llmd-epp` | `scrape-configs/llmd-epp.yaml` | Every llm-d EPP Service (port `http-metrics`). It sends the collector's ServiceAccount token, because EPP authenticates `/metrics`. Series get `job="epp"` |

## Panels

Throughput, TTFT/TPOT/queue time, per-replica request share and cache hit rate, running/queued requests and KV, the EPP's predicted prefix hit against the server's actual hit, and GPU.

`sglang:token_usage` excludes the evictable prefix cache, so it does not show whether the cache is full. The "KV occupied" panel uses `(kv_used_tokens + kv_evictable_tokens) / max_total_num_tokens` instead.

## Quick checks

In Grafana Explore:

```promql
up{job="sglang", namespace="dsv4flash-llmd"}                       # 4 series, value 1
llm_d_epp_ready_endpoints{job="epp", namespace="dsv4flash-llmd"}   # Ready replica count
sum by (pod) (rate(sglang:num_requests_total{job="sglang", namespace="dsv4flash-llmd"}[1m]))
```

## Flow control dashboard

[`dashboard_flow_control.json`](./dashboard_flow_control.json) (uid `llmd-dsv4flash-flowcontrol`) shows the EPP flow-control queues, from the `llmd-epp` and `sglang` jobs. Its panels have data only with the [flow-control overlay](../router/README.md#flow-control-overlay) on, and the queue panels only once the pool saturates (more than 256 in flight).

| Panel | What to look for |
| --- | --- |
| Pool saturation | ≥ 100% means EPP is holding requests |
| Where requests wait: EPP vs SGLang | Under overload, the backlog should be in the EPP queue, not in the SGLang queues |
| Queued requests / Queue time P90 by priority | Higher bands drain first; premium wait stays near 0 |
| Request outcomes by priority | `RejectedCapacity` = band queue full, `EvictedTTL` = waited past `defaultRequestTTL`, both return `429` |
| Band queue occupancy | Queued / band `maxRequests`; at 100% the band rejects |
| By fairness ID | Wait per `x-llm-d-inference-fairness-id` within a band |

The dashboard shows queue time in EPP only. Client TTFT by priority comes from AIPerf, because SGLang does not see the priority header.
