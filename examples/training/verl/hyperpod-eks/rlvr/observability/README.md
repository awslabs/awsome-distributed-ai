# Ray Observability for HyperPod EKS

Set up Ray metrics monitoring using HyperPod's observability stack with Amazon Managed Prometheus and Grafana.

## Overview

To integrate Ray metrics into the HyperPod Observability stack, we use the `customServiceScrapeTargets` feature in the ObservabilityConfig CRD.

## Prerequisites

1. HyperPod observability stack is installed (check with `kubectl get deployment -n hyperpod-observability`)
2. Ray cluster is deployed with metrics port exposed (8080)
3. Environment variables are set in `setup/env_vars`

## Setup

### 1. Add Ray Metrics to ObservabilityConfig

> **Note:** The following script overwrites spec.customServiceScrapeTargets on the ObservabilityConfig. If you've manually added other scrape targets, either:
> - add Ray targets manually instead of using the following script
> - run the script then merge your additional customServiceScrapeTargets manually back into the ObservabilityConfig 

Run the provided script to add Ray metrics scraping to the HyperPod ObservabilityConfig:

```bash
./observability/add-ray-metrics.sh
```

This patches the ObservabilityConfig CRD to add your Ray cluster's head service as a custom scrape target. The operator will automatically update the OTEL collector configuration.

**What it does:**
- Adds `customServiceScrapeTargets` to the ObservabilityConfig
- Configures scraping of Ray head service on port 8080
- Won't be overwritten by the operator's reconciliation loop

### 2. Restart OTEL Collector (Optional)

To speed up the configuration update (otherwise it takes ~10 minutes):

```bash
kubectl rollout restart deployment hyperpod-observability-central-collector -n hyperpod-observability
kubectl rollout status deployment hyperpod-observability-central-collector -n hyperpod-observability
```

### 3. Download Grafana Dashboards

1. You can download them directly from your cluster head pod:
```
HEAD_POD=$(kubectl get pods --selector ray.io/node-type=head,ray.io/cluster=rayml-efa -o jsonpath='{.items[0].metadata.name}')

# Copy dashboard files from the pod
kubectl cp $HEAD_POD:/tmp/ray/session_latest/metrics/grafana/dashboards/ ./dashboards/
```

2. Or you can download them directly from [KubeRay GitHub](https://github.com/ray-project/kuberay/tree/master/config/grafana):
```
# Clone the repo
git clone https://github.com/ray-project/kuberay.git --depth 1
cd kuberay/config/grafana
ls *_grafana_dashboard.json
```

### 4. Import Dashboards to Grafana

- `default_grafana_dashboard.json` - Main Ray Dashboard
- `data_grafana_dashboard.json` - Ray Data metrics
- `serve_grafana_dashboard.json` - Ray Serve metrics
- `serve_deployment_grafana_dashboard.json` - Per-deployment metrics

To import:
1. Open your Grafana workspace (check `$GRAFANA_ENDPOINT` in `setup/env_vars`)
2. Click "+" → "Import"
3. Upload each JSON file
4. Select your AMP data source
5. Use the "Cluster" dropdown to filter by `rayml-efa`

## Verify It's Working

```bash
# Check that custom scrape target was added
kubectl get observabilityconfig hyperpod-observability -n hyperpod-observability -o yaml | grep -A 5 customServiceScrapeTargets

# Verify Ray metrics endpoint is responding
HEAD_POD=$(kubectl get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
kubectl exec $HEAD_POD -- curl -s http://localhost:8080/metrics | head -n 20

# Check OTEL collector picked up the config
kubectl logs -n hyperpod-observability deployment/hyperpod-observability-central-collector --tail=100 | grep "custom-scrape-target"
```

Wait 2-3 minutes for metrics to flow to AMP, then check your Grafana dashboards.

## Troubleshooting

**No metrics in Grafana?**
- Wait 2-3 minutes for data to propagate
- Check time range in Grafana (set to "Last 15 minutes")
- Verify the Ray cluster name in the "Cluster" dropdown
- Confirm Ray metrics endpoint is responding (see verification steps above)

**Need to update the Ray service target?**
Edit the script `observability/add-ray-metrics.sh` to change the service name, then re-run it.

**Want to scrape multiple Ray clusters?**
You can add multiple entries to `customServiceScrapeTargets`. Edit the ObservabilityConfig:
```bash
kubectl edit observabilityconfig hyperpod-observability -n hyperpod-observability
```

Add additional targets under `spec.customServiceScrapeTargets`:
```yaml
customServiceScrapeTargets:
  - target: "cluster1-head-svc.default.svc.cluster.local:8080"
    metricsPath: "/metrics"
    scrapeInterval: 30
  - target: "cluster2-head-svc.default.svc.cluster.local:8080"
    metricsPath: "/metrics"
    scrapeInterval: 30
```

## Dashboard Preview

![Ray Dashboard](img/ray-dashboard.png)

