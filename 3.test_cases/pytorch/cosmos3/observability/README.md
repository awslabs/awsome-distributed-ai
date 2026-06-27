<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Cosmos 3 Observability — Goodput Dashboard & Prerequisites

This directory ships a reproducible **goodput metrics harness** for the Cosmos 3
sample on Amazon SageMaker HyperPod (EKS): a Grafana dashboard plus the steps to
wire up the two metric sources behind it.

- [`cosmos3-goodput-dashboard.json`](./cosmos3-goodput-dashboard.json) — an
  importable Grafana dashboard model (Amazon Managed Grafana compatible) titled
  **"Cosmos 3 Goodput (AWS HyperPod)"**.

> **Validated on:** both trainer-metrics export paths below were functionally
> validated on a **2× g6.8xlarge HyperPod-EKS cluster (us-east-1)** — DCGM/GPU
> metrics and both `cosmos3_*` sources were observed together in a single AMP
> workspace and unified in one Grafana pane.

## What you get

The dashboard unifies **two metric sources** into two rows, sharing a single
Prometheus (Amazon Managed Prometheus / AMP) datasource template variable
(`${DS_PROMETHEUS}`):

| Row | Source | Metrics → Panels |
| --- | --- | --- |
| **Trainer** | Shipped callbacks (see [How trainer metrics reach AMP](#how-trainer-metrics-reach-amp)) → AMP | `cosmos3_loss` (loss), `cosmos3_step_time_seconds` (step time), `cosmos3_iteration` (stat) |
| **GPU / infra** | HyperPod observability addon (DCGM exporter + node exporters) → AMP | `DCGM_FI_PROF_SM_ACTIVE` (saturation headline), `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_POWER_USAGE`, PCIe/NVLink traffic |

> **Saturation headline = `DCGM_FI_PROF_SM_ACTIVE`.** This is the trustworthy
> GPU-saturation signal and is the headline panel. **MFU is intentionally NOT
> shown** — it is still under validation (an open investigation), so we don't
> report a number we can't yet stand behind.

## How trainer metrics reach AMP

The trainer metrics (`cosmos3_loss`, `cosmos3_step_time_seconds`,
`cosmos3_iteration`) are emitted by the **shipped callbacks** and activated by
env vars in the launcher (see [`env_vars.example`](../env_vars.example)). There
are **two validated paths** to land them in AMP, unified with the addon's
DCGM/GPU metrics.

> **Important:** callback metrics do **not** reach AMP automatically. The
> HyperPod observability addon's central collector keeps a **fixed** scrape job
> set (node-exporter, dcgm-exporter, kube-state-metrics, training-operator) and
> does **not** scrape arbitrary services. You must use one of the paths below.

| | **Path B — OTLP direct** *(recommended default)* | **Path A — Pushgateway** *(alternative)* |
| --- | --- | --- |
| Callback | `OTLPCallback` | `PrometheusCallback` |
| Activation env | `OTEL_EXPORTER_OTLP_ENDPOINT` | `PROMETHEUS_PUSHGATEWAY_URL` |
| Extra deploy | none | `observability/pushgateway.yaml` |
| Edit managed addon? | **No** | **Yes** — add a scrape job to the central-collector ConfigMap |
| Durability | survives addon upgrades | fragile — addon reconcile/upgrade can revert the edit |
| Use when | addon exposes an OTLP receiver (default) | no OTLP receiver available |

Optional env for either path: `PROMETHEUS_JOB_NAME` (default `cosmos3`),
`PROMETHEUS_EVERY_N` (push/export every N steps).

### Path B — OTLP direct (recommended default)

The addon's central collector **already** exposes an OTLP receiver wired into
the same remote-write→AMP pipeline. No Pushgateway, no collector-config edit —
this is more durable across addon upgrades. The OpenTelemetry deps
(`opentelemetry-sdk` + OTLP grpc/http exporters) are already baked into the
sample Dockerfile.

```bash
# on the training pod / job spec
export OTEL_EXPORTER_OTLP_ENDPOINT="http://hyperpod-otel-collector.hyperpod-observability.svc:4317"
export OTEL_EXPORTER_OTLP_PROTOCOL="grpc"   # grpc (4317) or http (4318)
```

The OTLP receiver lives in the `hyperpod-observability` namespace as service
`hyperpod-otel-collector`, ports `4317` (grpc) / `4318` (http).

### Path A — Pushgateway (alternative)

1. Deploy the shipped Pushgateway (Deployment + Service in namespace
   `cosmos3-obs`, service `pushgateway:9091`):

   ```bash
   kubectl apply -f observability/pushgateway.yaml
   ```

2. Point the training pods at it:

   ```bash
   export PROMETHEUS_PUSHGATEWAY_URL="http://pushgateway.cosmos3-obs.svc:9091"
   ```

3. **Required glue:** add a scrape job so the addon's **central** collector
   scrapes the Pushgateway. Edit the `hyperpod-observability-central-collector-config`
   ConfigMap's `collector.yaml`, append the following under `scrape_configs:`,
   then restart `deploy/hyperpod-observability-central-collector`:

   ```yaml
           - job_name: cosmos3-pushgateway
             scrape_interval: 15s
             honor_labels: true
             kubernetes_sd_configs:
               - role: service
             relabel_configs:
               - action: keep
                 regex: pushgateway
                 source_labels:
                   - __meta_kubernetes_service_name
               - target_label: cluster_name
                 replacement: <your-cluster-name>
   ```

> **Downside (be honest):** editing the addon's **managed** ConfigMap is fragile
> — an addon upgrade/reconcile can revert it. Prefer Path B if you want to avoid
> editing managed resources.

**Fallback (no gateway, no OTLP):** run with `wandb_mode=offline` — the same
trainer metrics are captured locally for later inspection.

## Prerequisites & setup

### 1. Enable the HyperPod observability addon (Terraform)

Use the existing IaC at
[`1.architectures/7.sagemaker-hyperpod-eks/terraform-modules/hyperpod-eks-tf/`](../../../../1.architectures/7.sagemaker-hyperpod-eks/terraform-modules/hyperpod-eks-tf/).
Do **not** re-ship infra — set these variables:

```hcl
create_observability_module = true   # installs amazon-sagemaker-hyperpod-observability addon + AMP + Managed Grafana
create_prometheus_workspace = true   # creates the AMP workspace (default)
enable_gpu_operator         = false  # IMPORTANT: leave false on the observability path
```

- **Confirm your region is AMP-allowed.** `create_observability_module` is gated
  on an AMP-supported region. Allowed regions include: `us-east-1`, `us-east-2`,
  `us-west-1`, `us-west-2`, `ap-south-1`, `ap-northeast-1`,
  `ap-southeast-1/2/3/4`, `eu-central-1`, `eu-west-1/2`, `eu-north-1`,
  `eu-south-2`, `sa-east-1`.
- **Leave `enable_gpu_operator = false`.** The observability addon bundles its
  **own** DCGM exporter. Enabling the GPU operator stands up a **second,
  duplicate** DCGM exporter and produces conflicting/duplicated GPU series.

### 2. Confirm the Kubeflow Training Operator is present

HyperPod managed auto-resume relies on the Kubeflow Training Operator
(**1.7.0 / 1.8.0 / 1.8.1**). This is already a sample prerequisite — see the
[hyperpod-eks README](../hyperpod-eks/README.md) for the install/verify steps.

### 3. Activate trainer metrics

Pick **Path B (OTLP, recommended)** or **Path A (Pushgateway)** above and set
the corresponding env var(s) on the training pod. See
[How trainer metrics reach AMP](#how-trainer-metrics-reach-amp).

### 4. Import the dashboard into Amazon Managed Grafana

1. Open your Amazon Managed Grafana workspace → **Dashboards → New → Import**.
2. Upload [`cosmos3-goodput-dashboard.json`](./cosmos3-goodput-dashboard.json).
3. When prompted for the `DS_PROMETHEUS` input, select your **AMP** datasource.
4. Save. The two rows populate once both metric sources are flowing.

### 5. Experiment tracking (out of scope here)

For experiment tracking, SageMaker MLflow integrates separately via the
Terraform `enable_mlflow` variable. It is **out of scope** for this goodput
observability harness — pointer only.
