<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Cosmos 3 Observability — Goodput Dashboard & Prerequisites

This directory ships a reproducible **goodput metrics harness** for the Cosmos 3
sample on Amazon SageMaker HyperPod (EKS): a Grafana dashboard plus the steps to
wire up the two metric sources behind it.

- [`cosmos3-goodput-dashboard.json`](./cosmos3-goodput-dashboard.json) — an
  importable Grafana dashboard model (Amazon Managed Grafana compatible) titled
  **"Cosmos 3 Goodput (AWS HyperPod)"**.

## What you get

The dashboard unifies **two metric sources** into two rows, sharing a single
Prometheus (Amazon Managed Prometheus / AMP) datasource template variable
(`${DS_PROMETHEUS}`):

| Row | Source | Metrics → Panels |
| --- | --- | --- |
| **Trainer** | Shipped `PrometheusCallback` (`src/cosmos3_aws/observability/prometheus_callback.py`) → Pushgateway → AMP | `cosmos3_loss` (loss), `cosmos3_step_time_seconds` (step time), `cosmos3_iteration` (stat) |
| **GPU / infra** | HyperPod observability addon (DCGM exporter + node exporters) → AMP | `DCGM_FI_PROF_SM_ACTIVE` (saturation headline), `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_POWER_USAGE`, PCIe/NVLink traffic |

> **Saturation headline = `DCGM_FI_PROF_SM_ACTIVE`.** This is the trustworthy
> GPU-saturation signal and is the headline panel. **MFU is intentionally NOT
> shown** — it is still under validation (an open investigation), so we don't
> report a number we can't yet stand behind.

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

The shipped `PrometheusCallback` activates when the env var
`PROMETHEUS_PUSHGATEWAY_URL` is set on the training pod. With it set, the
callback pushes `cosmos3_loss`, `cosmos3_step_time_seconds`, and
`cosmos3_iteration` to the Pushgateway, which is then scraped into AMP and
unified with the GPU/DCGM metrics.

```bash
# on the training pod / job spec
export PROMETHEUS_PUSHGATEWAY_URL="http://<pushgateway-host>:9091"
```

**Fallback (no gateway):** run with `wandb_mode=offline` — the same trainer
metrics are captured locally for later inspection.

### 4. Import the dashboard into Amazon Managed Grafana

1. Open your Amazon Managed Grafana workspace → **Dashboards → New → Import**.
2. Upload [`cosmos3-goodput-dashboard.json`](./cosmos3-goodput-dashboard.json).
3. When prompted for the `DS_PROMETHEUS` input, select your **AMP** datasource.
4. Save. The two rows populate once both metric sources are flowing.

### 5. Experiment tracking (out of scope here)

For experiment tracking, SageMaker MLflow integrates separately via the
Terraform `enable_mlflow` variable. It is **out of scope** for this goodput
observability harness — pointer only.
