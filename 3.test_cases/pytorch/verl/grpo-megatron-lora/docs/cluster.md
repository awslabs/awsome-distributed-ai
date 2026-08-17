# Cluster & Setup Reference

Cluster topology (compute, Ray, storage, networking, monitoring), how to connect, and the
detailed workload-setup path that the README's Quickstart condenses.

> **Placeholders.** Every `${VAR}` below is a name `env_vars.example` already defines:
> `ACCOUNT`, `AWS_REGION`, `REGISTRY`, `EKS_CLUSTER_NAME`, `KUBE_NAMESPACE`,
> `S3_BUCKET_NAME`, `RAY_ADDRESS`, `MLFLOW_TRACKING_NAME` and the `FSX_*` group.
> `ACCOUNT` and `AWS_REGION` are derived from your AWS credentials automatically. Run
> `source env_vars` first and the commands below are copy-pasteable as-is.

This describes the reference cluster the runs in [results.md](results.md) were executed on.
Nothing here is required -- any EKS cluster with GPU nodes, KubeRay and a shared filesystem
works. Cluster name, account and region come from `env_vars` (`EKS_CLUSTER_NAME`, `ACCOUNT`,
`AWS_REGION`).

The underlying infrastructure (VPC, EKS, Karpenter, FSx, the KubeRay operator and the MLflow
tracking server) was provisioned separately with Terraform/Terragrunt + ArgoCD; that IaC is
outside this repo.

## Cluster Overview

| Component | Details |
|-----------|---------|
| **EKS Cluster** | `${EKS_CLUSTER_NAME}` (K8s 1.33) |
| **Account** | `${ACCOUNT}` (workload) |
| **Region** | `${AWS_REGION}` |
| **VPC CIDR** | `10.2.0.0/16` |
| **Availability Zones** | us-east-1a, us-east-1b, us-east-1c |

## Compute

| Node Group | Instance Type | Count | Purpose |
|------------|--------------|-------|---------|
| System | m5.xlarge | 2 (min) – 4 (max) | Control plane pods, Ray head, CoreDNS |
| GPU (Karpenter) | g6e.48xlarge | 0–8 workers (autoscaled) | Ray GPU workers, on-demand training |
| GPU (Karpenter) | p6-b200.48xlarge | 0–8 workers (autoscaled) | Large-scale GPU training |
| HyperPod GPU | ml.p6-b200.48xlarge | Reserved via Training Plan | Large-scale fault-tolerant training |

- **Karpenter** manages GPU node scaling (NodePool `gpu-training`, up to 256 NVIDIA GPUs)
- **HyperPod** (optional) can overlay the same EKS cluster
- Deep health checks enabled: InstanceStress, InstanceConnectivity

## Ray Cluster

| Setting | Value |
|---------|-------|
| **KubeRay Operator** | v1.3.0 (Helm chart) |
| **Ray Version** | 2.53.0 |
| **Namespace** | `default` |
| **Head Node** | m5.24xlarge, 8 CPU / 32Gi requests, 16 CPU / 64Gi limits, no GPUs |
| **Head Image** | `rayproject/ray:2.53.0-py312` |
| **GPU Workers (g6e)** | 0–8 replicas, 8 GPUs each |
| **GPU Workers (p6-b200)** | 0–8 replicas, 8 GPUs each |
| **Autoscaler** | Enabled, 300s idle timeout |
| **Shared Memory** | 16Gi (head), 512Gi (workers, for NCCL) |
| **Service Account** | `default:ray-cluster` (IAM role via ArgoCD annotations) |

**Ray S3 access**: The Ray service account has `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`,
`s3:ListBucket` on the S3 data replica bucket for checkpoints and distributed data loading.

## Storage

| Type | Capacity | Mount | Purpose |
|------|----------|-------|---------|
| **FSx for Lustre** | 4,800 GiB | `/fsx` (via PVC `fsx-lustre`) | Training data, model weights, checkpoints |

### Provisioning the FSx volume

`lustre/` holds the StorageClass, PV and PVC as `envsubst` templates. Create the FSx
filesystem first, read its identifiers off the API, put them in `env_vars`, then apply:

```bash
aws fsx describe-file-systems \
  --query 'FileSystems[].{Id:FileSystemId,DNS:DNSName,Mount:LustreConfiguration.MountName}'

# FSX_FILESYSTEM_ID, FSX_MOUNT_NAME, FSX_SUBNET_ID, FSX_SECURITY_GROUP_ID,
# FSX_STORAGE_CAPACITY -> env_vars
source env_vars

envsubst < lustre/storageclass.yaml | kubectl apply -f -
envsubst < lustre/pv.yaml           | kubectl apply -f -
envsubst < lustre/pvc.yaml          | kubectl apply -f -

kubectl get pvc fsx-lustre -n "${KUBE_NAMESPACE}"   # must reach Bound
```

Two things that will silently not work if they are wrong:

- **`mountname` is not the filesystem id.** It is `LustreConfiguration.MountName`, a short
  opaque string. Using the `fs-...` id here mounts nothing.
- **The subnet and security group must match your EKS worker nodes.** Otherwise the volume
  provisions successfully and then never mounts.

The PVC is named **`fsx-lustre`** because that is the `claimName` every manifest in
`kubernetes/` mounts. If you rename it, rename it there too.

- FSx has a Data Repository Association (DRA) to `s3://${S3_BUCKET_NAME}` at `/fsx/data/`
- Directory structure:
  ```
  /fsx/data/verl/
  ├── models/
  │   └── Qwen3-235B-A22B/             # Model weights (~470 GB)
  ├── data/
  │   ├── eurus/                        # Eurus math+code (480K train, 2K val)
  │   ├── apps/                         # APPS coding (4.8K train, 5K val)
  │   ├── taco/                         # TACO coding (25K train, 1K val)
  │   ├── codecontests/                 # CodeContests (9.8K train, 1K val)
  │   └── mixed-code-math/             # 70/30 code/math mix (59K train, 8K val)
  ├── ckpts/
  │   ├── eurus/megatron/...           # Eurus-only run checkpoints
  │   └── mixed-code-math/megatron/... # Mixed dataset run checkpoints
  ├── profiling/                        # PyTorch profiler traces
  └── cache/                            # pip, huggingface, triton caches
  ```

## Networking

- **Private subnets**: `10.2.0.0/18` (a), `10.2.64.0/18` (b), `10.2.192.0/18` (c)
- **Public subnets**: `10.2.128.0/24` (a), `10.2.129.0/24` (b), `10.2.130.0/24` (c)
- **NAT Gateways**: One per AZ (HA)
- **EFA**: Self-referencing security group for all-traffic (multi-node GPU training)
- **Placement Groups**: Per-AZ cluster placement for low-latency GPU-to-GPU
- **VPC Endpoints**: S3 (gateway), ECR API + DKR (interface)

## Connecting to the Cluster

```bash
# Configure kubeconfig
aws eks update-kubeconfig \
  --name "${EKS_CLUSTER_NAME}" \
  --region us-east-1 \
  --role-arn arn:aws:iam::${ACCOUNT}:role/TerraformExecutionRole

# Verify
kubectl get nodes
kubectl get raycluster -n default
```

## Monitoring

### Prometheus & Grafana

Ray metrics are scraped by a dedicated Prometheus instance deployed alongside the Ray cluster,
with Grafana for visualization.

The Ray Cluster Overview dashboard shows:
- **Active Nodes** — head + worker count
- **Running/Pending Tasks** — Ray task queue status
- **CPU & Memory Utilization** — per-node breakdown
- **GPU Utilization & GPU Memory** — via DCGM exporter metrics
- **Object Store Memory** — Ray plasma store usage

### Ray Dashboard

The Ray dashboard provides job submission, logs, and actor/task details.

| Access Method | URL |
|---------------|-----|
| **Web UI** | `${RAY_ADDRESS}` (`http://localhost:8265` with a port-forward) |
| **Jobs API** | `${RAY_ADDRESS}/api/jobs/` |

### CloudWatch

Container Insights is enabled via EKS add-on, providing node/pod-level metrics
and logs in the CloudWatch console under your `${EKS_CLUSTER_NAME}` cluster.

---

---

## Workload setup (detailed)

The README Quickstart is the condensed path. This is the full version, including the
alternate data-prep routes and the MLflow wiring.

## Deploy Sandbox Fusion (Code Execution for Reward Scoring)

Sandbox Fusion is a remote code sandbox service that securely executes model-generated code
during reward scoring. It replaces local code execution (which requires `pyext`, incompatible
with Python 3.12) and provides a 10-30% reduction in reward stage time by leveraging dedicated
CPU resources for concurrent code verification.

It runs as a standalone Kubernetes service on a dedicated CPU node, independent of the
RayCluster. verl connects to it via HTTP during the reward computation phase.

```bash
source env_vars

# Deploy Sandbox Fusion service
envsubst < kubernetes/sandbox-fusion.yaml | kubectl apply -f -

# Wait for it to be ready
kubectl rollout status deploy/sandbox-fusion

# Verify from the Ray head pod
kubectl exec <ray-head-pod> -c ray-head -- \
  wget -qO- --header='Content-Type: application/json' \
  --post-data='{"code":"print(42)","language":"python"}' \
  http://sandbox-fusion.${KUBE_NAMESPACE}.svc.cluster.local:8080/run_code
# Expected: {"status":"Success","run_result":{"stdout":"42\n",...}}
```

The sandbox URL and settings are configured in the Hydra config group `conf/sandbox/`
(`enabled.yaml` / `disabled.yaml`). `submit_training.py` conditionally includes the
sandbox reward function args when `sandbox.enabled` is `true` (the default). To disable:

```bash
python3 scripts/submit_training.py sandbox=disabled
```

## Deploy FSx Utility Pod

Data preparation and model download scripts need a pod with the FSx filesystem mounted
at `/fsx`. The **fsx-utils** pod is a lightweight, persistent pod that serves this
purpose. It runs independently of the Ray cluster, so you can prepare data and download
models before the cluster is up, or after it's been torn down.

```bash
source env_vars

# Deploy the FSx utility pod
envsubst < kubernetes/fsx-utils.yaml | kubectl apply -f -

# Wait for it to be ready (~30-60s for pip install on first boot)
kubectl wait --for=condition=Ready pod/fsx-utils -n ${KUBE_NAMESPACE} --timeout=180s
```

The pod uses a minimal `python:3.12-slim` image with `datasets` and `huggingface_hub`
installed at startup. It mounts the same `fsx-claim` PVC used by the Ray cluster.

To tear it down when no longer needed:
```bash
kubectl delete pod fsx-utils -n ${KUBE_NAMESPACE}
```

## Prepare Data

Data preparation downloads datasets from HuggingFace, transforms them into verl-compatible
parquet format, and optionally mixes multiple datasets with configurable ratios.

**Supported datasets**: `eurus` (math+code reasoning, 480K), `apps` (competitive programming,
4.8K), `taco` (coding, 25K), `codecontests` (Code-Contests-Plus, 9.8K).

### Option A: All-in-one Ray job (recommended)

Use `submit_data_prep.sh` to prepare and mix datasets in a single Ray job. This runs on
the cluster with FSx access and HuggingFace credentials.

```bash
source env_vars

# Prepare individual datasets (written to /fsx/data/verl/data/{name}/)
./data/submit_data_prep.sh --datasets eurus apps taco codecontests \
    --output-dir /fsx/data/verl/data

# Prepare + mix in one shot (70/30 code/math split)
./data/submit_data_prep.sh \
    --datasets eurus apps taco codecontests \
    --output-dir /fsx/data/verl/data \
    --mix --mix-output /fsx/data/verl/data/mixed-code-math \
    --mix-ratios eurus:0.04 apps:1.0 taco:1.0 codecontests:1.0
```

The `--mix-ratios` control sampling: `1.0` uses all samples, `0.04` samples 4% (e.g.,
~19K from Eurus's 480K). Datasets are prepared in parallel as Ray remote tasks, then
mixed and shuffled on the driver.

### Option B: Two-step (prepare then mix separately)

If the all-in-one mix step fails (e.g., HF cache permission errors), prepare datasets
first, then mix on the `fsx-utils` pod:

```bash
# Step 1: Prepare datasets via Ray job
./data/submit_data_prep.sh --datasets apps taco codecontests \
    --output-dir /fsx/data/verl/data

# Step 2: Mix on fsx-utils pod (has Python + datasets package)
kubectl cp data/mix_datasets.py fsx-utils:/tmp/mix_datasets.py
kubectl exec fsx-utils -- python3 /tmp/mix_datasets.py \
    --datasets \
        /fsx/data/verl/data/taco/train.parquet:1.0 \
        /fsx/data/verl/data/apps/train.parquet:1.0 \
        /fsx/data/verl/data/codecontests/train.parquet:1.0 \
        /fsx/data/verl/data/eurus/train.parquet:0.04 \
    --val-files \
        /fsx/data/verl/data/taco/val.parquet:1.0 \
        /fsx/data/verl/data/apps/val.parquet:1.0 \
        /fsx/data/verl/data/codecontests/val.parquet:1.0 \
        /fsx/data/verl/data/eurus/val.parquet:0.5 \
    --output /fsx/data/verl/data/mixed-code-math \
    --seed 42
```

### Verify prepared data

```bash
kubectl exec fsx-utils -- ls -lh /fsx/data/verl/data/mixed-code-math/
# Expected: train.parquet (~500MB), val.parquet (~100MB)
```

All datasets use the verl-compatible parquet format with columns: `data_source`, `prompt`,
`ability`, `reward_model`, `extra_info`. The `data_source` field routes each sample to the
correct reward scorer (code execution for `apps`/`taco`/`codecontests`, math verification
for `numina_*` sources).

## Download Models to FSx (Recommended)

By default, each Ray worker downloads model weights from HuggingFace Hub independently
at training startup. For large models this takes 5-15+ minutes and saturates network
bandwidth across all nodes simultaneously. Pre-downloading the model to the shared FSx
filesystem eliminates this -- all workers load from local storage in seconds.

```bash
source env_vars

# Option A: Ray job submission (recommended -- runs on Ray cluster)
./models/submit_download.sh Qwen/Qwen3-8B /fsx/data/verl/models/Qwen3-8B

# Option B: On the fsx-utils pod (for models that don't need Ray)
./models/run_on_cluster.sh models/download_qwen3_coder_next.sh
./models/run_on_cluster.sh models/download_qwen25_coder_7b.sh
./models/run_on_cluster.sh models/download_qwen3_30b_a3b.sh
```

Models are saved to `${RAY_DATA_HOME}/models/<MODEL_NAME>` (e.g.,
`/fsx/data/verl/models/Qwen3-Coder-Next`). The training scripts automatically detect local
models on FSx and use them instead of downloading from HuggingFace Hub -- no configuration
change needed.

To verify, you can run:
```bash
kubectl exec -n ${KUBE_NAMESPACE} fsx-utils -- ls -lh /fsx/data/verl/models/Qwen3-Coder-Next/
```

## Reaching the Ray Dashboard (CLI)

The `ray job` CLI talks to the Ray head's dashboard on port 8265. How you reach it depends
on how your cluster exposes it.

### Default: port-forward (works on any cluster)

```bash
# Find the Ray head service, then forward it locally
kubectl get svc -n "${KUBE_NAMESPACE:-default}" | grep ray
kubectl port-forward -n "${KUBE_NAMESPACE:-default}" svc/<ray-head-svc> 8265:8265

# In another shell
export RAY_ADDRESS="http://localhost:8265"
ray job list
```

This is what `env_vars.example` assumes (`RAY_ADDRESS="http://localhost:8265"`), and it
needs no authentication headers — leave `RAY_HEADERS` unset.

If instead you reach the dashboard through an authenticating proxy (for example an ALB with
OIDC), export `RAY_HEADERS` as a JSON object of headers to send with every request, and pass
`--headers "$RAY_HEADERS"` to the `ray job` CLI. The submit scripts in `scripts/` and `data/`
forward it automatically when it is set.

```bash
export RAY_ADDRESS="https://<your-ray-dashboard-host>"
export RAY_HEADERS='{"Cookie": "<session-cookie>"}'
```

## SageMaker Managed MLflow

MLflow tracks training metrics (loss, reward, KL divergence, throughput) and displays
them in the MLflow dashboard. The tracking server is provisioned outside this repo (the
reference setup used Terraform); this repo only reads `MLFLOW_TRACKING_ARN`. MLflow is
optional -- use `tracking=console` to skip it entirely.

| Component | Value |
|-----------|-------|
| **Tracking Server** | `${MLFLOW_TRACKING_NAME}` |
| **URL** | (presigned -- see below) |
| **ARN** | `arn:aws:sagemaker:${AWS_REGION}:${ACCOUNT}:mlflow-tracking-server/${MLFLOW_TRACKING_NAME}` |
| **MLflow Version** | 3.0.0 |
| **Artifact Bucket** | `s3://<mlflow-artifact-bucket>` (KMS encrypted) |
| **Auth** | EKS Pod Identity via `ray-cluster` ServiceAccount (automatic) |

**Authentication**: The `ray-cluster` ServiceAccount in the `kuberay` namespace is
configured with EKS Pod Identity, so all Ray pods automatically have credentials to
talk to the MLflow tracking server. No manual credential setup is needed.

The `MLFLOW_TRACKING_ARN` environment variable is automatically injected into all Ray
pods by the Ray Helm chart. Training scripts use this to connect to MLflow:

```python
import os
import mlflow

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_ARN'])
mlflow.autolog()

with mlflow.start_run():
    mlflow.log_param("model", "Qwen3-8B")
    mlflow.log_metric("loss", 0.5)
```

**Access the MLflow UI** by generating a presigned URL:

```bash
aws sagemaker create-presigned-mlflow-tracking-server-url \
  --tracking-server-name "${MLFLOW_TRACKING_NAME}" \
  --session-expiration-duration-in-seconds 1800 \
  --expires-in-seconds 300 \
  --region us-east-1
```

The `MLFLOW_TRACKING_URI` is read from the environment via Hydra's
`${oc.env:MLFLOW_TRACKING_URI}` resolver (set in `env_vars` or exported directly).
Training metrics are automatically logged -- the Hydra config group `tracking=mlflow`
(the default) sets `trainer.logger='["console","mlflow"]'`, which uses verl's built-in
`Tracking` class. `trainer.project_name` maps to the MLflow **Experiment** name, and
`trainer.experiment_name` maps to the MLflow **Run** name. To disable MLflow and log
to console only:

```bash
python3 scripts/submit_training.py tracking=console
```
