# Observability for SageMaker HyperPod Slurm Clusters

This standalone sample installs a metrics collection and export pipeline on
SageMaker HyperPod Slurm clusters. It deploys per-node metric exporters and an
OpenTelemetry (OTel) collector that ships metrics to
[Amazon Managed Service for Prometheus (AMP)](https://docs.aws.amazon.com/prometheus/latest/userguide/what-is-Amazon-Managed-Service-Prometheus.html),
where they can be visualized with
[Amazon Managed Grafana](https://docs.aws.amazon.com/grafana/latest/userguide/what-is-Amazon-Managed-Service-Grafana.html).

## Architecture

Each cluster node runs its own set of metric exporters and an OTel collector.
The OTel collector scrapes the local exporters and remote-writes metrics to
AMP using SigV4 authentication (via the cluster execution role). There is no
central Prometheus server -- each node independently ships its own metrics.

```
  Controller Node              Compute Node (GPU)          Login Node
  +------------------+        +------------------+        +------------------+
  | Node Exporter    |        | Node Exporter    |        | Node Exporter    |
  | Slurm Exporter   |        | DCGM Exporter    |        +--------+---------+
  +--------+---------+        | EFA Exporter     |                 |
           |                  +--------+---------+                 |
           v                           |                           v
  +------------------+                 |                  +------------------+
  | OTel Collector   |                 |                  | OTel Collector   |
  +--------+---------+                 |                  +--------+---------+
           |                           |                           |
           |    +----------------------+                           |
           |    |                                                  |
           |    |  (When NCCL metrics enabled)                     |
           |    |  +---------------------------+                   |
           |    |  | Slurm Job (GPU training)  |                   |
           |    |  |  NCCL Inspector Plugin    |                   |
           |    |  |    |                      |                   |
           |    |  |    v                      |                   |
           |    |  |  textfile (.prom files)   |                   |
           |    |  |  /var/lib/node_exporter/  |                   |
           |    |  |  nccl_inspector/          |                   |
           |    |  +----------+----------------+                   |
           |    |             |                                    |
           |    |             v (textfile collector)               |
           |    +---> Node Exporter                                |
           |                  |                                    |
           |                  v                                    |
           |         +------------------+                          |
           |         | OTel Collector   |                          |
           |         +--------+---------+                          |
           |                  |                                    |
           +--------+---------+------------------------------------+
                    |
                    v
          +-------------------+
          | Amazon Managed    |
          | Prometheus (AMP)  |
          +--------+----------+
                   |
                   v
          +-------------------+
          | Amazon Managed    |
          | Grafana           |
          +-------------------+
```

## What gets installed

The stack is node-type aware. The entrypoint script (`setup_observability.sh`)
auto-detects whether a node is a controller, compute, or login node by
checking which Slurm daemon is running, then installs the appropriate
exporters.

| Component | Controller | Compute | Login | Port | Description |
| --- | :---: | :---: | :---: | --- | --- |
| Node Exporter | Y | Y | Y | 9100 | OS-level metrics (CPU, memory, disk, network) |
| DCGM Exporter | | Y | | 9400 | NVIDIA GPU metrics with Slurm job-ID mapping |
| EFA Exporter | | Y | | 9109 | Elastic Fabric Adapter network metrics |
| Slurm Exporter | Y | | | 9341 | Slurm scheduler metrics (jobs, nodes, partitions) |
| OTel Collector | Y | Y | Y | 4317/4318 | Scrapes all local exporters and remote-writes to AMP |

Node type detection logic:
- `slurmctld` running -> controller
- `slurmd` running + hostname in `sinfo` output -> compute
- `slurmd` running + hostname NOT in `sinfo` output -> login

On compute nodes, DCGM Exporter gracefully skips installation if no NVIDIA
GPU is detected (CPU-only compute nodes).

### GPU-to-job-ID mapping (DCGM + Slurm Prolog/Epilog)

When a Slurm job allocates GPUs, it's useful to know which job is using
which GPU so that GPU metrics (temperature, utilization, errors) can be
attributed to specific training jobs. This is handled by Slurm
Prolog/Epilog scripts that the HyperPod AMI provides automatically.

The flow works like this:

1. A user submits a GPU job (e.g., `sbatch --gres=gpu:1 ...`)
2. Slurm allocates GPUs and runs the **Prolog** script before the job starts
3. The Prolog writes the job ID to a mapping file named after the GPU index
   (e.g., `/run/slurm/dcgm-job-mapping/0` contains `"42"` for job 42)
4. DCGM Exporter reads this directory and adds an `hpc_job` label to all
   GPU metrics for that device
5. When the job ends, the **Epilog** script removes the job ID from the
   mapping file

The result is that DCGM metrics in AMP/Grafana include the `hpc_job` label:

```
DCGM_FI_DEV_GPU_TEMP{gpu="0",...,hpc_job="42"} 27
```

This lets you filter GPU dashboards by job ID, correlate GPU utilization
with specific training runs, and identify which jobs are causing thermal
or error issues.

The Prolog/Epilog scripts are provided by the HyperPod AMI at:
- `/opt/slurm/etc/prolog.d/700_dcgm_job_map_register.sh`
- `/opt/slurm/etc/epilog.d/700_dcgm_job_map_cleanup.sh`

They use file locking (`flock`) for safe concurrent access and are designed
to never return non-zero exit codes -- a prolog failure would block the job
from starting, and an epilog failure would drain the node, both of which
are too disruptive for a metrics-only feature.

> **Note:** When using AMI-based configuration (`OnInitComplete`), the
> Prolog/Epilog scripts and the mapping directory are set up automatically.
> No action is needed. When using `OnCreate` with custom lifecycle scripts,
> ensure your `start_slurm.sh` configures `Prolog` and `Epilog` in
> `slurm.conf` and that the DCGM Exporter mounts
> `/run/slurm/dcgm-job-mapping` (with hyphens, not underscores).

## Prerequisites

### 1. Enable IAM Identity Center

IAM Identity Center is required for Amazon Managed Grafana authentication.
Follow the instructions in
[Enabling IAM Identity Center](https://docs.aws.amazon.com/singlesignon/latest/userguide/get-set-up-for-idc.html)
and create an admin user account.

### 2. Create the observability infrastructure

Deploy the CloudFormation stack that provisions an AMP workspace and an
Amazon Managed Grafana workspace:

```bash
wget https://raw.githubusercontent.com/aws-samples/awsome-distributed-training/main/4.validation_and_observability/4.prometheus-grafana/cluster-observability.yaml

aws cloudformation create-stack \
    --stack-name hyperpod-observability \
    --template-body file://cluster-observability.yaml \
    --capabilities CAPABILITY_NAMED_IAM
```

After the stack completes, note the `AMPRemoteWriteURL` from the outputs:

```bash
aws cloudformation describe-stacks \
    --stack-name hyperpod-observability \
    --query "Stacks[0].Outputs[?OutputKey=='AMPRemoteWriteURL'].OutputValue" \
    --output text
```

This URL goes into `config.json` as `prometheus_remote_write_url`.

If you already have existing AMP and Grafana workspaces, pass their IDs:

```bash
aws cloudformation create-stack \
    --stack-name hyperpod-observability \
    --template-body file://cluster-observability.yaml \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameters \
        ParameterKey=PrometheusExistingWorkspaceId,ParameterValue=ws-abc123 \
        ParameterKey=GrafanaExistingWorkspaceId,ParameterValue=g-xyz789
```

Alternatively, you can create just an AMP workspace without Grafana:

```bash
aws amp create-workspace --alias hyperpod-observability \
    --tags Environment=dev
```

The remote write URL is:
`https://aps-workspaces.<region>.amazonaws.com/workspaces/<workspace-id>/api/v1/remote_write`

### 3. Add required permissions to the cluster execution role

The HyperPod cluster execution role needs two additional policies beyond
the base `AmazonSageMakerClusterInstanceRolePolicy`:

**a) AMP remote write** -- Required for the OTel Collector to ship metrics
to Amazon Managed Prometheus:

```bash
aws iam put-role-policy \
    --role-name <your-execution-role> \
    --policy-name AMPRemoteWritePolicy \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["aps:RemoteWrite"],
            "Resource": "arn:aws:aps:<region>:<account-id>:workspace/<workspace-id>"
        }]
    }'
```

**b) ECR image pull** -- Required for pulling the exporter container images
(Node Exporter, DCGM Exporter, EFA Exporter, OTel Collector) from the
HyperPod ECR registry (`602401143452.dkr.ecr.<region>.amazonaws.com`):

```bash
aws iam put-role-policy \
    --role-name <your-execution-role> \
    --policy-name ECRPullPolicy \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": [
                "ecr:BatchGetImage",
                "ecr:GetAuthorizationToken",
                "ecr:GetDownloadUrlForLayer"
            ],
            "Resource": "*"
        }]
    }'
```

> **Important:** Both policies are required. Without the ECR policy, the
> exporter container images cannot be pulled and the observability setup
> will fail. Without the AMP policy, metrics cannot be shipped to
> Prometheus.

### 4. Network requirements

- The VPC must allow outbound internet access (NAT gateway) for the
  controller node to download Go and build the Slurm Exporter from source.
- All nodes need HTTPS access to the AMP endpoint for remote write.
- All nodes need HTTPS access to ECR (`602401143452.dkr.ecr.<region>.amazonaws.com`)
  to pull the exporter container images.
- For SSM access to cluster nodes, create VPC endpoints for `ssm`,
  `ssmmessages`, and `ec2messages` in the cluster's VPC and subnet.

### 5. S3 bucket

Upload the observability scripts to an S3 bucket. The bucket name must start
with `sagemaker-` to be accessible by the
`AmazonSageMakerClusterInstanceRolePolicy` managed policy.

```bash
aws s3 sync ./observability s3://sagemaker-<your-bucket>/observability/
```

> **Important:** If editing scripts on Windows, ensure all files have LF
> (Unix) line endings before uploading. CRLF line endings will cause
> `$'\r': command not found` errors on Linux. The included `.gitattributes`
> file enforces LF endings when using Git.


## Usage

### Option A: OnInitComplete (recommended for new clusters)

Use this when creating a cluster with AMI-based configuration (no
`LifeCycleConfig` or `OnCreate`). HyperPod sets up Slurm and Docker
automatically, then runs your extension script.

1. Edit `config.json` with your AMP remote write URL:

   > **Important:** The `config.json` file ships with placeholder values
   > (`<region>`, `<workspace-id>`). You **must** replace
   > `prometheus_remote_write_url` with your actual AMP workspace URL
   > before uploading to S3. The setup script validates this at runtime
   > and will fail with an error if placeholder values are detected.
   > Cluster creation with `OnInitComplete` will fail if the script
   > exits non-zero.

   ```json
   {
       "prometheus_remote_write_url": "https://aps-workspaces.<region>.amazonaws.com/workspaces/<workspace-id>/api/v1/remote_write",
       "advanced_metrics": false,
       "nccl_metrics_enabled": false,
       "nccl_metrics_dump_interval_seconds": 30,
       "nccl_profiler_plugin_path": "/opt/aws/hyperpod/observability/lib/libnccl-profiler-inspector.so"
   }
   ```

2. Upload to S3:
   ```bash
   aws s3 sync ./observability s3://sagemaker-<your-bucket>/observability/
   ```

3. Create the cluster with `OnInitComplete` on each instance group. Here is
   a complete `CreateCluster` example:

   ```json
   {
       "ClusterName": "my-hyperpod-cluster",
       "InstanceGroups": [
           {
               "InstanceGroupName": "controller",
               "InstanceType": "ml.c5.xlarge",
               "InstanceCount": 1,
               "SlurmConfig": { "NodeType": "Controller" },
               "LifeCycleConfig": {
                   "OnInitComplete": "setup_observability.sh",
                   "SourceS3Uri": "s3://sagemaker-<your-bucket>/observability/"
               },
               "ExecutionRole": "arn:aws:iam::<account-id>:role/<execution-role>",
               "InstanceStorageConfigs": [
                   { "EbsVolumeConfig": { "VolumeSizeInGB": 500 } }
               ]
           },
           {
               "InstanceGroupName": "login",
               "InstanceType": "ml.m5.xlarge",
               "InstanceCount": 1,
               "SlurmConfig": { "NodeType": "Login" },
               "LifeCycleConfig": {
                   "OnInitComplete": "setup_observability.sh",
                   "SourceS3Uri": "s3://sagemaker-<your-bucket>/observability/"
               },
               "ExecutionRole": "arn:aws:iam::<account-id>:role/<execution-role>"
           },
           {
               "InstanceGroupName": "gpu-workers",
               "InstanceType": "ml.p4d.24xlarge",
               "InstanceCount": 4,
               "SlurmConfig": {
                   "NodeType": "Compute",
                   "PartitionNames": ["gpu-training"]
               },
               "LifeCycleConfig": {
                   "OnInitComplete": "setup_observability.sh",
                   "SourceS3Uri": "s3://sagemaker-<your-bucket>/observability/"
               },
               "ExecutionRole": "arn:aws:iam::<account-id>:role/<execution-role>"
           }
       ],
       "Orchestrator": {
           "Slurm": { "SlurmConfigStrategy": "Managed" }
       },
       "VpcConfig": {
           "SecurityGroupIds": ["sg-<your-sg>"],
           "Subnets": ["subnet-<your-subnet>"]
       },
       "NodeRecovery": "Automatic"
   }
   ```

   Submit with:
   ```bash
   aws sagemaker create-cluster --cli-input-json file://create_cluster.json
   ```

   > **Note:** `OnInitComplete` and `SlurmConfig` are new API fields. If
   > your AWS CLI version does not recognize them, update the CLI or use
   > boto3/SDK to call the API directly.

### Option B: Manual execution on a running cluster

Connect to each node via SSM and run the script. The SSM target format for
HyperPod nodes is:

```
sagemaker-cluster:<cluster-id>_<instance-group-name>-<instance-id>
```

Find the cluster ID and instance IDs:
```bash
aws sagemaker describe-cluster --cluster-name <cluster-name> --query "ClusterArn"
# Extract the ID after the last slash, e.g., "abc123def456"

aws sagemaker list-cluster-nodes --cluster-name <cluster-name>
```

Connect and run:
```bash
aws ssm start-session \
    --target sagemaker-cluster:<cluster-id>_<group-name>-<instance-id> \
    --region <region>

# Once connected:
sudo aws s3 sync s3://sagemaker-<your-bucket>/observability/ /tmp/observability/
sudo bash /tmp/observability/setup_observability.sh
```

### Option C: OnCreate (legacy lifecycle scripts)

If you are using full custom lifecycle scripts with `OnCreate`, call
`setup_observability.sh` from your `lifecycle_script.py` after Slurm has
been started. Add the following to your `lifecycle_script.py`:

```python
# After ExecuteBashScript("./start_slurm.sh").run(...)
# Install observability
if Config.enable_observability:
    ExecuteBashScript("./utils/install_docker.sh").run()
    subprocess.run(
        ["bash", "./observability/setup_observability.sh"],
        check=True
    )
```

Ensure the `observability/` directory is uploaded alongside your other
lifecycle scripts in the same S3 prefix.


## Configuration

All configuration is in `config.json`:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `prometheus_remote_write_url` | string | (required) | AMP workspace remote write endpoint |
| `advanced_metrics` | bool | `false` | Enable extended metrics (see below) |
| `nccl_metrics_enabled` | bool | `false` | Enable NCCL Inspector metrics via Slurm task prolog |
| `nccl_metrics_dump_interval_seconds` | int | `30` | NCCL metrics dump interval in seconds |
| `nccl_profiler_plugin_path` | string | `/opt/aws/hyperpod/observability/lib/libnccl-profiler-inspector.so` | Path to the NCCL Inspector `.so` plugin |

### Advanced metrics

When `advanced_metrics` is `true`:
- Node Exporter enables additional collectors: `cgroups`, `ksmd`,
  `meminfo_numa`, `ethtool`, `mountstats`, `network_route`, `processes`,
  `tcpstat`
- DCGM Exporter uses the advanced metrics CSV which adds: ECC errors,
  retired pages, NVLink error counters, per-link bandwidth, SM occupancy,
  FP16/FP32/FP64 pipe activity, and additional throttling violation counters
- EFA Exporter enables the `amazonefa` collector with detailed per-device
  metrics

### NCCL metrics

[NCCL (NVIDIA Collective Communications Library)](https://github.com/NVIDIA/nccl)
is the library that powers multi-GPU and multi-node communication in
distributed training. Operations like AllReduce, AllGather, and
ReduceScatter are NCCL collectives that move data between GPUs during
training. Monitoring NCCL performance is critical for identifying
communication bottlenecks in large-scale training jobs.

The [NCCL Inspector](https://github.com/NVIDIA/nccl/tree/master/plugins/profiler/inspector)
is an NCCL profiler plugin that hooks into NCCL collective and P2P
operations at runtime and exports per-operation performance metrics. When
configured in Prometheus mode, it writes metrics as textfiles that Node
Exporter scrapes and the OTel Collector ships to AMP.

#### How it works

The NCCL metrics data flow has these components:

1. **Slurm TaskProlog** (`/opt/slurm/etc/task_prolog.sh`) -- A script that
   Slurm runs before every task. It sets environment variables that tell
   NCCL to load the Inspector plugin and output Prometheus-format metrics.

2. **NCCL Inspector Plugin** (`.so` library) -- Loaded by NCCL at runtime
   via `NCCL_PROFILER_PLUGIN`. It intercepts collective and P2P operations,
   measures bandwidth and execution time, and periodically dumps metrics as
   Prometheus textfiles.

3. **Textfile directory** (`/var/lib/node_exporter/nccl_inspector/`) -- The
   Inspector writes `.prom` files here, one per GPU. Each file contains
   gauge metrics with labels identifying the job, GPU, collective type,
   message size, and algorithm.

4. **Node Exporter textfile collector** -- Node Exporter's
   `--collector.textfile` flag scrapes the textfile directory and exposes
   the NCCL metrics on port 9100 alongside OS metrics.

5. **OTel Collector** -- Scrapes Node Exporter and remote-writes all
   metrics (including NCCL) to AMP.

#### What gets configured

When `nccl_metrics_enabled` is `true`, the setup script:

1. Creates `/var/lib/node_exporter/nccl_inspector/` on compute nodes
   (mode 777 so any Slurm job user can write)
2. Starts Node Exporter with `--collector.textfile` pointing to that
   directory
3. Creates `/opt/slurm/etc/task_prolog.sh` that outputs the following
   environment variables for every Slurm task:

   | Variable | Value | Purpose |
   | --- | --- | --- |
   | `NCCL_PROFILER_PLUGIN` | Path to `.so` | Tells NCCL to load the Inspector |
   | `NCCL_INSPECTOR_ENABLE` | `1` | Activates the Inspector |
   | `NCCL_INSPECTOR_PROM_DUMP` | `1` | Outputs Prometheus format (not JSON) |
   | `NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS` | Configurable (default 30s) | How often metrics are written |
   | `NCCL_INSPECTOR_DUMP_DIR` | `/var/lib/node_exporter/nccl_inspector/` | Where textfiles are written |

4. Adds `TaskProlog=/opt/slurm/etc/task_prolog.sh` to `slurm.conf` on the
   controller and runs `scontrol reconfigure` to push the config to all
   compute nodes

The NCCL Inspector plugin is pre-installed on the HyperPod AMI at
`/opt/aws/hyperpod/observability/lib/libnccl-profiler-inspector.so`.
If using a custom AMI, build from
[NCCL source](https://github.com/NVIDIA/nccl/tree/master/plugins/profiler/inspector)
(requires NCCL v2.28.3+).

#### Exported NCCL metrics

The Inspector exports these Prometheus metrics:

| Metric | Description |
| --- | --- |
| `nccl_bus_bandwidth_gbs` | Bus bandwidth in GB/s for collective operations |
| `nccl_collective_exec_time_microseconds` | Execution time in microseconds for collectives |
| `nccl_p2p_bus_bandwidth_gbs` | P2P bus bandwidth in GB/s (Send/Recv) |
| `nccl_p2p_exec_time_microseconds` | P2P execution time in microseconds |

Each metric carries labels:

| Label | Description |
| --- | --- |
| `slurm_job_id` | Slurm job ID |
| `node` | Hostname |
| `gpu` | GPU identifier (e.g., `GPU0`) |
| `comm_name` | NCCL communicator name (e.g., `DP Group 0`) |
| `n_nodes` | Number of nodes (`1` = intra-node NVLink, `>1` = multi-node) |
| `nranks` | Total number of ranks |
| `collective` | Collective type: `AllReduce`, `AllGather`, `ReduceScatter`, etc. |
| `message_size` | Bucketed size range (e.g., `4-5GB`, `512-513MB`) |
| `algo_proto` | Algorithm/protocol (e.g., `Ring_ll`) |
| `p2p_operation` | `Send` or `Recv` (P2P metrics only) |

Example metric output:
```
nccl_bus_bandwidth_gbs{slurm_job_id="42",node="ip-10-1-35-255",gpu="GPU0",comm_name="DP Group 0",n_nodes="1",nranks="4",collective="AllReduce",message_size="4-5GB",algo_proto="Ring_ll"} 678.263
nccl_collective_exec_time_microseconds{slurm_job_id="42",node="ip-10-1-35-255",gpu="GPU0",comm_name="DP Group 0",n_nodes="1",nranks="4",collective="AllReduce",message_size="4-5GB",algo_proto="Ring_ll"} 9498.47
```

#### NCCL Grafana dashboard

NVIDIA provides an official Grafana dashboard template for NCCL Inspector
metrics at
[`nccl-inspector-job-performance-template.json`](https://github.com/NVIDIA/nccl/tree/master/plugins/profiler/inspector/grafana).
The dashboard shows:

| Panel Row | Metrics |
| --- | --- |
| P2P [Recv] | Recv bus bandwidth and exec time (NVLink-only and multi-node) |
| P2P [Send] | Send bus bandwidth and exec time (NVLink-only and multi-node) |
| ReduceScatter | ReduceScatter bus bandwidth and exec time |
| AllReduce | AllReduce bus bandwidth and exec time |
| AllGather | AllGather bus bandwidth and exec time |

Each row splits into NVLink-only (`n_nodes="1"`) and network
(`n_nodes!="1"`) views.

To import: download the template JSON from the NCCL repo and import it in
Grafana via **Dashboards > New > Import**. Configure the Prometheus data
source to point to your AMP workspace.

#### NCCL metrics requirements

- Multi-GPU compute nodes (NCCL collectives require 2+ GPUs)
- A running distributed training job that uses NCCL (e.g., PyTorch DDP,
  FSDP, Megatron-LM)
- Metrics are only produced while NCCL operations are active; idle GPUs
  produce no NCCL metrics

## Validating the setup

After the cluster reaches `InService` status (typically 10-15 minutes),
verify the observability stack is running.

### Check CloudWatch lifecycle logs

```bash
aws logs filter-log-events \
    --log-group-name /aws/sagemaker/Clusters/<cluster-name>/<cluster-id> \
    --filter-pattern "lifecycle scripts"
```

All instance groups should show `The lifecycle scripts succeeded.`

### Check exporters on the controller node

```bash
# Connect via SSM
aws ssm start-session \
    --target sagemaker-cluster:<cluster-id>_controller-<instance-id> \
    --region <region>

# Verify containers and services
sudo docker ps --format 'table {{.Names}}\t{{.Status}}'
systemctl is-active slurm_exporter

# Verify metrics endpoints
curl -s http://localhost:9100/metrics | head -5    # Node Exporter
curl -s http://localhost:9341/metrics | grep slurm | head -5  # Slurm Exporter
```

### Check exporters on a compute node

```bash
sudo docker ps --format 'table {{.Names}}\t{{.Status}}'

curl -s http://localhost:9100/metrics | head -5    # Node Exporter
curl -s http://localhost:9400/metrics | grep DCGM | head -5   # DCGM Exporter
curl -s http://localhost:9109/metrics | head -5    # EFA Exporter
```

### Verify metrics in AMP

Open Grafana and run a PromQL query `up` in the Explore view. You should
see targets from all cluster nodes with `job` labels like `node_exporter`,
`slurm_exporter`, `dcgm_exporter`, and `efa_exporter`.

### Import Grafana dashboards

Import the following community dashboards via **Dashboards > New > Import**:

| Dashboard | Grafana ID | URL |
| --- | --- | --- |
| Slurm Dashboard | 4323 | `https://grafana.com/grafana/dashboards/4323-slurm-dashboard/` |
| Node Exporter Full | 1860 | `https://grafana.com/grafana/dashboards/1860-node-exporter-full/` |
| NVIDIA DCGM Exporter | 12239 | `https://grafana.com/grafana/dashboards/12239-nvidia-dcgm-exporter-dashboard/` |
| EFA Metrics | 20579 | `https://grafana.com/grafana/dashboards/20579-efa-metrics-dev/` |
| FSx for Lustre | 20906 | `https://grafana.com/grafana/dashboards/20906-fsx-lustre/` |

Ensure the AMP workspace is configured as a Prometheus data source in
Grafana: **Apps > AWS Data Sources > Data sources**, select your region,
and choose the AMP workspace.

## Stopping observability

To stop all observability containers and services on a node:
```bash
sudo python3 stop_observability.py --node-type controller  # or compute, login
```

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `$'\r': command not found` in lifecycle logs | CRLF line endings (edited on Windows) | Convert all files to LF before uploading to S3. Use `.gitattributes` or `dos2unix`. |
| GPU worker detected as `login` node | `sinfo` not available during cluster creation (controller still starting) | Fixed in `setup_observability.sh` with retry loop. If using an older version, re-run the script manually after the cluster is `InService`. |
| NCCL env vars not injected into jobs | `scontrol reconfigure` not run after adding `TaskProlog` | Run `sudo scontrol reconfigure` on the controller. Fixed in current version. |
| DCGM Exporter not running on GPU node | No NVIDIA GPU detected | Expected on CPU-only compute nodes. DCGM Exporter exits gracefully. |
| Slurm Exporter stuck in `activating` | Binary not copied due to previous partial install | Run `sudo rm -rf /usr/bin/slurm_exporter` then re-run `setup_observability.sh`. Fixed in current version with clean build directory. |
| Metrics not appearing in AMP | OTel Collector can't reach AMP endpoint | Check VPC has NAT gateway or AMP VPC endpoint. Check execution role has `aps:RemoteWrite`. |
| `docker pull` fails | Can't reach ECR | Check VPC has NAT gateway or ECR VPC endpoints. |

## File structure

```
observability/
|-- README.md                            # This file
|-- config.json                          # User configuration (edit before deploying)
|-- .gitattributes                       # Enforces LF line endings
|-- setup_observability.sh               # Entrypoint script (OnInitComplete)
|-- install_observability.py             # Orchestrator (called by setup_observability.sh)
|-- install_node_exporter.sh             # Node Exporter (all nodes)
|-- install_dcgm_exporter.sh             # DCGM Exporter (compute nodes with GPU)
|-- install_efa_exporter.sh              # EFA Exporter (compute nodes)
|-- install_slurm_exporter.sh            # Slurm Exporter (controller node)
|-- install_otel_collector.sh            # OTel Collector (all nodes)
|-- stop_observability.py                # Stop all observability services
|-- LICENSE_SLURM_EXPORTER.txt           # License for Slurm Exporter dependency (GPLv3)
|-- otel_config/
|   |-- config-head-template.yaml        # OTel config template for controller
|   |-- config-compute-template.yaml     # OTel config template for compute
|   +-- config-login-template.yaml       # OTel config template for login
+-- dcgm_metrics_config/
    |-- dcgm-metrics-basic.csv           # Basic DCGM metrics (default)
    +-- dcgm-metrics-advanced.csv        # Advanced DCGM metrics (when advanced_metrics=true)
```

## Related resources

- [SageMaker HyperPod cluster resources monitoring](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-cluster-observability-slurm.html)
- [Getting started with SageMaker HyperPod using the AWS CLI](https://docs.aws.amazon.com/sagemaker/latest/dg/smcluster-getting-started-slurm-cli.html)
- [Customizing SageMaker HyperPod clusters using lifecycle scripts](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-lifecycle-best-practices-slurm.html)
- [Slurm Dashboard for Grafana](https://grafana.com/grafana/dashboards/4323-slurm-dashboard/)
- [NVIDIA DCGM Exporter Dashboard for Grafana](https://grafana.com/grafana/dashboards/12239-nvidia-dcgm-exporter-dashboard/)
- [EFA Metrics Dashboard for Grafana](https://grafana.com/grafana/dashboards/20579-efa-metrics-dev/)
- [NCCL Inspector source](https://github.com/NVIDIA/nccl/tree/master/plugins/profiler/inspector)
