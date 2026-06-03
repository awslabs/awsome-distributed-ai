# AWS Parallel Computing Service Distributed Training Reference Architecture

This repository provides reference architectures and deployment templates for setting up distributed training clusters using [AWS Parallel Computing Service (PCS)](https://aws.amazon.com/pcs/). AWS Parallel Computing Service is a fully managed service that makes it easy to run and scale HPC workloads using Slurm scheduler. These architectures are optimized for machine learning workloads and include configurations for high-performance computing instances (P and Trn EC2 families) with shared filesystems (FSx for Lustre and OpenZFS).

> **Upstream Repository**: These templates are based on [aws-samples/aws-hpc-recipes](https://github.com/aws-samples/aws-hpc-recipes/tree/main/recipes/pcs), customized for ML workloads: container support (Enroot/Pyxis) installable at first boot without an AMI build, built-in monitoring, updated Slurm versions (25.05/25.11), and dedicated P5/P6 multi-NIC EFA templates. The templates in this repository are maintained independently and may diverge from the upstream recipes.

## Key Features

- **No AMI build required**: builds on the AWS-managed **PCS-ready DLAMI** (NVIDIA driver, CUDA, PCS agent, and Slurm 25.05/25.11 pre-installed), so a cluster comes up in minutes with no Image Builder step.
- **Container runtime, two ways**: Enroot/Pyxis is either **installed at first boot** via a post-install hook (`PostInstallScriptUrl`, the default) or **pre-baked into a custom AMI** (`BuildAMI=true`) — pick boot speed vs. setup time.
- **Monitoring built in**: Prometheus + Grafana + GPU (DCGM) dashboards deploy automatically on the login node (`DeployMonitoring=true`, on by default); access via SSM port-forward, no public endpoint.
- **GPU-ready, multi-NIC EFA**: dedicated launch templates for P5/P5e/P5en and P6-B200/P6-B300, selected automatically by instance type, for high-bandwidth multi-node training.
- **Flexible capacity**: On-Demand, On-Demand Capacity Reservations (ODCR), and Capacity Blocks for ML.
- **High-performance storage**: FSx for Lustre (shared scratch, `/fsx`) and FSx for OpenZFS (home directories, `/home`).
- **One-click or modular**: deploy a complete cluster from a single nested stack, or compose individual components.

## Architecture

![AWS PCS diagram](./images/ml-pcs-architecture.png)

A default deployment (`pcs-ml-cluster-deploy-all.yaml`) provisions:
- VPC with public/private subnets, NAT gateway, and S3 endpoint
- FSx for Lustre (`/fsx`, high-performance shared scratch) and FSx for OpenZFS (`/home`)
- PCS cluster with the Slurm scheduler (25.05 or 25.11), on the PCS-ready DLAMI
- Login node group (public subnet) with the monitoring stack (Prometheus/Grafana/DCGM)
- CPU compute node group (private subnet); optional GPU (P5/P6) node group with EFA
- Enroot/Pyxis container runtime installed at first boot (default) or pre-baked via `BuildAMI=true`

---

## Quick Start

Deploy a complete cluster with one nested CloudFormation stack:

[![Launch](images/launch-stack.svg)](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml&stackName=pcs-ml-cluster)

The only required parameter is the Availability Zone. The minimal CLI equivalent:

```bash
aws cloudformation create-stack \
  --stack-name pcs-ml-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-1a \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

This brings up (≈25–30 min, mostly VPC/FSx): 1 login node (m6i.4xlarge) with monitoring,
a `cpu1` queue (c6i.4xlarge, 0–4 nodes, dynamic scaling), and Enroot/Pyxis on every node.
Add a GPU queue and tune storage/monitoring via the parameters below.

Once it's up:
- **Connect** to the login node via SSM Session Manager — see [Accessing the Cluster](#accessing-the-cluster).
- **Open the Grafana dashboards** (deployed by default) via SSM port forwarding — see [Accessing Grafana](#accessing-grafana).

Prefer step-by-step instructions? See the [AI/ML for AWS PCS Workshop](https://catalog.workshops.aws/ml-on-pcs/).

---

## Configuration

Key `pcs-ml-cluster-deploy-all.yaml` parameters (the console groups them and shows
friendly labels). Defaults give the most common production setup —
`BuildAMI=false` + Enroot/Pyxis via `PostInstallScriptUrl` + `DeployMonitoring=true`.

| Parameter | Default | Purpose |
|---|---|---|
| `PrimarySubnetAZ` | *(required)* | Availability Zone to deploy into |
| `BuildAMI` | `false` | Pre-bake Enroot/Pyxis into a custom DLAMI (adds ~30 min Image Builder step) instead of installing at first boot |
| `PostInstallScriptUrl` | Enroot/Pyxis installer | Script run on every node at first boot (PCS equivalent of ParallelCluster `OnNodeConfigured`). Empty = skip; or point at any HTTP(S) script |
| `DeployMonitoring` | `true` | Deploy Prometheus/Grafana/DCGM on the login node |
| `DeployOnDemandCNG` | `true` | Deploy the `cpu1` CPU queue (`OnDemandInstanceType`, default `c6i.4xlarge`) |
| `DeployPseriesCNG` | `false` | Deploy a GPU (P5/P6) queue — see [GPU compute](#gpu-compute-p5p6) |
| `PseriesInstanceType` | `p5.48xlarge` | GPU instance type; selects the matching multi-NIC template automatically |
| `CapacityReservationId` | *(empty)* | ODCR / Capacity Block ID for the GPU queue |
| `SlurmVersion` | `25.11` | Slurm version (`25.05` or `25.11`) |
| `RootVolumeSize` | `300` | Node root volume (GiB); 300 leaves room for large container images |

### Container runtime (Enroot/Pyxis)

Two independent, combinable options:

- **First-boot install (default)**: `PostInstallScriptUrl` runs [`scripts/install-enroot-pyxis.sh`](./scripts/install-enroot-pyxis.sh) on each node — no AMI build, ~8–12 min node boot. Best for testing/infrequent scaling.
- **Pre-baked AMI** (`BuildAMI=true`): `pcs-ready-dlami-with-enroot-pyxis.yaml` bakes Enroot 3.5.0 + Pyxis 0.20.0 into a custom DLAMI (~30 min build, ~3 min node boot). Best for production/frequent scaling.

The PCS-ready DLAMI base already includes the PCS agent, Slurm 25.05 & 25.11
(`/opt/aws/pcs/scheduler/slurm-*`), NVIDIA driver + CUDA, and SSM agent.

### Storage: FSx deployment types (Region availability)

**FSx deployment types are not available in every Region.** Defaults match the most
capable type; switch to a more widely available one if your Region needs it.

| Filesystem | Parameter | Default | Other values | Notes |
|---|---|---|---|---|
| Lustre (`/fsx`) | `LustreDeploymentType` | `PERSISTENT_2` | `PERSISTENT_1` | `PERSISTENT_2` (throughput 125/250/500/1000, metadata config) isn't in every Region; `PERSISTENT_1` (50/100/200) is in more Regions |
| Lustre (`/fsx`) | `PerUnitStorageThroughput` | `250` | any valid number | Must be valid for the type: P2 = 125/250/500/1000, P1 = 50/100/200 |
| OpenZFS (`/home`) | `OpenZFSDeploymentType` | `SINGLE_AZ_HA_2` | `SINGLE_AZ_HA_1`, `SINGLE_AZ_2`, `SINGLE_AZ_1` | `SINGLE_AZ_1` is in all Regions; HA/2 variants vary. `MULTI_AZ` excluded (needs a second subnet) |

Check support before deploying:
[Lustre Regions](https://docs.aws.amazon.com/fsx/latest/LustreGuide/using-fsx-lustre.html) ·
[OpenZFS Regions](https://docs.aws.amazon.com/fsx/latest/OpenZFSGuide/available-aws-regions.html).
If a deploy fails at the FSx resource with an "unsupported deployment type" error,
switch these parameters to a type your Region supports.

### GPU compute (P5/P6)

P5 instances expose 16/32 uniform EFA interfaces, but **P6-B200 has 8 network cards
and P6-B300 has 17** (and on P6-B300 network card 0 cannot do EFA), so each family
uses its own launch-template NIC layout. With deploy-all you just set
`PseriesInstanceType` and the matching template is selected automatically.

| Instance type | GPUs | EFA | Template |
|---|---|---|---|
| `p5.48xlarge` | 8× H100 | 32 NICs | `add-cng-p5.yaml` |
| `p5e.48xlarge` | 8× H200 | 32 NICs | `add-cng-p5.yaml` |
| `p5en.48xlarge` | 8× H200 (NVSwitch) | 16 NICs | `add-cng-p5.yaml` |
| `p6-b200.48xlarge` | 8× B200 | 8 network cards | `add-cng-p6-b200.yaml` |
| `p6-b300.48xlarge` | 8× B300 | 17 network cards (16 EFA-capable) | `add-cng-p6-b300.yaml` |

**Capacity:** use On-Demand, ODCR (`CapacityReservationId`), or Capacity Blocks for ML
(`CapacityReservationId` + `MarketType=capacity-block`, handled by the template).

> **Capacity Block billing:** a block bills for its whole reserved window once it
> starts and cannot be stopped early. When the block is active, run the GPU node
> group at `PseriesMinCount = PseriesMaxCount = <reserved count>` so the reserved
> nodes launch immediately, rather than scaling from 0.

---

## Templates

All templates live in [`assets/`](./assets/). `pcs-ml-cluster-deploy-all.yaml` nests
the others; you can also deploy each individually for more control (e.g. reuse a VPC/FSx
across clusters). Click **Deploy** to 1-click-launch a single template.

| Template | Purpose | Deploy |
|---|---|---|
| [`pcs-ml-cluster-deploy-all.yaml`](./assets/pcs-ml-cluster-deploy-all.yaml) | All-in-one: Prerequisites + (optional AMI) + Cluster + login/CPU/GPU CNGs | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml&stackName=pcs-ml-cluster) |
| [`ml-cluster-prerequisites.yaml`](./assets/ml-cluster-prerequisites.yaml) | VPC, subnets, security groups, FSx for Lustre + OpenZFS | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/ml-cluster-prerequisites.yaml&stackName=pcs-prerequisites) |
| [`cluster.yaml`](./assets/cluster.yaml) | PCS cluster core (Slurm scheduler only, no nodes) | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/cluster.yaml&stackName=pcs-cluster) |
| [`add-cng.yaml`](./assets/add-cng.yaml) | Compute node group, single NIC — login nodes, CPU/single-NIC-GPU queues (C6i, G5, G6) | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/add-cng.yaml&stackName=pcs-add-cng) |
| [`add-cng-p5.yaml`](./assets/add-cng-p5.yaml) | P5/P5e/P5en nodes, 16/32 EFA interfaces | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/add-cng-p5.yaml&stackName=pcs-add-cng-p5) |
| [`add-cng-p6-b200.yaml`](./assets/add-cng-p6-b200.yaml) | P6-B200 nodes, 8 EFA network cards | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/add-cng-p6-b200.yaml&stackName=pcs-add-cng-p6-b200) |
| [`add-cng-p6-b300.yaml`](./assets/add-cng-p6-b300.yaml) | P6-B300 nodes, 17 EFA network cards | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/add-cng-p6-b300.yaml&stackName=pcs-add-cng-p6-b300) |
| [`pcs-ready-dlami-with-enroot-pyxis.yaml`](./assets/pcs-ready-dlami-with-enroot-pyxis.yaml) | EC2 Image Builder: bake Enroot 3.5.0 + Pyxis 0.20.0 into the PCS-ready DLAMI | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ready-dlami-with-enroot-pyxis.yaml&stackName=pcs-dlami) |

`add-cng*` templates create a Slurm queue only when `QueueName` is set (leave it empty
for login nodes). The P-series templates need a `CapacityReservationId` when using a
Capacity Block.

---

## Usage Examples

### Example 1: Default CPU cluster

```bash
aws cloudformation create-stack \
  --stack-name cpu-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-1a \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```
1 login node + `cpu1` queue (c6i.4xlarge, 0–4 nodes, dynamic scaling).

### Example 2: Single-NIC GPU queue (G6)

```bash
aws cloudformation create-stack \
  --stack-name gpu-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-1a \
    ParameterKey=OnDemandCngName,ParameterValue=gpu-g6 \
    ParameterKey=OnDemandQueueName,ParameterValue=gpu-g6 \
    ParameterKey=OnDemandInstanceType,ParameterValue=g6.12xlarge \
    ParameterKey=OnDemandMaxCount,ParameterValue=8 \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```
Replaces the default `cpu1` queue with a `gpu-g6` queue of g6.12xlarge instances.

### Example 3: Multi-NIC GPU with a Capacity Block (P6-B300)

```bash
AZ_ID=us-west-2b
CAPACITY_RESERVATION_ID="cr-0a1b2c3d4e5f6g7h8"

aws cloudformation create-stack \
  --stack-name p6-b300-cb-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=${AZ_ID} \
    ParameterKey=DeployPseriesCNG,ParameterValue=true \
    ParameterKey=PseriesCngName,ParameterValue=gpu-p6b300 \
    ParameterKey=PseriesQueueName,ParameterValue=gpu-p6b300 \
    ParameterKey=PseriesInstanceType,ParameterValue=p6-b300.48xlarge \
    ParameterKey=PseriesMinCount,ParameterValue=2 \
    ParameterKey=PseriesMaxCount,ParameterValue=2 \
    ParameterKey=CapacityReservationId,ParameterValue=${CAPACITY_RESERVATION_ID} \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```
The `add-cng-p6-b300.yaml` template (17 network cards) is selected automatically from
`PseriesInstanceType`. For `p6-b200.48xlarge` just change that value; for P5, set
`PseriesInstanceType=p5.48xlarge` (optionally `NetworkInterfaceCount=16|32`). For an
ODCR instead of a Capacity Block, pass the reservation as `CapacityReservationId` the
same way (or omit it for pure on-demand).

---

## Accessing the Cluster

Connect to the login node with AWS Systems Manager Session Manager (no public SSH needed).

**Console:** [EC2 Console](https://console.aws.amazon.com/ec2/home#Instances:) → filter
by tag `aws:pcs:compute-node-group-name = login` → select the instance → **Connect** →
**Session Manager** → **Connect**.

**CLI** (needs `ec2:DescribeInstances` + `ssm:StartSession`; AWS CloudShell has these):

```bash
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:aws:pcs:compute-node-group-name,Values=login" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)

aws ssm start-session --target $INSTANCE_ID
```

Then switch to the default user and check the cluster:

```bash
sudo su - ubuntu
sinfo                 # partitions and nodes
squeue                # job queue
scontrol show nodes   # node detail
```

See [Connect to Cluster](https://catalog.workshops.aws/ml-on-pcs/en-US/03-cluster/02-connect-cluster) in the workshop for more.

---

## Monitoring

With `DeployMonitoring=true` (default), an integrated monitoring stack based on
[aws-parallelcluster-monitoring](https://github.com/aws-samples/aws-parallelcluster-monitoring)
is installed automatically:

- **Login node**: Prometheus, Grafana, Nginx (reverse proxy), Node Exporter, Pushgateway, CloudWatch Exporter
- **Compute nodes**: Node Exporter, plus DCGM Exporter on GPU nodes
- **Slurm**: native OpenMetrics on the controller (jobs/nodes/partitions/scheduler)

Metrics cover Slurm jobs, GPU (utilization/memory/temperature/power/ECC/NVLink via DCGM),
node CPU/memory/disk/network, and CloudWatch (EC2/FSx/PCS). The stack installs on
node-local `/opt` (not the shared `/home`).

**Monitoring-related parameters:**
- `DeployMonitoring` (default `true`)
- `MonitoringVersion` — [aws-parallelcluster-monitoring](https://github.com/aws-samples/aws-parallelcluster-monitoring) git ref (release tag, branch, or `latest`; default `v2.6.3`). Pinned to a tag so upstream changes can't break deployments unexpectedly.
- `MonitoringRepo` — `owner/repo` to fetch from (default `aws-samples/aws-parallelcluster-monitoring`). Point at a fork + a branch in `MonitoringVersion` to test unreleased changes.

> Node type is identified by the `monitoring-role` tag (`login`/`compute`), not the EC2
> `Name` tag — the `Name` tag defaults to `PCS-<cngname>` and is free for you to retag.

### Accessing Grafana

Grafana is reached via SSM port forwarding (no public access).

```bash
# 1. Login node instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:aws:pcs:compute-node-group-name,Values=login" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)

# 2. Port-forward 443 -> localhost:8443 (needs the Session Manager plugin)
aws ssm start-session --target $INSTANCE_ID \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["443"],"localPortNumber":["8443"]}'

# 3. Grafana admin password (CLUSTER_ID from the stack's ClusterId output)
aws ssm get-parameter --name "/pcs/${CLUSTER_ID}/grafana/admin-password" \
  --with-decryption --query 'Parameter.Value' --output text
```

Open `https://localhost:8443/grafana/` and log in as `admin` with that password.
Use the dashboard nav bar to switch between **Cluster Summary**, **Slurm Detail**,
**Compute Node List**, **GPU Node List**, **GPU Health**, **Cluster Costs**, and
**Storage**. For example, the **GPU Node List** shows each GPU node's model, instance
type, utilization, temperature, power, and memory:

![Grafana GPU Node List dashboard](images/dashboard-screenshot-gpu-list.png)

For detailed validation, the full metric list, and troubleshooting, see
[tests/monitoring-stack-test.md](tests/monitoring-stack-test.md).

> **Ubuntu/PCS support is native as of `v2.6.3`** ([PR #44](https://github.com/aws-samples/aws-parallelcluster-monitoring/pull/44)) — the old `ec2-user` symlink and `local`-var workarounds are gone; just keep `MonitoringVersion` at `v2.6.3`+.

---

## Cleanup

```bash
aws cloudformation delete-stack --stack-name pcs-ml-cluster
```

Nested stacks are deleted automatically. Back up any FSx data first — the filesystems
are deleted with the stack.

---

## Testing and Validation

Validated configurations:

- **Infrastructure** (`ml-cluster-prerequisites.yaml`, `cluster.yaml`): multiple Regions (us-east-1/us-west-2/us-east-2), Slurm 25.05 & 25.11.
- **CPU / single-NIC GPU** (`add-cng.yaml`): login (m6i.4xlarge), CPU (c6i.4xlarge), GPU (g6.xlarge/g6.12xlarge).
- **P5** (`add-cng-p5.yaml`): p5.48xlarge / p5en.48xlarge with ODCR and Capacity Blocks for ML.
- **P6-B300** (`add-cng-p6-b300.yaml`): validated on real p6-b300.48xlarge (Capacity Blocks, us-west-2) — 17 network cards, EFA active (NCCL `found 16 nics`), 2-node all_reduce ~761 GB/s peak, FSDP Llama-2 7B multi-node.
- **P6-B200** (`add-cng-p6-b200.yaml`): 8-network-card template (same EFA layout family as B300).
- **All-in-one** (`pcs-ml-cluster-deploy-all.yaml`): selects the P5/P6-B200/P6-B300 CNG template automatically from `PseriesInstanceType`; tested end-to-end with monitoring, container jobs, NCCL, and FSDP on p6-b300.
- **AMI builder** (`pcs-ready-dlami-with-enroot-pyxis.yaml`): Ubuntu 24.04 x86_64 AMIs with Enroot 3.5.0 + Pyxis 0.20.0, validated with PyTorch/CUDA containers.

---

## User Management

- [LDAP Server Setup Guide](../6.ldap_server/README.md) — OpenLDAP for cluster-wide user authentication.

## Additional Resources

- [AWS Parallel Computing Service Documentation](https://docs.aws.amazon.com/pcs/)
- [AI/ML for AWS PCS Workshop](https://catalog.workshops.aws/ml-on-pcs/)
- [Slurm Documentation](https://slurm.schedmd.com/documentation.html)
- [Enroot](https://github.com/NVIDIA/enroot) · [Pyxis](https://github.com/NVIDIA/pyxis)
- [Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)
- [aws-parallelcluster-monitoring](https://github.com/aws-samples/aws-parallelcluster-monitoring) (upstream monitoring)
- [Prometheus & Grafana Setup](../../4.validation_and_observability/4.prometheus-grafana/README.md) (alternative monitoring stack)
