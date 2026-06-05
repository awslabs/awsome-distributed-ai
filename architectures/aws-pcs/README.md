# AWS Parallel Computing Service Distributed Training Reference Architecture

This repository provides reference architectures and deployment templates for setting up distributed training clusters using [AWS Parallel Computing Service (PCS)](https://aws.amazon.com/pcs/). AWS Parallel Computing Service is a fully managed service that makes it easy to run and scale HPC workloads using Slurm scheduler. These architectures are optimized for machine learning workloads and include configurations for high-performance computing instances (P and Trn EC2 families) with shared filesystems (FSx for Lustre and OpenZFS).

> **Upstream Repository**: These templates are based on [aws-samples/aws-hpc-recipes](https://github.com/aws-samples/aws-hpc-recipes/tree/main/recipes/pcs), customized for ML workloads: container support (Enroot/Pyxis) installable at first boot without an AMI build, built-in monitoring, updated Slurm versions (25.05/25.11), and dedicated P5/P6 multi-NIC EFA templates. The templates in this repository are maintained independently and may diverge from the upstream recipes.

## 1. Key Features

- **One click to an ML-training-ready cluster**: a single CloudFormation stack gives you a complete, ready-to-train environment — Slurm scheduler, GPU compute with EFA, shared FSx storage, the Enroot/Pyxis container runtime, and monitoring — with only the Availability Zone to choose. Submit distributed training jobs minutes after launch.
- **Container runtime included**: Enroot/Pyxis is set up automatically, so `srun --container-image=...` works out of the box for containerized training.
- **Monitoring built in**: Prometheus + Grafana + GPU (DCGM) dashboards deploy automatically on the login node (`DeployMonitoring=true`, on by default); reach it privately via SSM port-forward, or open it to a trusted CIDR with `GrafanaPublicAccessCidr`.
- **GPU-ready, multi-NIC EFA**: dedicated launch templates for the P5 and P6 families, selected automatically by instance type, for high-bandwidth multi-node training.
- **Broad capacity-purchase support**: covers the full range of EC2 capacity options out of the box — On-Demand, On-Demand Capacity Reservations (ODCR), and Capacity Blocks for ML — selected per node group.
- **High-performance storage**: FSx for Lustre (shared scratch, `/fsx`) and FSx for OpenZFS (home directories, `/home`).
- **Modular components**: compose individual stacks (network/storage prerequisites, cluster scheduler, per-family compute node groups) instead of the all-in-one nested stack when you want to reuse infrastructure across clusters or iterate on one piece at a time.

> Built on the AWS-managed **PCS-ready DLAMI** (NVIDIA driver, CUDA, PCS agent, and
> Slurm 25.05/25.11 pre-installed), so no custom AMI build is required by default —
> the cluster comes up without an Image Builder step. (Pre-baking Enroot/Pyxis into a
> custom AMI is still available via `BuildAMI=true` for faster node boot at scale.)

## 2. Architecture

![AWS PCS diagram](./images/ml-pcs-architecture.png)

A default deployment (`pcs-ml-cluster-deploy-all.yaml`) provisions:
- VPC with public/private subnets, NAT gateway, and S3 endpoint
- FSx for Lustre (`/fsx`, high-performance shared scratch) and FSx for OpenZFS (`/home`)
- PCS cluster with the Slurm scheduler (25.05 or 25.11), on the PCS-ready DLAMI
- Login node group (public subnet) with the monitoring stack (Prometheus/Grafana/DCGM)
- CPU compute node group (private subnet); optional GPU (P5/P6) node group with EFA
- Enroot/Pyxis container runtime installed at first boot (default) or pre-baked via `BuildAMI=true`

---

## 3. Quick Start

Deploy a complete cluster with one nested CloudFormation stack:

[![Launch](images/launch-stack.svg)](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml&stackName=pcs-ml-cluster)

**The only decision you must make is which Availability Zone to deploy into**
(`PrimarySubnetAZ`) — everything else has a sensible default. The minimal CLI
equivalent (set your AZ in the first line):

```bash
AZ_ID=us-east-1a   # <-- the one required choice: your target Availability Zone

aws cloudformation create-stack \
  --stack-name pcs-ml-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters ParameterKey=PrimarySubnetAZ,ParameterValue=${AZ_ID} \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

This brings up (≈25–30 min, mostly VPC/FSx): 1 login node (m6i.4xlarge) with monitoring,
a `cpu1` queue (c6i.4xlarge, 0–4 nodes, dynamic scaling), and Enroot/Pyxis on every node.
Add a GPU queue and tune storage/monitoring via the parameters below.

Once it's up:
- **Connect** to the login node via SSM Session Manager — see [Accessing the Cluster](#6-accessing-the-cluster).
- **Open the Grafana dashboards** (deployed by default) via SSM port forwarding — see [Accessing Grafana](#accessing-grafana).
- **Want to reach Grafana directly in a browser** (no port forwarding)? Set `GrafanaPublicAccessCidr` to a trusted CIDR at deploy time — see [Option B — Direct public access](#option-b--direct-public-access-opt-in-via-grafanapublicaccesscidr).

Prefer step-by-step instructions? See the [AI/ML for AWS PCS Workshop](https://catalog.workshops.aws/ml-on-pcs/).

**Clean up.** When you're done, delete the stack — either from the **CloudFormation
Management Console** (select the stack → **Delete**) or via the CLI:

```bash
aws cloudformation delete-stack --stack-name pcs-ml-cluster
```

Nested stacks are deleted automatically. Back up any FSx data first — the filesystems
are deleted with the stack.

---

## 4. Configuration

Defaults give the most common production setup — `BuildAMI=false` + Enroot/Pyxis via
`PostInstallScriptUrl` + `DeployMonitoring=true` — so a default deploy only needs the
Availability Zone (`PrimarySubnetAZ`). The most-used parameters:

| Parameter | Default | Purpose |
|---|---|---|
| `PrimarySubnetAZ` | *(required)* | Availability Zone to deploy into — the one required parameter |
| `SlurmVersion` | `25.11` | Slurm version (`25.05` or `25.11`); 25.11 is needed for the Slurm OpenMetrics dashboards. Drives Pyxis build version too. See [OPERATIONS.md §1](./docs/OPERATIONS.md#1-slurm-version-selection) |
| `BuildAMI` | `false` | Pre-bake Enroot/Pyxis into a custom DLAMI instead of installing at first boot |
| `DeployMonitoring` | `true` | Deploy Prometheus/Grafana/DCGM on the login node |
| `DeployOnDemandCNG` | `true` | Deploy the `cpu1` CPU queue (`OnDemandInstanceType`, default `c6i.4xlarge`) |
| `DeployPseriesCNG` | `false` | Deploy a GPU (P5/P6) queue — see [GPU compute](#gpu-compute-p5p6) |
| `PseriesInstanceType` | `p5.48xlarge` | GPU instance type; auto-selects the multi-NIC template + EFA count |
| `CapacityReservationId` | *(empty)* | Capacity **Block** ID for the GPU queue; empty for On-Demand/ODCR |

**See [PARAMETERS.md](./docs/PARAMETERS.md) for the complete parameter reference** (all 8
console parameter groups, with every default). The concept guides below cover the
choices that need the most thought.

### Container runtime (Enroot/Pyxis)

Choose **one** of two ways to provide Enroot/Pyxis:

- **First-boot install (default)**: `PostInstallScriptUrl` runs [`scripts/install-enroot-pyxis.sh`](./scripts/install-enroot-pyxis.sh) on each node — no AMI build, ~8–12 min node boot. Best for testing/infrequent scaling.
- **Pre-baked AMI** (`BuildAMI=true`): `pcs-ready-dlami-with-enroot-pyxis.yaml` bakes Enroot 3.5.0 + Pyxis 0.20.0 into a custom DLAMI (~30 min build, ~3 min node boot). Best for production/frequent scaling. The template also wraps the build in a managed Image Builder pipeline with optional **scheduled rebuilds** (`BuildSchedule=Weekly`/`Monthly`), an **AMI lifecycle policy** that deprecates older AMIs after a configurable window, and **SSM parameter publishing** so other stacks can resolve the latest AMI ID. Without these, an equivalent build would be a one-shot ad-hoc invocation; the in-stack pipeline is what justifies the extra surface vs. just passing `AmiId` to a hand-built AMI.

> **`BuildAMI=true` + `PostInstallScriptUrl`.** `PostInstallScriptUrl` is a generic
> first-boot hook (not Enroot/Pyxis-specific), so the templates don't force it empty under
> `BuildAMI=true`. If you set `BuildAMI=true` and leave the default Enroot/Pyxis installer,
> nothing breaks — the installer is idempotent and skips components already baked into the
> AMI (a fast no-op) — but for the cleanest boot pass **`PostInstallScriptUrl=""`** (or
> point it at a different script for other first-boot setup). With `BuildAMI=false`
> (default), leave `PostInstallScriptUrl` at its default.

The PCS-ready DLAMI base already includes the PCS agent, Slurm 25.05 & 25.11
(`/opt/aws/pcs/scheduler/slurm-*`), NVIDIA driver + CUDA, and SSM agent.

> **Production tip — pin the AMI.** Empty `AmiId` re-resolves the SSM `/latest/` parameter on
> every stack update, so scale-out nodes can drift to a newer AMI. For production, resolve it
> once and pass the result as `AmiId`. Details:
> [OPERATIONS.md §4](./docs/OPERATIONS.md#4-ami-selection-amiid--pin-in-production).

### GPU compute (P5/P6)

Different P-series instances expose different numbers of EFA interfaces, so each family
has its own launch template with the right interface layout. With deploy-all you just
set `PseriesInstanceType` and the matching template (and interface count) is selected
automatically.

| Instance type | GPUs | EFA interfaces | Template |
|---|---|---|---|
| `p5.48xlarge` | 8× H100 | 32 | `add-cng-p5.yaml` |
| `p5e.48xlarge` | 8× H200 | 32 | `add-cng-p5.yaml` |
| `p5en.48xlarge` | 8× H200 | 16 | `add-cng-p5.yaml` |
| `p6-b200.48xlarge` | 8× B200 | 8 | `add-cng-p6-b200.yaml` |
| `p6-b300.48xlarge` | 8× B300 | 16 (of 17 interfaces; the primary is ENA-only) | `add-cng-p6-b300.yaml` |

**Capacity options:**
- **On-Demand**: leave `CapacityReservationId` empty.
- **On-Demand Capacity Reservation (ODCR)**: also leave `CapacityReservationId` **empty** — create the ODCR with **"open"** instance matching and it is consumed automatically by the node group's On-Demand launches. (Do **not** put the ODCR ID in `CapacityReservationId`; that parameter forces Capacity-Block mode.)
- **Capacity Blocks for ML**: set `CapacityReservationId` to the Capacity Block ID. The template then launches with `MarketType=capacity-block` against it.

> **Capacity Block billing:** a block bills for its whole reserved window once it
> starts and cannot be stopped early. When the block is active, run the GPU node
> group at `PseriesMinCount = PseriesMaxCount = <reserved count>` so the reserved
> nodes launch immediately, rather than scaling from 0.

### Storage: FSx deployment types (Region availability)

**FSx deployment types are not available in every Region.** Defaults match the most
capable type; switch to a more widely available one if your Region needs it.

| Filesystem | Parameter | Default | Other values | Notes |
|---|---|---|---|---|
| Lustre (`/fsx`) | `LustreDeploymentType` | `PERSISTENT_2` | `PERSISTENT_1` | `PERSISTENT_2` (throughput 125/250/500/1000, metadata config) isn't in every Region; `PERSISTENT_1` (50/100/200) is in more Regions |
| Lustre (`/fsx`) | `PerUnitStorageThroughput` | `250` | any valid number | Must be valid for the type: P2 = 125/250/500/1000, P1 = 50/100/200 |
| OpenZFS (`/home`) | `OpenZFSDeploymentType` | `SINGLE_AZ_HA_2` | `SINGLE_AZ_HA_1`, `SINGLE_AZ_2`, `SINGLE_AZ_1` | `SINGLE_AZ_1` is in all Regions; HA/2 variants vary. `MULTI_AZ` excluded (needs a second subnet) |
| OpenZFS (`/home`) | `HomeThroughput` | `320` | any valid number | Throughput (MB/s). Valid values depend on the deployment type: `SINGLE_AZ_2`/`SINGLE_AZ_HA_2` = 160/320/640/1280/2560/3840/5120/7680/10240; `SINGLE_AZ_HA_1` = 128/256/512/1024/2048/3072/4096; `SINGLE_AZ_1` = 64/128/256/512/1024/2048/3072/4096 |

Check support before deploying:
[Lustre Regions](https://docs.aws.amazon.com/fsx/latest/LustreGuide/using-fsx-lustre.html) ·
[OpenZFS Regions](https://docs.aws.amazon.com/fsx/latest/OpenZFSGuide/available-aws-regions.html).
If a deploy fails at the FSx resource with an "unsupported deployment type" error,
switch these parameters to a type your Region supports.

---

## 5. Usage Examples

All examples start by setting `AZ_ID` — the one required choice.

### Example 1: Default CPU cluster

```bash
AZ_ID=us-east-1a   # your target Availability Zone

aws cloudformation create-stack \
  --stack-name cpu-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters ParameterKey=PrimarySubnetAZ,ParameterValue=${AZ_ID} \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```
1 login node + `cpu1` queue (c6i.4xlarge, 0–4 nodes, dynamic scaling).

### Example 2: Single-NIC GPU queue (G6)

```bash
AZ_ID=us-east-1a   # your target Availability Zone

aws cloudformation create-stack \
  --stack-name gpu-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=${AZ_ID} \
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
The `add-cng-p6-b300.yaml` template is selected automatically from `PseriesInstanceType`,
and the EFA interface count is derived from the instance type — no interface-count
parameter to set. For `p6-b200.48xlarge` or any P5 type, just change
`PseriesInstanceType`. `CapacityReservationId` here is the **Capacity Block** ID; for
On-Demand or an "open" ODCR, leave it empty (see [GPU compute](#gpu-compute-p5p6)).

---

## 6. Accessing the Cluster

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

## 7. Running a multi-node GPU job (NCCL test)

The repo's canonical NCCL launcher
[`micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch`](../../micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch)
runs an `all_reduce_perf` benchmark across 2 nodes and is the quickest way to confirm
the GPU queue, Pyxis containers, and EFA work end-to-end. Two PCS-specific deltas are
all you need to add:

**1. Import the image on the login node** — `enroot import` builds its overlayfs on the
node-local root disk (the login node has 300 GiB), and FSx for Lustre can't host that
overlay; only the resulting `.sqsh` lands on shared `/fsx`. Pin a specific image tag
for reproducible numbers (don't use `latest`):

```bash
TAG=cuda12.8.1-efa1.43.2-ofiv1.16.3-ncclv2.27.7-1-testsv2.16.9
enroot import -o /fsx/nccl-tests.sqsh "docker://public.ecr.aws#hpc-cloud/nccl-tests:${TAG}"
```

**2. Submit on your GPU partition** — the canonical sbatch reads `$IMAGE`
(`/fsx/nccl-tests.sqsh` by default) and defaults to 2 nodes / 8 tasks per node:

```bash
cd /fsx && git clone --depth 1 https://github.com/awslabs/awsome-distributed-ai.git
sbatch --partition=gpu-p6b300 \
  /fsx/awsome-distributed-ai/micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch
```

**3. Check the result** (`nccl-all_reduce_perf_<jobid>.out`). EFA is in use when you see
`NET/OFI Selected provider is efa ... (found N nics)`, and a healthy run ends with
`# Out of bounds values : 0 OK` plus a busbw column that scales up with message size
(e.g. ~751 GB/s at 64 GiB on 2× p6-b300; raise `-e` past the default 16 GiB to saturate
B300's 16 EFA cards).

For a full training example, see the [PyTorch FSDP test case](../../3.test_cases/pytorch/FSDP).
For the full validation matrix (monitoring, containers, CPU/GPU, NCCL, FSDP) and the
PCS deltas worth knowing, see the [Test & Validation Guide](tests/README.md).

---

## 8. Monitoring

With `DeployMonitoring=true` (default), an integrated monitoring stack based on
[aws-parallelcluster-monitoring](https://github.com/aws-samples/aws-parallelcluster-monitoring)
is installed automatically:

- **Login node**: Prometheus, Grafana, Nginx (reverse proxy), Node Exporter, Pushgateway, CloudWatch Exporter
- **Compute nodes**: Node Exporter, plus DCGM Exporter on GPU nodes
- **Slurm**: native OpenMetrics on the controller (jobs/nodes/partitions/scheduler)

Metrics cover Slurm jobs, GPU (utilization/memory/temperature/power/ECC/NVLink via DCGM),
node CPU/memory/disk/network, and CloudWatch (EC2/FSx/PCS). The stack installs on
node-local `/opt` (not the shared `/home`). Pre-built Grafana dashboards (Cluster Summary,
Slurm Detail, GPU Node List, GPU Health, Cluster Costs, Storage) are provisioned
automatically — see the [screenshot below](#accessing-grafana).

> **GPU metrics work out of the box across the supported GPU range** (Hopper / B200 /
> B300). `DcgmExporterImage` defaults to a DCGM 4.5.2 build pinned by digest, validated
> on 2× p6-b300 and on B200. The monitoring stack's own default (DCGM 4.2.0) tops out
> at B200 and can't pull newer NVCR tags on Docker 29.x — overriding via digest at the
> deploy-all level is what bridges that. Override `DcgmExporterImage` only if you need
> to pin to a different build; details:
> [OPERATIONS.md §3.1](./docs/OPERATIONS.md#31-dcgmexporterimage-the-default-and-when-to-change-it).

> **Prefer AWS-managed Prometheus/Grafana?** If you'd rather use Amazon Managed Service
> for Prometheus + Amazon Managed Grafana instead of the self-hosted stack on the login
> node, see [`4.validation_and_observability/4.prometheus-grafana`](../../4.validation_and_observability/4.prometheus-grafana).

**Monitoring-related parameters:**
- `DeployMonitoring` (default `true`)
- `MonitoringVersion` — [aws-parallelcluster-monitoring](https://github.com/aws-samples/aws-parallelcluster-monitoring) git ref (release tag, branch, or `latest`; default `v2.9.1`). Pinned to a tag so upstream changes can't break deployments unexpectedly. `v2.9.1` adds the `DCGM_EXPORTER_IMAGE` override (lets `DcgmExporterImage` enable B300 GPU metrics); `v2.6.4`+ carry the PCS fixes (node-local `/opt` install + Docker-29.x DCGM tag).
- `MonitoringRepo` — `owner/repo` to fetch from (default `aws-samples/aws-parallelcluster-monitoring`). Point at a fork + a branch in `MonitoringVersion` to test unreleased changes.
- `DcgmExporterImage` — dcgm-exporter image used on GPU nodes; defaults to a DCGM 4.5.2 build pinned by digest (covers Hopper/B200/B300). Override only if you need to pin to a different build (e.g. the older monitoring-default DCGM 4.2.0).

> Node type is identified by the `monitoring-role` tag (`login`/`compute`), not the EC2
> `Name` tag — the `Name` tag defaults to `PCS-<cngname>` and is free for you to retag.

### Accessing Grafana

Log in to Grafana as **`admin`**; the password is generated per cluster and stored in
SSM Parameter Store. Retrieve it (with `CLUSTER_ID` from the stack's `ClusterId` output):

```bash
aws ssm get-parameter --name "/pcs/${CLUSTER_ID}/grafana/admin-password" \
  --with-decryption --query 'Parameter.Value' --output text
```

There are two ways to reach the UI.

#### Option A — SSM port forwarding (default, private)

No public access required; works even when the login node has no inbound rules.

```bash
# Login node instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:aws:pcs:compute-node-group-name,Values=login" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)

# Port-forward remote 443 -> local 8443 (needs the Session Manager plugin)
aws ssm start-session --target $INSTANCE_ID \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["443"],"localPortNumber":["8443"]}'
```

Then open `https://localhost:8443/grafana/`.

#### Option B — Direct public access (opt-in, via `GrafanaPublicAccessCidr`)

To browse Grafana directly without port forwarding, set **`GrafanaPublicAccessCidr`** at
deploy time to a CIDR you trust (e.g. your office IP `203.0.113.4/32`). deploy-all then
creates a **login-only security group** that opens HTTPS/**443** to that CIDR and
attaches it to the login node, so you can open:

```
https://<login-node-public-ip>/grafana/
```

Get the login node's public IP from the EC2 console, or:

```bash
aws ec2 describe-instances \
  --filters "Name=tag:aws:pcs:compute-node-group-name,Values=login" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

Security notes:
- The security group is attached **only to the login node** — compute nodes and FSx
  (which share the cluster security group) are **not** exposed.
- **Opening 443 exposes more than Grafana.** The login node's nginx also reverse-proxies
  `/prometheus/`, `/pushgateway/`, and `/slurmexporter/`, and **those endpoints are
  unauthenticated**. Anyone who can reach the allowed CIDR can read all cluster metrics
  (and push to Pushgateway) without credentials — only the `/grafana/` path is
  password-gated. Treat this as exposing the whole monitoring stack, not just the Grafana
  login.
- Prefer a tight CIDR (a `/32` host or your VPN range). **`0.0.0.0/0` is accepted** — it
  can be convenient for a short-lived PoC or workshop where granting each user local SSM
  permissions is impractical — but it exposes the unauthenticated endpoints above to the
  whole internet. If you use it, narrow it to a real CIDR or clear it (Option A) as soon
  as you are done.
- The certificate is self-signed, so browsers show a warning — proceed past it, or put
  an ALB + ACM certificate in front for a trusted cert.
- Leaving `GrafanaPublicAccessCidr` empty (the default) keeps monitoring private; use
  Option A.

---

Once logged in, use the dashboard nav bar to switch between **Cluster Summary**,
**Slurm Detail**, **Compute Node List**, **GPU Node List**, **GPU Health**,
**Cluster Costs**, and **Storage**. For example, the **GPU Node List** shows each GPU
node's model, instance type, utilization, temperature, power, and memory:

![Grafana GPU Node List dashboard](images/dashboard-screenshot-gpu-list.png)

For detailed validation steps and the full test matrix (monitoring, containers, CPU/GPU,
NCCL, FSDP), see the [Test & Validation Guide](tests/README.md).

> **Use `v2.9.1` or newer for PCS.** Carries the PCS `/opt` install fix (v2.6.4),
> Docker-29.x DCGM tag (v2.6.5), Grafana 13 (v2.9), and the `DCGM_EXPORTER_IMAGE` override
> needed by `DcgmExporterImage` for B300 (v2.9.1). Migration notes:
> [OPERATIONS.md §3](./docs/OPERATIONS.md#3-monitoring-monitoringversion).

---

## 9. Templates

All templates live in [`assets/`](./assets/). `pcs-ml-cluster-deploy-all.yaml` nests
the others; you can also deploy each individually for more control (e.g. reuse a VPC/FSx
across clusters). Click **Deploy** to 1-click-launch a single template. For every
parameter and default, see [PARAMETERS.md](./docs/PARAMETERS.md).

| Template | Purpose | Deploy |
|---|---|---|
| [`pcs-ml-cluster-deploy-all.yaml`](./assets/pcs-ml-cluster-deploy-all.yaml) | All-in-one: Prerequisites + (optional AMI) + Cluster + login/CPU/GPU CNGs | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml&stackName=pcs-ml-cluster) |
| [`ml-cluster-prerequisites.yaml`](./assets/ml-cluster-prerequisites.yaml) | VPC, subnets, security groups, FSx for Lustre + OpenZFS | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/ml-cluster-prerequisites.yaml&stackName=pcs-prerequisites) |
| [`cluster.yaml`](./assets/cluster.yaml) | PCS cluster core (Slurm scheduler only, no nodes) | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/cluster.yaml&stackName=pcs-cluster) |
| [`add-cng.yaml`](./assets/add-cng.yaml) | Compute node group, single NIC — login nodes, CPU/single-NIC-GPU queues (C6i, G5, G6) | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/add-cng.yaml&stackName=pcs-add-cng) |
| [`add-cng-p5.yaml`](./assets/add-cng-p5.yaml) | P5/P5e/P5en nodes (16/32 EFA interfaces, by type) | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/add-cng-p5.yaml&stackName=pcs-add-cng-p5) |
| [`add-cng-p6-b200.yaml`](./assets/add-cng-p6-b200.yaml) | P6-B200 nodes (8 EFA interfaces) | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/add-cng-p6-b200.yaml&stackName=pcs-add-cng-p6-b200) |
| [`add-cng-p6-b300.yaml`](./assets/add-cng-p6-b300.yaml) | P6-B300 nodes (16 EFA interfaces) | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/add-cng-p6-b300.yaml&stackName=pcs-add-cng-p6-b300) |
| [`pcs-ready-dlami-with-enroot-pyxis.yaml`](./assets/pcs-ready-dlami-with-enroot-pyxis.yaml) | EC2 Image Builder: bake Enroot 3.5.0 + Pyxis 0.20.0 into the PCS-ready DLAMI | [<kbd>🚀</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ready-dlami-with-enroot-pyxis.yaml&stackName=pcs-dlami) |

`add-cng*` templates create a Slurm queue only when `QueueName` is set (leave it empty
for login nodes). The P-series templates need a `CapacityReservationId` when using a
Capacity Block.

---

## 10. Testing and Validation

Validated configurations:

- **Infrastructure** (`ml-cluster-prerequisites.yaml`, `cluster.yaml`): multiple Regions (us-east-1/us-west-2/us-east-2), Slurm 25.05 & 25.11.
- **CPU / single-NIC GPU** (`add-cng.yaml`): login (m6i.4xlarge), CPU (c6i.4xlarge), GPU (g6.xlarge/g6.12xlarge).
- **P5** (`add-cng-p5.yaml`): p5.48xlarge / p5en.48xlarge with ODCR and Capacity Blocks for ML.
- **P6-B300** (`add-cng-p6-b300.yaml`): validated on real p6-b300.48xlarge (Capacity Blocks, us-west-2) — 17 network cards, EFA active (NCCL `found 16 nics`), 2-node all_reduce ~761 GB/s peak, FSDP Llama-2 7B multi-node.
- **P6-B200** (`add-cng-p6-b200.yaml`): 8-network-card template (same EFA layout family as B300).
- **All-in-one** (`pcs-ml-cluster-deploy-all.yaml`): selects the P5/P6-B200/P6-B300 CNG template automatically from `PseriesInstanceType`; tested end-to-end with monitoring, container jobs, NCCL, and FSDP on p6-b300.
- **AMI builder** (`pcs-ready-dlami-with-enroot-pyxis.yaml`): Ubuntu 24.04 x86_64 AMIs with Enroot 3.5.0 + Pyxis 0.20.0, validated with PyTorch/CUDA containers.

---

## 11. Additional Resources

- [Operations guide](./docs/OPERATIONS.md) — version trade-offs, AMI single-version rule, monitoring/B300 dcgm setup, AMI pinning, FSx coupling, recommended production settings
- [Roadmap / TODO](./docs/ROADMAP.md) — implementation items under consideration
- [Parameter reference](./docs/PARAMETERS.md) — every deploy-all parameter and default
- [AWS Parallel Computing Service Documentation](https://docs.aws.amazon.com/pcs/)
- [AI/ML for AWS PCS Workshop](https://catalog.workshops.aws/ml-on-pcs/)
- [Slurm Documentation](https://slurm.schedmd.com/documentation.html)
- [Enroot](https://github.com/NVIDIA/enroot) · [Pyxis](https://github.com/NVIDIA/pyxis)
- [Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)
- [aws-parallelcluster-monitoring](https://github.com/aws-samples/aws-parallelcluster-monitoring) (upstream monitoring)
- [Prometheus & Grafana Setup](../../4.validation_and_observability/4.prometheus-grafana/README.md) (alternative: AWS-managed Prometheus/Grafana)
- [LDAP Server Setup Guide](../../1.architectures/6.ldap_server/README.md) — OpenLDAP for cluster-wide user management
