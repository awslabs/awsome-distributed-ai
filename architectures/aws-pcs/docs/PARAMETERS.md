# Parameter Reference — `pcs-ml-cluster-deploy-all.yaml`

Full parameter list for the all-in-one deployment template, grouped to match the
sections shown in the CloudFormation console (the console also shows friendly labels via
`AWS::CloudFormation::Interface`). Defaults give the most common production setup —
`BuildAMI=false` + Enroot/Pyxis via `PostInstallScriptUrl` + `DeployMonitoring=true` —
so a default deploy only needs the Availability Zone (`PrimarySubnetAZ`).

For conceptual guidance (GPU instance/EFA selection, FSx Region availability, container
runtime options), see the [README](../README.md#4-configuration).

## 1. Network Configuration

| Parameter | Default | Purpose |
|---|---|---|
| `PrimarySubnetAZ` | *(required)* | Availability Zone to deploy into — the one required parameter |
| `VPCName` | `ML-Cluster-VPC` | Name for the created VPC |
| `CreateS3Endpoint` | `true` | Create an S3 VPC endpoint |

## 2. PCS Cluster Configuration

| Parameter | Default | Purpose |
|---|---|---|
| `LoginNodeInstanceType` | `m6i.4xlarge` | Login node instance type |
| `SlurmVersion` | `25.11` | Slurm version (`25.05` or `25.11`). Drives which monitoring you get (Slurm OpenMetrics is 25.11+ only) and is also threaded into the AMI build / CNG UserData so the right-version Pyxis is installed; see [OPERATIONS.md §1](./OPERATIONS.md#1-slurm-version-selection) |
| `DeployMonitoring` | `true` | Deploy Prometheus/Grafana/DCGM on the login node |
| `GrafanaPublicAccessCidr` | *(empty)* | When set to a CIDR, opens HTTPS/443 on the login node to that CIDR via a login-only security group. Empty = SSM port-forward only. **443 also exposes the unauthenticated `/prometheus/`, `/pushgateway/`, `/slurmexporter/` proxy paths**, not just the password-gated Grafana. Use the tightest CIDR you can; `0.0.0.0/0` is accepted for short-lived PoC/workshop use but exposes those endpoints to the whole internet |
| `ManagedAccounting` | `disabled` | Enable Slurm managed accounting (requires Slurm 24.11+) |
| `AccountingPolicyEnforcement` | `none` | Slurm accounting policy enforcement (`none` or `associations,limits,safe`) |

## 3. Container Runtime (Post-install Script)

| Parameter | Default | Purpose |
|---|---|---|
| `PostInstallScriptUrl` | Enroot/Pyxis installer | HTTP(S) script run on every node at first boot (PCS equivalent of ParallelCluster `OnNodeConfigured`). Empty = skip; or override with any other HTTP(S) script. Idempotent under `BuildAMI=true` (no-op when already pre-baked) |
| `PostInstallScriptArgs` | *(empty)* | Arguments passed to the post-install script |
| `RootVolumeSize` | `300` | Node root volume (GiB); 300 leaves room for large container images (Megatron `.sqsh` ~20 GB) |

## 4. On-Demand Compute Node Group (CPU)

| Parameter | Default | Purpose |
|---|---|---|
| `DeployOnDemandCNG` | `true` | Deploy the CPU queue |
| `OnDemandInstanceType` | `c6i.4xlarge` | CPU queue instance type |
| `OnDemandMinCount` | `0` | CPU queue minimum nodes (0 = dynamic scaling) |
| `OnDemandMaxCount` | `4` | CPU queue maximum nodes |
| `OnDemandCngName` | `cpu1` | CPU node-group name |
| `OnDemandQueueName` | `cpu1` | CPU Slurm queue name |

## 5. GPU Compute Node Group — P5/P6 (Optional)

See [GPU compute](../README.md#gpu-compute-p5p6) for instance/EFA/capacity guidance.

| Parameter | Default | Purpose |
|---|---|---|
| `DeployPseriesCNG` | `false` | Deploy a GPU (P5/P6) queue |
| `PseriesInstanceType` | `p5.48xlarge` | GPU instance type; selects the matching multi-NIC template **and** EFA interface count automatically |
| `PseriesMinCount` | `0` | GPU queue minimum nodes |
| `PseriesMaxCount` | `4` | GPU queue maximum nodes |
| `CapacityReservationId` | *(empty)* | Capacity **Block** reservation ID (sets `MarketType=capacity-block`). Leave empty for On-Demand / ODCR — **do not** put an ODCR ID here |
| `PseriesCngName` | `gpu-p5` | GPU node-group name |
| `PseriesQueueName` | `gpu-p5` | GPU Slurm queue name |

## 6. FSx Storage Configuration (Advanced)

See [Storage: FSx deployment types](../README.md#storage-fsx-deployment-types-region-availability) for Region availability.

| Parameter | Default | Purpose |
|---|---|---|
| `Capacity` | `1200` | FSx Lustre capacity (GiB; 1200 or increments of 2400) |
| `LustreDeploymentType` | `PERSISTENT_2` | Lustre deployment type (`PERSISTENT_2` / `PERSISTENT_1`) — Region-dependent |
| `PerUnitStorageThroughput` | `250` | Lustre throughput (MB/s/TiB); valid values depend on the deployment type |
| `Compression` | `LZ4` | Lustre data compression (`LZ4` / `NONE`) |
| `LustreVersion` | `2.15` | Lustre software version (`2.15` / `2.12`) |
| `HomeCapacity` | `512` | FSx OpenZFS (`/home`) capacity (GiB) |
| `HomeThroughput` | `320` | FSx OpenZFS (`/home`) throughput (MB/s) |
| `OpenZFSDeploymentType` | `SINGLE_AZ_HA_2` | OpenZFS deployment type (`SINGLE_AZ_HA_2` / `SINGLE_AZ_HA_1` / `SINGLE_AZ_2` / `SINGLE_AZ_1`) — Region-dependent |

## 7. Custom AMI Build (Optional)

Skip unless you need a pre-baked DLAMI (faster scale-out, deterministic state).
The default first-boot install (`PostInstallScriptUrl` in §3) is what almost
every cluster wants.

| Parameter | Default | Purpose |
|---|---|---|
| `BuildAMI` | `false` | Pre-bake Enroot/Pyxis into a custom DLAMI via Image Builder (~30 min). When `true`, set `PostInstallScriptUrl=""` for the cleanest boot. The AMI is **single-Slurm-version** by design — match `SlurmVersion`. See [OPERATIONS.md §2](./OPERATIONS.md#2-container-runtime-postinstall-vs-ami-build) |
| `BaseAmiId` | *(auto)* | Base AMI for the custom build; empty = auto-resolve from SSM (only used when `BuildAMI=true`) |
| `SemanticVersion` | `1.0.0` | Image Builder recipe version (only used when `BuildAMI=true`) |
| `BuildSchedule` | `Manual` | AMI build cadence: `Manual` / `Weekly` / `Monthly` (only used when `BuildAMI=true`) |

## 8. Developer / Advanced

| Parameter | Default | Purpose |
|---|---|---|
| `S3BucketName` | `awsome-distributed-ai` | S3 bucket the nested templates are fetched from |
| `S3KeyPrefix` | `templates/` | S3 key prefix for the nested templates |
| `MonitoringVersion` | `v2.9.1` | [aws-parallelcluster-monitoring](https://github.com/aws-samples/aws-parallelcluster-monitoring) git ref (release tag, branch, or `latest`). `v2.9.1` adds the `DCGM_EXPORTER_IMAGE` override (needed for B300 GPU metrics) and brings Grafana 13; `v2.6.4`+ carry the PCS `/opt` install + Docker-29.x DCGM fixes. Pin to a tag for stability. Migration notes: [OPERATIONS.md §3](./OPERATIONS.md#3-monitoring-monitoringversion) |
| `MonitoringRepo` | `aws-samples/aws-parallelcluster-monitoring` | GitHub `owner/repo` for the monitoring stack; override with a fork + a branch in `MonitoringVersion` to test unreleased changes |
| `DcgmExporterImage` | DCGM 4.5.2 by digest | `dcgm-exporter` image used on GPU nodes. Defaults to a DCGM 4.5.2 build pinned by digest (`nvcr.io/nvidia/k8s/dcgm-exporter@sha256:a7ad6547...`) covering Hopper / B200 / B300. The digest pull bypasses the Docker-29.x OCI-index failure on newer NVCR tags. Override (any image reference, ideally also a digest) to pin to a different build — e.g. the monitoring stack's older default 4.2.0. No effect on CPU nodes. See [OPERATIONS.md §3.1](./OPERATIONS.md#31-dcgmexporterimage-the-default-and-when-to-change-it) |
