# Parameter Reference — `pcs-ml-cluster-deploy-all.yaml`

Full parameter list for the all-in-one deployment template, grouped to match the
sections shown in the CloudFormation console (the console also shows friendly labels via
`AWS::CloudFormation::Interface`). Defaults give the most common production setup —
the latest PCS-Ready Deep Learning AMI auto-resolved from SSM, Enroot/Pyxis installed
at first boot via `PostInstallScriptUrl`, monitoring enabled — so a default deploy
only needs the Availability Zone (`PrimarySubnetAZ`). To pre-bake Enroot/Pyxis into
a custom AMI for faster boots, build it separately with
[`pcs-ready-dlami-with-enroot-pyxis.yaml`](../README.md#84-pre-baking-enrootpyxis-into-a-custom-ami)
and pass its output as `AmiId`.

For conceptual guidance (GPU instance/EFA selection, FSx Region availability, container
runtime options), see the [README](../README.md#4-configuration).

## 1. Network Configuration

| Parameter | Default | Purpose |
|---|---|---|
| `PrimarySubnetAZ` | *(required)* | Availability Zone to deploy into — the one required parameter. Holds the public subnet (login node), the primary private subnet (compute, FSx), and the single NAT gateway |
| `AdditionalSubnetAZ2` | *(empty)* | (Optional) 2nd AZ for an additional private subnet. Empty = single-AZ. Enables multi-AZ layouts (e.g. OpenZFS `MULTI_AZ`). Shares the primary AZ's NAT gateway (cross-AZ egress, no per-AZ NAT) |
| `AdditionalSubnetAZ3` | *(empty)* | (Optional) 3rd AZ for an additional private subnet. Requires `AdditionalSubnetAZ2` to also be set. Max 3 private AZs total |
| `VPCName` | *(empty → `${StackName}-VPC`)* | Name for the created VPC. Empty (default) auto-derives from the stack name so multiple deployments in one account get unique names |
| `CreateS3Endpoint` | `true` | Create an S3 VPC endpoint |

## 2. PCS Cluster Configuration

| Parameter | Default | Purpose |
|---|---|---|
| `SlurmVersion` | `25.11` | Slurm version (`25.05` or `25.11`). Drives which monitoring you get (Slurm OpenMetrics is 25.11+ only) and is also threaded into the CNG UserData so the right-version Pyxis is installed; see [OPERATIONS.md §1](./OPERATIONS.md#1-slurm-version-selection) |
| `LoginNodeInstanceType` | `m6i.4xlarge` | Login node instance type |
| `RootVolumeSize` | `300` | Root EBS volume size (GiB) on every node (login + compute); 300 leaves room for large container images (Megatron `.sqsh` ~20 GB) |
| `AmiId` | *(empty → SSM auto-resolve)* | AMI ID for every node group. **Empty (default) auto-resolves to the latest PCS-Ready Deep Learning AMI** (Ubuntu 24.04, x86_64) from SSM (`/aws/service/pcs/ami/dlami-base-ubuntu2404/x86_64/latest/ami-id`). For production, **pin to a specific `ami-xxx`** so a later scale-out cannot drift onto a newer base. Use a custom AMI built off the PCS-Ready DLAMI base (e.g. via [`pcs-ready-dlami-with-enroot-pyxis.yaml`](../README.md#84-pre-baking-enrootpyxis-into-a-custom-ami)) when you want Enroot/Pyxis pre-baked or other customizations. See [OPERATIONS.md §4](./OPERATIONS.md#4-ami-selection-amiid--pin-in-production) |
| `SSHAccessCidr` | *(empty)* | When set to a CIDR, opens SSH/22 on the login node to that CIDR via a login-only security group (attached to the login node only, never compute). Empty (default) = SSH over SSM only. Set to your office/VPN range for direct `ssh`/`scp`/VS Code Remote (common for multi-user clusters) |
| `ManagedAccounting` | `disabled` | Enable Slurm managed accounting (requires Slurm 24.11+) |
| `AccountingPolicyEnforcement` | `none` | Slurm accounting policy enforcement (`none` or `associations,limits,safe`) |

## 2b. Additional Cluster Configuration (Monitoring, Multi-User)

| Parameter | Default | Purpose |
|---|---|---|
| `MonitoringStack` | `Prometheus-LoginNode` | Monitoring stack to deploy. `Prometheus-LoginNode` = self-hosted Prometheus + Grafana + DCGM Exporter on the login node. `none` = no monitoring. (Renamed from the old boolean `DeployMonitoring`; `<what>-<where>` enum, extensible to future `AMP-AMG`/`CloudWatch`) |
| `GrafanaAccessCidr` | *(empty)* | When set to a CIDR, opens HTTPS/443 (Grafana) on the login node to that CIDR via the login-only security group. Empty = SSM port-forward only. **443 also exposes the unauthenticated `/prometheus/`, `/pushgateway/`, `/slurmexporter/` proxy paths**, not just the password-gated Grafana. Use the tightest CIDR you can. (Renamed from `GrafanaPublicAccessCidr`) |
| `DirectoryService` | `none` | Multi-user directory. `none` = single `ubuntu` user. `OpenLDAP-LoginNode` = slapd on the login node (DB on shared `/home/ldap-db`) + SSSD on all compute nodes. **Single login node only** — keep the login node group at 1 instance while enabled. See [USER-MANAGEMENT.md](./USER-MANAGEMENT.md) |
| `DirectoryDomainSuffix` | `dc=cluster,dc=internal` | LDAP domain suffix. Only used when `DirectoryService != none` |

## 3. Container Runtime (Post-install Script)

| Parameter | Default | Purpose |
|---|---|---|
| `PostInstallScriptUrl` | Enroot/Pyxis installer | HTTP(S) script run on every node at first boot (PCS equivalent of ParallelCluster `OnNodeConfigured`). Empty = skip; or override with any other HTTP(S) script. Idempotent: a no-op if Enroot/Pyxis is already pre-baked into `AmiId` |
| `PostInstallScriptArgs` | *(empty)* | Arguments passed to the post-install script |

## 4. On-Demand Compute Node Group (CPU)

| Parameter | Default | Purpose |
|---|---|---|
| `DeployOnDemandCNG` | `true` | Deploy the CPU queue |
| `OnDemandInstanceType` | `c6i.4xlarge` | CPU queue instance type |
| `OnDemandMinCount` | `0` | CPU queue minimum nodes (0 = dynamic scaling) |
| `OnDemandMaxCount` | `4` | CPU queue maximum nodes |
| `OnDemandCngName` | `cpu1` | CPU node-group name |
| `OnDemandQueueName` | `cpu1` | CPU Slurm queue name |
| `OnDemandEnableEfa` | `false` | Enable EFA on the CPU CNG (HPC/MPI workloads on hpc6a/hpc7a/hpc6id/hpc8a, c7i.metal, etc.). Switches the CNG's LaunchTemplate to a `NetworkInterfaces` block with `InterfaceType=efa` and wires in a cluster placement group. No effect on the GPU CNG. See [README §EFA on CPU HPC instances](../README.md#efa-on-cpu-hpc-instances-ondemandenableefa) |
| `OnDemandEfaInterfaceCount` | `0` (auto) | Number of EFA interfaces, used only when `OnDemandEnableEfa=true`. **`0` (default) auto-derives from `OnDemandInstanceType`** — `hpc8a.96xlarge`=2, `hpc7a.{96,48,24,12}xlarge`=2, `hpc6id.32xlarge`=2, `hpc6a.48xlarge`=1, `c7i.metal-*`=1, any other type=1. **Only enable EFA on an EFA-capable type** (hpc6/hpc7/hpc8 family + select metal); on a non-EFA type (e.g. `c6i.4xlarge`) the launch fails regardless. Override with `1`/`2` only to pin a value that differs from the auto map (e.g. a new HPC type) |
| `OnDemandPlacementGroupName` | *(empty)* | Existing cluster placement group name to launch nodes into. Empty + `OnDemandEnableEfa=true` auto-creates a per-CNG cluster placement group; supplying a name reuses an existing one (e.g. shared across CPU + GPU CNGs for heterogeneous tightly-coupled jobs). Ignored when `OnDemandEnableEfa=false` |

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

## 6. FSx for Lustre (`/fsx`) + FSx for OpenZFS (`/home`) (Advanced)

See [Storage: FSx deployment types](../README.md#storage-fsx-deployment-types-region-availability) for Region availability.

| Parameter | Default | Purpose |
|---|---|---|
| `Capacity` | `1200` | FSx for Lustre (`/fsx`) capacity (GiB; 1200 or increments of 2400) |
| `LustreDeploymentType` | `PERSISTENT_2` | FSx for Lustre (`/fsx`) deployment type (`PERSISTENT_2` / `PERSISTENT_1`) — Region-dependent |
| `PerUnitStorageThroughput` | `250` | FSx for Lustre (`/fsx`) throughput (MB/s/TiB); valid values depend on the deployment type |
| `Compression` | `LZ4` | FSx for Lustre (`/fsx`) data compression (`LZ4` / `NONE`) |
| `LustreVersion` | `2.15` | FSx for Lustre (`/fsx`) software version (`2.15` / `2.12`) |
| `FSxLustreEnableEfa` | `false` | Enable EFA on the FSx for Lustre filesystem. **The headline feature is GPUDirect Storage (GDS) for P5/P5e/P5en/P6-B200 GPU clients**, which DMAs file data straight into GPU memory (requires the NVIDIA `nvidia-fs` / cuFile stack on the client — tracked as a follow-up in [docs/ROADMAP.md](./ROADMAP.md#client-side-lustre-on-efa--gds-support)). EFA-capable CPU CNGs (`OnDemandEnableEfa=true`) get the EFA *transport* path to storage as a secondary benefit, useful when a single client is pushing past ~10 GBps. **PERSISTENT_2 SSD only** — a CFN Rule on the prerequisites template fails the stack at create time when combined with PERSISTENT_1 (rather than silently ignoring the opt-in). **Requires a much larger `Capacity` than non-EFA**: at `PerUnitStorageThroughput=250` the minimum is **19200 GiB** (16× the 1200 GiB non-EFA default). The full minimum-capacity matrix per throughput tier is in the [FSx for Lustre User Guide](https://docs.aws.amazon.com/fsx/latest/LustreGuide/efa.html). The FSx side rejects undersized capacity at stack-create time with a clear error |
| `HomeCapacity` | `512` | FSx for OpenZFS (`/home`) capacity (GiB) |
| `HomeThroughput` | `320` | FSx for OpenZFS (`/home`) throughput (MB/s) |
| `OpenZFSDeploymentType` | `SINGLE_AZ_HA_2` | FSx for OpenZFS (`/home`) deployment type (`SINGLE_AZ_HA_2` / `SINGLE_AZ_HA_1` / `SINGLE_AZ_2` / `SINGLE_AZ_1`) — Region-dependent |

## 7. Developer / Advanced

| Parameter | Default | Purpose |
|---|---|---|
| `S3BucketName` | `awsome-distributed-ai` | S3 bucket the nested templates are fetched from |
| `S3KeyPrefix` | `templates/` | S3 key prefix for the nested templates |
| `MonitoringVersion` | `v2.9.1` | [aws-parallelcluster-monitoring](https://github.com/aws-samples/aws-parallelcluster-monitoring) git ref (release tag, branch, or `latest`). `v2.9.1` adds the `DCGM_EXPORTER_IMAGE` override (needed for B300 GPU metrics) and brings Grafana 13; `v2.6.4`+ carry the PCS `/opt` install + Docker-29.x DCGM fixes. Pin to a tag for stability. Migration notes: [OPERATIONS.md §3](./OPERATIONS.md#3-monitoring-monitoringversion) |
| `MonitoringRepo` | `aws-samples/aws-parallelcluster-monitoring` | GitHub `owner/repo` for the monitoring stack; override with a fork + a branch in `MonitoringVersion` to test unreleased changes |
| `DcgmExporterImage` | DCGM 4.5.2 by digest | `dcgm-exporter` image used on GPU nodes. Defaults to a DCGM 4.5.2 build pinned by digest (`nvcr.io/nvidia/k8s/dcgm-exporter@sha256:a7ad6547...`) covering Hopper / B200 / B300. The digest pull bypasses the Docker-29.x OCI-index failure on newer NVCR tags. Override (any image reference, ideally also a digest) to pin to a different build — e.g. the monitoring stack's older default 4.2.0. No effect on CPU nodes. See [OPERATIONS.md §3.1](./OPERATIONS.md#31-dcgmexporterimage-the-default-and-when-to-change-it) |
