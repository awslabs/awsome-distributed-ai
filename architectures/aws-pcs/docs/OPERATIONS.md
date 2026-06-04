# Operations Guide — AWS PCS Reference Architecture

Constraints, recommended settings, and the *why* behind defaults that don't fit cleanly
in the parameter reference. Covers things that have bitten real deploys; the parameter
list itself lives in [`PARAMETERS.md`](./PARAMETERS.md), and verified results are in
[`../tests/README.md`](../tests/README.md).

If you're trying to deploy quickly, follow the [README Quick Start](../README.md#3-quick-start);
come here when something doesn't behave as expected, or when planning a production-grade run.

---

## 1. Slurm version selection

The templates' `SlurmVersion` parameter accepts **`25.05`** and **`25.11`**. The choice
affects more than just the scheduler binary:

| Capability | `25.05` | `25.11` |
|---|---|---|
| PCS cluster create + Pyxis container jobs | ✅ | ✅ |
| Slurm native OpenMetrics endpoint (port 6817) | ❌ | ✅ |
| Prometheus `slurm_openmetrics` job + Slurm dashboards in Grafana | ❌ | ✅ |
| Node/CPU/memory/GPU/CloudWatch dashboards | ✅ | ✅ |

The Slurm OpenMetrics endpoint requires `MetricsType=metrics/openmetrics` +
`CommunicationParameters=enable_http`, and **PCS rejects those settings on Slurm <
25.11** ("Slurm custom settings parameter MetricsType is only supported for Slurm
version 25.11 or later"). `cluster.yaml` therefore emits them only when
`SlurmVersion=25.11`. On a 25.05 cluster the rest of monitoring still works; only the
Slurm-specific dashboards stay empty. Pick 25.11 unless you have a reason to pin 25.05.

`24.11` is intentionally **out of scope** here — `cluster.yaml`'s `AllowedValues`
doesn't include it, and `install-enroot-pyxis.sh` builds Pyxis only for 25.05/25.11.

## 2. Container runtime: PostInstall vs. AMI build

Two paths install Enroot/Pyxis on a node:

- **PostInstall (default, `BuildAMI=false`)** — `PostInstallScriptUrl` runs
  `install-enroot-pyxis.sh` on every node at first boot. Adds ~2-3 min to boot but the
  cluster stack itself is faster to create (no Image Builder step).
- **AMI build (`BuildAMI=true`)** — `pcs-ready-dlami-with-enroot-pyxis.yaml` bakes
  Enroot/Pyxis into a custom DLAMI via Image Builder once, then nodes boot from it
  ready-to-go. Use `PostInstallScriptUrl=""` so the AMI's pre-baked installer doesn't run again
  (the installer is idempotent so leaving the default is harmless, but it adds minutes
  to every boot for nothing).

### 2.1 The AMI is single-Slurm-version, by design

Pyxis is a SPANK plugin and **its ABI is locked to the Slurm version it was compiled
against**. A `spank_pyxis.so` built for 25.11 makes a 25.05 slurmd refuse to start with
*"Incompatible Slurm plugin version"*. The DLAMI build template therefore takes a
`SlurmVersion` parameter and bakes Pyxis for **that one version only**. Use the same
`SlurmVersion` value on the AMI build stack and on the cluster stack, otherwise nodes
won't come up.

### 2.2 PostInstall passes the version via `PCS_SLURM_VERSION`

For the PostInstall path, `add-cng*.yaml`'s UserData exports
`PCS_SLURM_VERSION="${SlurmVersion}"` before invoking the script. The script can't
discover the cluster's Slurm version itself at first boot (cloud-init runs before
slurmd / `/etc/profile.d/slurm.sh` / the controller config exist), so it relies on this
explicit hand-off. When PCS adds a native post-install hook in the future, it should
expose the cluster Slurm version the same way.

If you run `install-enroot-pyxis.sh` manually with `PCS_SLURM_VERSION` unset, the
script falls back to building every supported version and using the newest installed
bin on slurmd's PATH — slower and less precise than the cluster path, but enough for a
manual node fix.

## 3. Monitoring (`MonitoringVersion`)

`MonitoringVersion` defaults to **`v2.9.1`**, which is what the GPU dashboards on
**p6-b300** require. Notable upstream changes since the older `v2.6.5`:

- **v2.9.1**: `dcgm-exporter` image is now configurable via `DCGM_EXPORTER_IMAGE`
  (lets `DcgmExporterImage` enable B300 GPU metrics without forking the monitoring repo).
- **v2.9**: Grafana **11 → 13**.
- **v2.7**: EFA fabric metrics + Cluster Logs dashboard.
- **v2.6.4**: node-local `/opt` install (fixes the shared-`/home` Stale-file-handle race).
- **v2.6.5**: DCGM exporter pin that pulls on Docker 29.x.

### 3.1 B300 GPU metrics need `DcgmExporterImage`

The default `dcgm-exporter` pin is DCGM 4.2.0 — that's the newest tag that pulls
cleanly on Docker 29.x (newer NVCR tags publish OCI image-index manifests Docker 29.x
can't pull), and it covers up to B200. **B300 needs DCGM ≥ 4.4.0**, so set
`DcgmExporterImage` to a B300-capable build by **digest** (a digest pull bypasses the
OCI-index issue):

```
DcgmExporterImage=nvcr.io/nvidia/k8s/dcgm-exporter@sha256:a7ad6547d4546eaf4dd5d6b4c0b4db4101e63ef7dc3cdff7f42b767d2c60b706
```

(linux/amd64 manifest for `4.5.2-4.8.1-ubuntu22.04`; an arm64 sibling digest exists for
Grace-based nodes when those land.) Validated on 2× p6-b300 with all 16 GPUs reporting
in Grafana. p5/p5e/p5en/p6-b200 don't need an override.

### 3.2 Public Grafana exposure

`GrafanaPublicAccessCidr` opens **TCP/443 on the login node** to a CIDR via a
login-only security group. The nginx in front of Grafana also proxies
**`/prometheus/`, `/pushgateway/`, `/slurmexporter/` without authentication** — opening
the CIDR exposes those too, not just password-gated Grafana. Use the tightest CIDR you
can. Empty (the default) means SSM port-forward only, which is the safest path for
production. `0.0.0.0/0` is accepted for short-lived PoC/workshop use; narrow it (or
clear it) when you're done.

## 4. AMI selection (`AmiId`) — pin in production

Left empty, `AmiId` resolves the PCS-ready DLAMI from the SSM public parameter
`/aws/service/pcs/ami/dlami-base-ubuntu2404/x86_64/latest/ami-id`. Only a `/latest/`
path is published, and CloudFormation re-resolves SSM parameter values on every stack
update — so a later scale-out can boot a *newer* AMI than the original nodes (drift).
For evaluation that's fine; for production:

```bash
aws ssm get-parameter --name /aws/service/pcs/ami/dlami-base-ubuntu2404/x86_64/latest/ami-id \
  --query 'Parameter.Value' --output text
```

then pass that exact `ami-xxx` as `AmiId` so every node in the cluster's lifetime is
identical.

## 5. FSx storage: deployment type vs. throughput coupling

Lustre `PerUnitStorageThroughput` and OpenZFS `HomeThroughput` allowed values **depend
on the deployment type**:

| Filesystem | DeploymentType | Valid throughput |
|---|---|---|
| Lustre | `PERSISTENT_2` (default) | 125 / 250 / 500 / 1000 MB/s/TiB |
| Lustre | `PERSISTENT_1` (older Regions) | 50 / 100 / 200 MB/s/TiB |
| OpenZFS | `SINGLE_AZ_HA_2` (default) | 160 / 320 / 640 / 1280 / 2560 / 3840 / 5120 / 7680 MB/s |
| OpenZFS | `SINGLE_AZ_HA_1` / `SINGLE_AZ_2` / `SINGLE_AZ_1` | 64 / 128 / 192 / 256 / 384 / 512 / 768 / 1024 MB/s |

The templates enforce the valid pair via CloudFormation `Rules` so a mismatch fails at
stack-create time with a clear message instead of deep in the nested FSx stack. If you
change `LustreDeploymentType` or `OpenZFSDeploymentType`, also pick a throughput valid
for the new type. The defaults (250 / 320) target `PERSISTENT_2` / `SINGLE_AZ_HA_2`;
override both throughput parameters when falling back to the older deployment types.

## 6. P6-B300 NIC topology — single-template lock-in

`add-cng-p6-b300.yaml` is a hand-built `NetworkInterfaces` block for **exactly
`p6-b300.48xlarge`** (17 cards: card 0 ENA-only + EFA on cards 1-16 at `DeviceIndex 0`,
which differs from p5/b200's "card 0 = EFA, DeviceIndex 1" layout). The instance type
is locked via `AllowedValues` so a different type can't be selected through this
template — pick the matching `add-cng-*.yaml` for the family you want.

## 7. Recommendations recap

For a new production deploy:

- `SlurmVersion=25.11` (full monitoring coverage)
- `MonitoringVersion=v2.9.1` (default; carries the B300 / `/opt` install / Docker 29.x fixes)
- `AmiId` pinned to a resolved AMI ID, not left empty
- `BuildAMI=true` for frequent scaling (~3 min boot vs ~6 min) — pair with matching `SlurmVersion`
- `DcgmExporterImage` set to a digest only on **p6-b300**; leave empty otherwise
- Minimum-CIDR `GrafanaPublicAccessCidr` if used at all; otherwise empty (SSM port-forward)
- Throughput values that match the chosen FSx deployment types
