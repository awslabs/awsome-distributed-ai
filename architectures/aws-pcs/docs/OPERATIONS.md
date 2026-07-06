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

## 2. AMI and container runtime

This section covers the node image (`AmiId`) and how the Enroot/Pyxis container
runtime gets onto it — two facets of the same decision.

### 2.1 Container runtime: PostInstall vs. pre-baked AMI

Two paths install Enroot/Pyxis on a node. They are **decoupled**: the cluster stack
(`pcs-ml-cluster-deploy-all.yaml`) does not run Image Builder — it accepts a ready
`AmiId`. To get a pre-baked AMI you run `pcs-ready-dlami-with-enroot-pyxis.yaml` once
as a separate stack, then pass its output to the cluster.

- **PostInstall (default)** — `PostInstallScriptUrl` runs `install-enroot-pyxis.sh`
  on every node at first boot. Adds ~2-3 min to boot but the cluster stack itself is
  faster to create (no Image Builder step). This is what `AmiId=""` (default) plus
  the default `PostInstallScriptUrl` delivers.
- **Pre-baked AMI** — build the AMI separately (see the README's
  *Pre-baking Enroot/Pyxis into a custom AMI* section), then pass its `ami-xxx` as
  the cluster's `AmiId`. Set `PostInstallScriptUrl=' '` (a single space) for the
  cleanest boot — that skips the Enroot/Pyxis install entirely. (Leaving it at the
  default — empty, which auto-installs from the templates bucket — also works on a
  pre-baked AMI: the installer is idempotent and detects Enroot/Pyxis is already
  present, a fast no-op; the single space just avoids the download+check.)

### 2.2 The AMI is single-Slurm-version, by design

Pyxis is a SPANK plugin and **its ABI is locked to the Slurm version it was compiled
against**. A `spank_pyxis.so` built for 25.11 makes a 25.05 slurmd refuse to start with
*"Incompatible Slurm plugin version"*. The DLAMI build template therefore takes a
`SlurmVersion` parameter and bakes Pyxis for **that one version only**. Use the same
`SlurmVersion` value on the AMI build stack and on the cluster stack, otherwise nodes
won't come up.

### 2.3 PostInstall passes the version via `PCS_SLURM_VERSION`

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

### 2.4 Why a single version is fine across cluster upgrades

Pinning Pyxis and slurmd's PATH to one Slurm version sounds brittle, but Slurm's
[upgrade policy](https://slurm.schedmd.com/upgrades.html) makes it safe in practice:

> Slurm has long supported in-place upgrades from the previous two major releases.
> Slurm 24.11 introduced compatibility with the previous three major releases.

So `scontrol`/`srun` from version *N* interoperate with a `slurmctld` of *N+1* (or
*N+2/N+3* on 24.11+) — a cluster upgrade does not break nodes built for the prior
version. When you do want to advance, set the new `SlurmVersion` and redeploy: the
PostInstall path rebuilds Pyxis for the new version on first boot, and the AMI build
path bakes a new AMI for it. Nothing dynamic is needed on the running node side.

The single-version pin is also why the AMI is *not* a one-AMI-fits-all artifact —
treat the AMI as bound to a specific cluster `SlurmVersion`, the same way you would
treat a binary built against a particular libc.

### 2.5 AMI selection (`AmiId`) — pin in production

Left empty, `AmiId` resolves the PCS-Ready DLAMI from the SSM public parameter
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

## 3. Monitoring (`MonitoringVersion`)

`MonitoringVersion` defaults to **`v2.10.2`**. Notable upstream changes since the
older `v2.6.5`:

- **v2.10.2**: `refresh-ec2-credentials.sh` also restarts **prometheus** on IMDS
  credential rotation. Without it, ~6 h after boot Prometheus's `ec2_sd` keeps a
  cached (now-expired) token, `DescribeInstances` starts returning
  `RequestExpired`, and compute node/GPU metrics silently stop being collected
  while the login-local targets still report. Earlier tags (incl. `v2.9.1`) are
  affected; pin to `v2.10.2`+ to avoid it.
- **v2.10**: Amazon RES VDI monitoring + GPU clock fix.
- **v2.9.1**: `dcgm-exporter` image is now configurable via `DCGM_EXPORTER_IMAGE`
  (lets `DcgmExporterImage` enable B300 GPU metrics without forking the monitoring repo).
- **v2.9**: Grafana **11 → 13**.
- **v2.7**: EFA fabric metrics + Cluster Logs dashboard.
- **v2.6.4**: node-local `/opt` install (fixes the shared-`/home` Stale-file-handle race).
- **v2.6.5**: DCGM exporter pin that pulls on Docker 29.x.

### 3.1 `DcgmExporterImage` — the default, and when to change it

The templates default `DcgmExporterImage` to a **DCGM 4.5.2 build pinned by digest**:

```
nvcr.io/nvidia/k8s/dcgm-exporter@sha256:a7ad6547d4546eaf4dd5d6b4c0b4db4101e63ef7dc3cdff7f42b767d2c60b706
```

This is the linux/amd64 manifest for `4.5.2-4.8.1-ubuntu22.04`. It covers Hopper /
B200 / B300 with the same image — validated on 2× p6-b300 (all 16 GPUs reporting in
Grafana) and aligns with the upstream DCGM changelog (no `DCGM_FI_DEV_*` field
removals or B200 regressions between 4.2.0 and 4.5.2).

**Why a digest, not a tag.** The monitoring stack's *own* default is the older
4.2.0 tag because newer NVCR tags publish an OCI image-index manifest that Docker 29.x
on the PCS-Ready DLAMI can't pull (`error from registry: Incorrect Repository Format`). A
digest pull bypasses that index negotiation, so we can ship a newer DCGM safely.
4.2.0 also caps coverage at B200 — too restrictive for this repo's GPU range.

**When you might want to override.** Set `DcgmExporterImage` to a different image
reference (preferably also a digest) if you have a reason to deviate:

| Goal | Set `DcgmExporterImage` to |
|---|---|
| Pin to the monitoring stack's older 4.2.0 (e.g. matching another fleet) | `nvcr.io/nvidia/k8s/dcgm-exporter:4.2.0-4.1.0-ubuntu22.04` |
| Test a newer DCGM | The new build's digest, e.g. `nvcr.io/.../dcgm-exporter@sha256:<newer>` |
| Use an arm64 build (Grace-based nodes once supported) | The arm64 sibling digest of the same release |

For all the standard x86_64 P-family clusters this PR targets, the default is what you
want — leave it alone.

### 3.2 Monitoring across a login-node stop / replacement

Monitoring (Prometheus + Grafana + the exporters) runs on the **login node**, so
it's worth knowing what happens when that node is stopped or replaced (PCS replaces
an unhealthy login node with a new instance that has a **new private IP** — see also
the directory-service equivalent in [USER-MANAGEMENT.md](./USER-MANAGEMENT.md)).
Most of the stack is **replacement-safe by design**; one thing is not.

**Survives a replacement (no action needed):**
- **Compute metrics keep flowing.** Prometheus *pulls* (scrapes) compute exporters;
  compute nodes never push and never store the login IP. The new login node's
  Prometheus re-discovers the running compute nodes automatically via **EC2 service
  discovery** filtering on the `aws:pcs:cluster-id` tag (10 s refresh) — there is no
  compute→login IP dependency to break (unlike SSSD's pinned `ldap_uri`). Verified on
  a real replacement: the new login's Prometheus re-listed the compute/slurm targets
  as `up` within seconds.
- **Grafana admin password.** Stored in SSM at `/pcs/<cluster-id>/grafana/admin-password`
  and **reused** on replacement (the installer only generates one on first boot), so
  the password you fetched still works.
- **The built-in dashboards.** Grafana's dashboards and the Prometheus datasource are
  **file-provisioned** (`type: file`, datasource `url: http://localhost:9090`), so the
  ~13 stock dashboards re-appear automatically on the new node and the datasource still
  points at the co-located Prometheus.

**Lost on a replacement (known limitation):**
- **Historical metrics and any hand-made Grafana changes.** The Prometheus TSDB and
  the Grafana DB live in **node-local Docker named volumes** (`/var/lib/docker/volumes/...`
  — deliberately node-local to avoid the shared-`/home` Stale-file-handle race that
  drove the `/opt` install). They are **not** on shared/persistent storage, so a
  replacement starts both fresh: you lose the pre-replacement time-series history and
  any **user-created or user-edited** Grafana state (custom dashboards, edits to the
  stock dashboards — they are `editable`, annotations, alert rules). New metrics
  collect normally from the moment the new node is up.

If long-term metric retention or custom dashboards matter for your cluster, export
dashboards to JSON (or keep them in the provisioning tree) and treat the on-node TSDB
as ephemeral. Persisting the TSDB/Grafana DB across replacement (shared FS, EBS
snapshot, or a managed AMP/AMG backend) is tracked in [ROADMAP.md](./ROADMAP.md).

### 3.3 Public Grafana exposure

`GrafanaAccessCidr` opens **TCP/443 on the login node** to a CIDR via a
login-only security group. The nginx in front of Grafana also proxies
**`/prometheus/`, `/pushgateway/`, `/slurmexporter/` without authentication** — opening
the CIDR exposes those too, not just password-gated Grafana. Use the tightest CIDR you
can. Empty (the default) means SSM port-forward only, which is the safest path for
production. `0.0.0.0/0` is accepted for short-lived PoC/workshop use; narrow it (or
clear it) when you're done.

## 4. FSx storage: deployment type vs. throughput coupling

Lustre `PerUnitStorageThroughput` and OpenZFS `HomeThroughput` allowed values **depend
on the deployment type**:

| Filesystem | DeploymentType | Valid throughput |
|---|---|---|
| Lustre | `PERSISTENT_2` (default) | 125 / 250 / 500 / 1000 MB/s/TiB |
| Lustre | `PERSISTENT_1` (older Regions) | 50 / 100 / 200 MB/s/TiB |
| OpenZFS | `SINGLE_AZ_2` / `SINGLE_AZ_HA_2` (default) | 160 / 320 / 640 / 1280 / 2560 / 3840 / 5120 / 7680 / 10240 MB/s |
| OpenZFS | `SINGLE_AZ_HA_1` / `SINGLE_AZ_1` | 64 / 128 / 256 / 512 / 1024 / 2048 / 3072 / 4096 MB/s |

The templates enforce the valid pair via CloudFormation `Rules` so a mismatch fails at
stack-create time with a clear message instead of deep in the nested FSx stack. If you
change `LustreDeploymentType` or `OpenZFSDeploymentType`, also pick a throughput valid
for the new type. The defaults (250 / 320) target `PERSISTENT_2` / `SINGLE_AZ_HA_2`;
override both throughput parameters when falling back to the older deployment types.

### 4.1 Lustre mount options — `noatime`

The CNG templates mount FSx for Lustre with `-o noatime,flock,lazystatfs`. Of
these, `flock` and `lazystatfs` are already enabled by default on FSx for
Lustre 2.15 (verified via `/proc/mounts`), so the explicit options serve as
documentation and forward-compatibility. The only *net-new* behaviour is
**`noatime`**.

**What `noatime` does:** suppresses updating the access timestamp on every
`read()`. Without it, every file read triggers a metadata RPC to the MDS to
record the access time — even when the file content is served from OST cache.

**When it matters:** under concurrent multi-node read workloads (distributed
training data loading, HuggingFace cache reads, Python imports from shared
`/fsx`). With 16+ concurrent readers the atime RPCs add measurable MDS load;
at 64+ nodes it can become a throughput bottleneck.

**Benchmark (us-east-2, 2026-06-11):**

| Environment | relatime (default) | noatime | Delta |
|---|---|---|---|
| 1 node, 10K stat | 1851 ms | 1930 ms | ±noise |
| 4 nodes × 4 procs (16 streams), 10K stat | 5033 ms | 4812 ms | **-4.4%** |

On a small filesystem (1.2 TiB / 2 OSTs) the MDS is nowhere near saturated,
so the improvement is modest. At production scale (64+ nodes, 100K+ files,
larger filesystems) the effect is proportionally larger as MDS lock contention
for atime writes grows.

**What you lose:** `atime` is no longer recorded. No ML training pipeline
depends on when a file was last read; if you need access-time auditing, switch
to `relatime` (the kernel default, which updates atime at most once per day).

**Why not fstab?** The `/home` (OpenZFS/NFS) mount uses fstab because NFS
with `_netdev` reliably waits for network readiness. Lustre requires not just
network but the `lustre` kernel module and LNet initialization to be complete;
fstab + `_netdev` alone doesn't guarantee this ordering. Mounting explicitly
in cloud-init `runcmd` (which runs after network + module load) is the most
portable approach.

### 4.2 Lustre runtime performance tuning (recommended)

The mount options above handle metadata behaviour. For I/O throughput and
metadata IOPS, the following `lctl` runtime tunables are recommended for ML
training workloads. They are **not set by the templates by default** (to keep
the base minimal and avoid surprising users), but can be added to a
post-install script or run manually on the login/compute nodes.

These settings do not persist across reboot — add them to a boot script or
UserData if you want them permanent.

```bash
# --- Data path (OSC): controls per-OST throughput ---
# Max concurrent bulk-data RPCs per OST per client (default 8).
# Raise to saturate network bandwidth to each OST.
sudo lctl set_param osc.*.max_rpcs_in_flight=64

# Write-back cache per OST (default 32 MB).
# Larger buffer → fewer, larger writes for checkpoints.
sudo lctl set_param osc.*.max_dirty_mb=256

# Max pages per RPC (default 256 = 1 MB). 1024 = 4 MB per RPC.
# Reduces per-operation overhead for large sequential I/O.
sudo lctl set_param osc.*.max_pages_per_rpc=1024

# --- Metadata path (MDC): controls file open/stat/create throughput ---
# Max concurrent metadata RPCs (default 8).
# Critical for Python imports, HuggingFace cache walks, readdir.
sudo lctl set_param mdc.*.max_rpcs_in_flight=64

# Max concurrent modifying metadata RPCs (default 8).
# Helps checkpoint directory creation (many simultaneous file creates).
sudo lctl set_param mdc.*.max_mod_rpcs_in_flight=50

# --- Read-ahead (llite): controls sequential-read prefetch ---
# Total read-ahead buffer (default 64 MB). Scale with instance RAM.
# 256 MB is conservative; 1024 MB is fine on 2 TB GPU nodes.
sudo lctl set_param llite.*.max_read_ahead_mb=256

# Per-file read-ahead cap (default 64 MB).
sudo lctl set_param llite.*.max_read_ahead_per_file_mb=256

# --- Directory stat-ahead (llite): helps large-directory traversal ---
# Entries to stat ahead (default 32).
# 512 helps HuggingFace cache dirs with thousands of files.
sudo lctl set_param llite.*.statahead_max=512
```

**When to apply:**
- Large-scale training (16+ nodes, large datasets on `/fsx`)
- Checkpoint-heavy workloads (frequent writes of multi-GB files)
- Python environments on `/fsx` (many small-file imports)

**When to skip:**
- Small PoC / evaluation deploys (default 8 RPCs is sufficient for 1–4 nodes)
- If the filesystem is 1.2 TiB / 2 OSTs (the throughput ceiling is the
  filesystem's provisioned bandwidth, not the client's RPC parallelism)

### 4.3 Lustre stripe configuration (per-directory)

Stripe settings control how a file's data is distributed across OSTs. The
default FSx PFL (Progressive File Layout) is reasonable for mixed workloads:
```
-E 100M -c 1  -E 10G -c 8  -E 100G -c 16  -E -1 -c 32
```

For ML directories with a known access pattern, explicit striping is better:

```bash
# Checkpoints: max parallelism, large stripe for sequential writes
lfs setstripe -c -1 -S 16M /fsx/checkpoints

# Large datasets (sequential read by all nodes)
lfs setstripe -c -1 -S 16M /fsx/datasets

# Container images (.sqsh, 5-30 GB, sequential read)
lfs setstripe -c -1 -S 16M /fsx/containers

# Log directories (small append-only files — wide striping adds lock overhead)
lfs setstripe -c 1 /fsx/logs
```

| Parameter | Meaning | Default | ML recommendation |
|---|---|---|---|
| `-c` (stripe count) | Number of OSTs per file | PFL auto | `-1` (all OSTs) for large files |
| `-S` (stripe size) | Chunk size per OST before rotating | 1 MiB | `16M` for sequential I/O (matches EFA MTU alignment) |

**Do not stripe widely for small files** (<10 MB): one DLM lock per OST per
file means wide striping on thousands of small files adds more lock overhead
than it gains in parallelism. The default PFL handles this correctly (starts
with `-c 1` for small extents).

### 4.4 Kernel module parameters (advanced, requires reboot)

For instances with 64+ vCPUs (all P5/P6/hpc types), the Lustre kernel module
defaults bottleneck RPC processing. Set these before first mount via
`/etc/modprobe.d/lustre.conf`:

```bash
# /etc/modprobe.d/lustre-perf.conf
options ptlrpc ptlrpcd_per_cpt_max=32
options ksocklnd credits=2560
```

| Parameter | Default | Recommended | Effect |
|---|---|---|---|
| `ptlrpcd_per_cpt_max` | 2 | **32** | RPC worker threads per CPU partition. Without this, `max_rpcs_in_flight > 8` is bottlenecked by thread starvation. |
| `ksocklnd credits` | 256 | **2560** | LNet flow-control credits for TCP path. Increases in-flight message limit. |

**Note:** For EFA-enabled FSx (`FSxLustreEnableEfa=true`), LNet uses the EFA
fabric directly (not ksocklnd). The `ksocklnd credits` setting only matters
for the TCP fallback path or non-EFA filesystems. `ptlrpcd_per_cpt_max` is
relevant regardless of transport.

These require a **reboot** (or module reload) to take effect. For the
reference architecture, the recommended path is to add them to a custom AMI
(via `pcs-ready-dlami-with-enroot-pyxis.yaml`) or to early UserData before
the Lustre mount.

### 4.5 OS-level sysctl (optional)

On GPU instances with 2 TB RAM, the kernel dirty-page defaults can cause
bursty flushes that compete with GPU-to-host DMA during checkpoint writes:

```bash
# Lower dirty-page thresholds (default 20%/10% of RAM = 400 GB / 200 GB on 2 TB nodes)
sudo sysctl -w vm.dirty_ratio=5
sudo sysctl -w vm.dirty_background_ratio=3
sudo sysctl -w vm.dirty_expire_centisecs=1000

# Retain dentry/inode cache longer (helps repeated Python imports)
sudo sysctl -w vm.vfs_cache_pressure=50
```

For TCP-based LNet (non-EFA filesystems), also raise socket buffer limits:
```bash
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
```

---

## 5. P6-B300 NIC topology — single-template lock-in

`add-cng-p6-b300.yaml` is a hand-built `NetworkInterfaces` block for **exactly
`p6-b300.48xlarge`** (17 cards: card 0 ENA-only + EFA on cards 1-16 at `DeviceIndex 0`,
which differs from p5/b200's "card 0 = EFA, DeviceIndex 1" layout). The instance type
is locked via `AllowedValues` so a different type can't be selected through this
template — pick the matching `add-cng-*.yaml` for the family you want.

## 6. Known issues

### 6.1 First-srun-on-a-fresh-cpu1-node can drain the node ("Prolog error")

**Symptom.** The very first `srun` against a freshly-launched compute node fails with
`Job prolog failed`, the node enters `drained` state with `Reason=Prolog error`, and the
job is cancelled. The next `srun` (which lands on a sibling node) works.

**Cause.** A systemd / cgroup-v2 race during PCS bootstrap, observed on the PCS-ready
DLAMI Ubuntu 24.04. Sequence in `journalctl -u slurmd` on the affected node:

```
T+0   pcs_bootstrap_finalize.sh starts slurmd  (slurmd PID A)
T+1   slurmd PID A starts up, loads pyxis, registers with the controller
T+9   systemd: slurmd.service: Failed to set 'cpuset.cpus' attribute on
      '/system.slice/slurmd.service' to '': No space left on device
      (Linux's misleading error for an empty/invalid cpuset value)
T+9   systemd SIGTERMs slurmd PID A and starts slurmd PID B
T+10  user srun arrives, controller asks PID B for the prolog
T+10..+430  prolog notification times out (420s default)
T+430 slurmd cancels the job: "error: Waiting for JobId=N REQUEST_LAUNCH_PROLOG
      notification failed, giving up after 420 sec" → controller drains the node
```

The `cpuset.*` "No space left on device" message is a kernel-level error that
surfaces when systemd hands the cgroup controller an empty cpuset value; the ensuing
forced restart of slurmd is what actually breaks the prolog handshake. Nothing the PCS
templates or this repo's install scripts touch is in the path — `/etc/default/slurmd`
is written to by `install-enroot-pyxis.sh` but post-install completes before slurmd's
first start, and no slurmd unit / cgroup config is modified afterwards.

**Workaround.**
- Resubmit; subsequent `srun`s land on sibling nodes that didn't hit the race.
- To bring the drained node back, on the login node run:
  `sudo scontrol update nodename=cpu1-N state=resume reason=cleared`.
- For latency-sensitive workloads, warm the queue with a trivial `sinfo` /
  zero-work `srun` that you don't mind losing before the real job; that exercises the
  prolog path while a sibling is still cold.

**Why we don't paper over it.** The trigger lives below this repo's code (PCS
bootstrap + systemd + slurmd 25.05/25.11 cgroup-v2 handling). Wrapping the post-install
or the install script wouldn't change the timing of the systemd cgroup setup. The
cleanest mitigation is upstream — recording it here so users seeing it know the
workaround and the next contributor doesn't waste time looking for a bug in this PR's
scripts.

### 6.2 `needrestart` restarting `slurmd` would stop running jobs — already handled

When an unattended security upgrade updates a base library `slurmd` links (e.g. glibc),
`needrestart` restarts `slurmd`, and that restart tears down the `slurmstepd` steps under
it — **stopping every job on the node**. It is reproducible, not random, and not a reboot
or a Slurm-package upgrade.

The compute-node-group templates already guard against this: each `add-cng*` UserData
writes a `needrestart` drop-in so `slurmd` is never auto-restarted (security updates still
install; `needrestart` only defers the `slurmd` restart):

```perl
# /etc/needrestart/conf.d/90-pcs-slurm.conf  (written by add-cng* UserData)
$nrconf{override_rc} = { qr(^slurmd) => 0 };
```

`slurmd` is the only Slurm systemd service on these nodes (the controller is managed by
PCS); `qr(^slurmd)` also matches the versioned units (e.g. `slurmd-25.11`). The drop-in is
a standalone `conf.d/*.conf` naming only `slurmd`, so if a later DLAMI or Slurm unit
handles this differently it neither conflicts nor errors — at worst it becomes redundant.

## 7. Recommendations recap

For a new production deploy:

- `SlurmVersion=25.11` (full monitoring coverage)
- `MonitoringVersion=v2.10.2` (default; carries the B300 / `/opt` install / Docker 29.x fixes plus the ec2_sd credential-refresh fix)
- `AmiId` pinned to a resolved AMI ID (PCS-Ready DLAMI from SSM, or a custom AMI you
  built off it), not left empty — avoids drift on later scale-out
- For frequent scaling, pre-bake Enroot/Pyxis with `pcs-ready-dlami-with-enroot-pyxis.yaml`
  and pass its output as `AmiId` (~3 min boot vs ~6 min). Match `SlurmVersion` between
  the AMI build stack and the cluster stack.
- Default `DcgmExporterImage` covers H100/B200/B300; override only to pin a different build
- Minimum-CIDR `GrafanaAccessCidr` if used at all; otherwise empty (SSM port-forward)
- Throughput values that match the chosen FSx deployment types
