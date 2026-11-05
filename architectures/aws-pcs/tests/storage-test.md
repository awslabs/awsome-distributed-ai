# Storage Tests (Test 10)

Validates shared-storage health (`/home` on FSx for OpenZFS or Amazon EFS, optional
FSx for Lustre `/fsx`), mount options, and performance regression/improvement testing.

---

## Test 10: Shared storage health

Validates that the shared filesystems (`/home` on OpenZFS or EFS, Lustre on
`/fsx` when `DeployFSxLustre=true`) mount cleanly on every node, are usable, and
that the filesystem-side configuration matches what the template asked for. Most of this is exercised implicitly by
Tests 1–9 (the Enroot/Pyxis install, monitoring stack, OSU / FSDP all touch
`/fsx`); this test is the explicit health check to run after a fresh deploy
or after touching `ml-cluster-prerequisites.yaml` / storage parameters. Run it once
per storage combination you change: the default (OpenZFS + Lustre),
`HomeStorageType=EFS`, and `DeployFSxLustre=false`.

### Step 1 — filesystems mounted on every node

On the login node and at least one compute node (via `srun`):

```bash
mount | grep -E ' /home | /fsx '
df -h /home
mountpoint /fsx && df -h /fsx
```

On a compute node, run the same commands through Slurm:

```bash
srun -p cpu1 -N1 bash -c "mount | grep -E ' /home | /fsx '; df -h /home; mountpoint /fsx && df -h /fsx"
```

Expected:

- `/fsx` mounted as type `lustre`, source `<ip>@tcp:/<mountname>` (`mount` shows the
  filesystem's DNS name resolved to an IP),
  size matches the `Capacity` parameter (1200 GiB default; 19200 GiB or larger
  when `FSxLustreEnableEfa=true`).
- `/home` mounted as type `nfs` over OpenZFS, NFS options include
  `nconnect=16,rsize=1048576,wsize=1048576` (the `mount-openzfs-home.sh`
  lifecycle-action mount string).
- With `HomeStorageType=EFS`: `/home` mounted as type `nfs4`, source
  `<fs-id>.efs.<region>.amazonaws.com:/`, options include `vers=4.1`,
  `rsize=1048576,wsize=1048576`, `hard`, `noresvport` (the `mount-efs-home.sh`
  mount string); `df -h` shows EFS's nominal 8.0E size.
- With `DeployFSxLustre=false`: no `/fsx` line, and `mountpoint` prints
  `mountpoint: /fsx: No such file or directory` (or `/fsx is not a mountpoint` on an
  AMI that ships an empty `/fsx` directory).
- Every `df -h` report shows `Avail` greater than zero.

If a mount is missing on a new node's first boot, check the per-script
lifecycle logs `/var/log/amazon/pcs/lifecycle/actions/nodeBootstrapped/mount-openzfs-home.log`
(or `mount-efs-home.log`) and `mount-lustre-fsx.log` (root-readable). A first-boot mount failure TERMINATES
and replaces the node, so also check the CNG's recent instance churn. The most
common first-boot failure is the OpenZFS or EFS DNS name not being resolvable yet
(NFS settle race); the mount log will show `mount.nfs: Failed to resolve server`
(`mount.nfs4` for EFS).

On subsequent boots, `/home` (OpenZFS or EFS) is restored from `/etc/fstab` by systemd, not by
the lifecycle action. Replaying the NFSv3 mount during early cloud-init can
wait for rpc-statd, which waits for network-online.target, which in turn waits
for cloud-init. `slurmd.service` therefore requires and starts after
`home.mount`; a failed or missing mount blocks Slurm instead of allowing jobs
on the local `/home`. This is not a lifecycle `TERMINATE` on later boots.
Inspect the journal for `home.mount` and `slurmd.service` for reboot failures.
Lustre keeps its existing `EVERY_BOOT` / `TERMINATE` policy.

Roll out the updated templates and script together and replace existing nodes
under a separately approved rollout; `CACHE_ONCE` and `FIRST_BOOT_ONLY` do not
retrofit the slurmd drop-in onto existing nodes. Filesystem ID changes also
require node replacement. Before declaring reboot recovery verified, validate
first boot and a normal Slurm-requested reboot on replacement nodes, then
confirm `/home` is the expected NFS export before Slurm becomes job-eligible.
The offline regression in `test_home_boot.py` is not this runtime validation.

### Step 2 — read/write sanity

On the login node, as `ubuntu`:

```bash
# /fsx (Lustre, when deployed): write 1 GiB, read it back
if mountpoint -q /fsx; then
  dd if=/dev/zero of=/fsx/.healthcheck bs=1M count=1024 conv=fsync 2>&1 | tail -1
  dd if=/fsx/.healthcheck of=/dev/null bs=1M 2>&1 | tail -1
  rm /fsx/.healthcheck
fi

# /home (OpenZFS or EFS, NFS): 100 MiB
dd if=/dev/zero of=/home/ubuntu/.healthcheck bs=1M count=100 conv=fsync 2>&1 | tail -1
dd if=/home/ubuntu/.healthcheck of=/dev/null bs=1M 2>&1 | tail -1
rm /home/ubuntu/.healthcheck
```

Expected: every write+read pair that runs (`/home` always, `/fsx` when mounted)
completes without error. Throughput is bounded by the
single-stream limits of NFS / Lustre on a single node — this is a sanity check,
not a benchmark. For Lustre throughput numbers, see the FSx Lustre User Guide
(provisioned throughput = `Capacity * PerUnitStorageThroughput / 1024` MB/s).

### Step 3 — filesystem-side parameters match the CFN inputs

Run from a workstation with the AWS CLI; `<your-stack>` is the deploy-all stack.
For FSx for Lustre (`DeployFSxLustre=true`):

```bash
FSX_ID=$(aws cloudformation describe-stacks \
  --stack-name <your-stack> \
  --query 'Stacks[0].Outputs[?OutputKey==`FSxLustreId`].OutputValue' \
  --region <region> --output text)

aws fsx describe-file-systems --file-system-ids "$FSX_ID" --region <region> \
  --query 'FileSystems[0].[StorageCapacity,StorageType,LustreConfiguration.[DeploymentType,PerUnitStorageThroughput,DataCompressionType,EfaEnabled,MetadataConfiguration.Mode]]' \
  --output text
```

Expected (default deploy):

- `StorageCapacity` = your `Capacity` parameter (1200 by default)
- `StorageType` = `SSD`
- `DeploymentType` = `PERSISTENT_2` (default; or `PERSISTENT_1` if you set it)
- `PerUnitStorageThroughput` = 250 (default)
- `DataCompressionType` = `LZ4` (default)
- `EfaEnabled` = `None` (default — the property is unset; `True` when `FSxLustreEnableEfa=true`. EFA on
  FSx is a PERSISTENT_2-only feature — the prerequisites and deploy-all templates
  enforce this with a CFN Rule that fails the stack at create time when
  `FSxLustreEnableEfa=true` is combined with `LustreDeploymentType=PERSISTENT_1`)
- `MetadataConfiguration.Mode` = `AUTOMATIC` on PERSISTENT_2

For the OpenZFS `/home` filesystem (`HomeStorageType=FSxOpenZFS`):

```bash
FSXO_ID=$(aws cloudformation describe-stacks \
  --stack-name <your-stack> \
  --query 'Stacks[0].Outputs[?OutputKey==`FSxOpenZFSId`].OutputValue' \
  --region <region> --output text)

aws fsx describe-file-systems --file-system-ids "$FSXO_ID" --region <region> \
  --query 'FileSystems[0].[StorageCapacity,OpenZFSConfiguration.[DeploymentType,ThroughputCapacity]]' \
  --output text
```

Expected (default deploy): `512`, then `SINGLE_AZ_HA_2  320`.

For the EFS `/home` filesystem (`HomeStorageType=EFS`):

```bash
EFS_ID=$(aws cloudformation describe-stacks \
  --stack-name <your-stack> \
  --query 'Stacks[0].Outputs[?OutputKey==`EFSHomeId`].OutputValue' \
  --region <region> --output text)

aws efs describe-file-systems --file-system-id "$EFS_ID" --region <region> \
  --query 'FileSystems[0].[ThroughputMode,PerformanceMode,Encrypted,NumberOfMountTargets]' \
  --output text
aws efs describe-backup-policy --file-system-id "$EFS_ID" --region <region> \
  --query 'BackupPolicy.Status' --output text
```

Expected: `elastic  generalPurpose  True  1` (one mount target per private subnet —
`2`/`3` with `AdditionalSubnetAZ2`/`AdditionalSubnetAZ3`), then `ENABLED`.

### Step 4 — Storage dashboard in Grafana

The `Compute Node Details` and `HPC Cluster Monitoring → Storage` Grafana
dashboards are populated by the **CloudWatch Exporter** (FSx CloudWatch
metrics, scraped by the monitoring stack on the login node — see the
[aws-parallelcluster-monitoring v2.6 release notes](https://github.com/aws-samples/aws-parallelcluster-monitoring/releases/tag/v2.6)).

Open Grafana (Test 1's port-forward / public CIDR), go to the Storage
dashboard, and verify the `/fsx` panels (Throughput, IOPS, Free Capacity,
Client Connections) populate within ~5 minutes of a workload starting. CW
metrics have a ~5 min publishing delay, so a brand-new filesystem with no I/O
shows blank panels for a while; the `dd` from Step 2 is enough to seed values.

With `DeployFSxLustre=false` the `/fsx` panels stay empty. With `HomeStorageType=EFS`,
`/home` appears in the dashboard's EFS panels instead (the CloudWatch Exporter also
collects the `AWS/EFS` metrics).

### `FSxLustreEnableEfa=true` specifics

When `FSxLustreEnableEfa=true` is set on a PERSISTENT_2 SSD filesystem, the
extra checks beyond the above are:

- `aws fsx describe-file-systems` `LustreConfiguration.EfaEnabled = true`
- `Capacity` is at-or-above the EFA minimum for the chosen
  `PerUnitStorageThroughput` tier (19200 GiB for tier 250; the FSx for
  Lustre User Guide has the full matrix). Below the minimum, the FSx side
  rejects the `CreateFileSystem` with `Invalid storage capacity provided:
  N GiB. Minimum storage capacity for an EFA enabled LUSTRE file systems
  with deployment type PERSISTENT_2, per unit storage throughput X and
  storage type SSD is M`. The Lustre nested stack fails first, then the
  whole stack rolls back; that's the expected behavior for an undersized
  Capacity.

The FSx-side EFA endpoints are usable from EFA-capable clients (CPU CNGs
deployed with `OnDemandEfaInterfaceCount > 0`, P5/P6 GPU CNGs). Plain Lustre client
mounts continue to work over TCP for non-EFA nodes; EFA support is additive.

### Performance regression criteria

Lustre only — not applicable when `DeployFSxLustre=false`.

For changes that touch mount options or FSx parameters, re-run the throughput
benchmark documented in
[OPERATIONS.md §4.1](../docs/OPERATIONS.md#41-lustre-mount-options--noatime)
(the `noatime` benchmark) and compare against the recorded baseline.
**A >10% degradation blocks the change.**

---
