# Storage Tests (Test 10)

Validates FSx for Lustre + OpenZFS health, mount options, and performance
regression/improvement testing.

---

## Test 10: FSx storage health

Validates that both shared filesystems (Lustre on `/fsx`, OpenZFS on `/home`)
mount cleanly on every node, are usable, and that the FSx-side configuration
matches what the template asked for. Most of this is exercised implicitly by
Tests 1–9 (the post-install script, monitoring stack, OSU / FSDP all touch
`/fsx`); this test is the explicit health check to run after a fresh deploy
or after touching `ml-cluster-prerequisites.yaml` / FSx-related parameters.

### Step 1 — both filesystems mounted on every node

On the login node and at least one compute node (via `srun`):

```bash
mount | grep -E ' /home | /fsx '
df -h /home /fsx
```

Expected:
- `/fsx` mounted as type `lustre`, source `<fs-id>.fsx.<region>.amazonaws.com@tcp:/<mountname>`,
  size matches the `Capacity` parameter (1200 GiB default; 19200 GiB or larger
  when `FSxLustreEnableEfa=true`).
- `/home` mounted as type `nfs` over OpenZFS, NFS options include
  `nconnect=16,rsize=1048576,wsize=1048576` (the deploy-all UserData mount
  string).
- Both `df -h` reports show `Avail` greater than zero.

If a mount is missing on a freshly booted node, check
`/var/log/cloud-init-output.log` and `/var/log/pcs-post-install.log` for the
`mount` line. The most common first-boot failure is the OpenZFS DNS name not
being resolvable yet (NFS settle race); the post-install log will show
`mount.nfs: Failed to resolve server`.

### Step 2 — read/write sanity

```bash
# /fsx (Lustre): write 1 GiB, read it back
dd if=/dev/zero of=/fsx/.healthcheck bs=1M count=1024 conv=fsync 2>&1 | tail -1
dd if=/fsx/.healthcheck of=/dev/null bs=1M 2>&1 | tail -1
rm /fsx/.healthcheck

# /home (OpenZFS / NFS): same, 100 MiB (it's a small home filesystem)
dd if=/dev/zero of=/home/ubuntu/.healthcheck bs=1M count=100 conv=fsync 2>&1 | tail -1
dd if=/home/ubuntu/.healthcheck of=/dev/null bs=1M 2>&1 | tail -1
rm /home/ubuntu/.healthcheck
```

Expected: both write+read complete without error. Throughput is bounded by the
single-stream limits of NFS / Lustre on a single node — this is a sanity check,
not a benchmark. For Lustre throughput numbers, see the FSx Lustre User Guide
(provisioned throughput = `Capacity * PerUnitStorageThroughput / 1024` MB/s).

### Step 3 — FSx-side parameters match the CFN inputs

```bash
FSX_ID=$(aws cloudformation describe-stacks \
  --stack-name <your-stack> \
  --query 'Stacks[0].Outputs[?OutputKey==`FSxLustreFilesystemId`].OutputValue' \
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
- `EfaEnabled` = `False` (default; `True` when `FSxLustreEnableEfa=true`. EFA on
  FSx is a PERSISTENT_2-only feature — the prerequisites and deploy-all templates
  enforce this with a CFN Rule that fails the stack at create time when
  `FSxLustreEnableEfa=true` is combined with `LustreDeploymentType=PERSISTENT_1`)
- `MetadataConfiguration.Mode` = `AUTOMATIC` on PERSISTENT_2

For the OpenZFS `/home` filesystem:

```bash
FSXO_ID=$(aws cloudformation describe-stacks \
  --stack-name <your-stack> \
  --query 'Stacks[0].Outputs[?OutputKey==`FSxOFilesystemId`].OutputValue' \
  --region <region> --output text)

aws fsx describe-file-systems --file-system-ids "$FSXO_ID" --region <region> \
  --query 'FileSystems[0].[StorageCapacity,OpenZFSConfiguration.[DeploymentType,ThroughputCapacity]]' \
  --output text
```

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

For changes that touch mount options or FSx parameters, re-run the throughput
benchmark documented in
[OPERATIONS.md §4.1](../docs/OPERATIONS.md#41-lustre-mount-options--noatime)
(the `noatime` benchmark) and compare against the recorded baseline.
**A >10% degradation blocks the change.**

---
