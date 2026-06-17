# AWS PCS — Test & Validation Guide

Test procedures for validating an AWS PCS cluster deployed from the templates
in [`../assets`](../assets).

For operational guidance (Slurm version trade-offs, AMI pinning, monitoring,
FSx tuning), see [`../docs/OPERATIONS.md`](../docs/OPERATIONS.md).

---

## Pre-merge test matrix

Run this **complete set** before merging template or script changes.

| # | Category | Tests | File | When to run |
|---|---|---|---|---|
| 0 | **Docs lint** | `bash tests/lint-docs.sh` — no stale/renamed param refs in docs, every deploy-all param documented, README anchors resolve (the docs counterpart of template-lint; runs in seconds, no AWS) | [`lint-docs.sh`](./lint-docs.sh) | Every PR (esp. param renames) |
| 1-3, 8 | **Infrastructure** | Monitoring stack, container runtime (first-boot + AMI build), template lint | [`infra-test.md`](./infra-test.md) | Every PR |
| 4-6 | **Compute** | CPU queue, GPU families (G/P5/P6), NCCL multi-node EFA | [`compute-test.md`](./compute-test.md) | Every PR |
| 7 | **Training** | FSDP Llama-2 7B distributed training | [`training-test.md`](./training-test.md) | GPU PRs |
| 9 | **HPC EFA** | EFA on CPU instances (hpc6a/hpc7a/hpc8a), OSU benchmarks | [`hpc-efa-test.md`](./hpc-efa-test.md) | EFA wiring changes |
| 10 | **Storage** | FSx health check + performance regression test (noatime benchmark) | [`storage-test.md`](./storage-test.md) | FSx / mount changes |
| 11-12 | **Multi-user** | OpenLDAP directory + Slurm managed accounting | [`multi-user-test.md`](./multi-user-test.md) | Directory / accounting changes |
| 13 | **GPU health** | GPU Cluster Health Check suite (DCGM, EFA, NVLink, NCCL thresholds) | [`gpu-healthcheck-test.md`](./gpu-healthcheck-test.md) | GPU CNG deploys |

> **GPU health check verified on B200** (ap-south-1 p6-b200): lightweight suite
> **4/4 PASS** — nvidia-smi (8 GPUs, no Xid/ECC), DCGM L2 diagnostics, EFA
> enumeration (8 PCI / 10 RDMA devices), topology (8 GPUs, connectivity validated;
> a B200-specific "unsupported P2P path" warning is non-blocking).

---

## Quick-start: single-cluster shortcut

A single `pcs-ml-cluster-deploy-all.yaml` deploy with `MonitoringStack=Prometheus-LoginNode`,
`DeployOnDemandCNG=true`, and `DeployPseriesCNG=true` exercises Tests 1–7 in
one cluster. Tests 8-13 are separate paths run only when their inputs change.

---

## Major-update PR — configurations run end-to-end on real hardware

The major-update PR (IAM policies, multi-user OpenLDAP, `MonitoringStack` rename,
`SSHAccessCidr`/`GrafanaAccessCidr`, multi-AZ subnets, `OnDemandEfaInterfaceCount`
0/1/2 collapse, VPCName fixed to `${StackName}-VPC`, instance-role perms split
inline) was validated end-to-end in us-east-2 with a single `deploy-all` cluster:

```
PrimarySubnetAZ=us-east-2b  AdditionalSubnetAZ2=us-east-2a  AdditionalSubnetAZ3=us-east-2c
DirectoryService=OpenLDAP-LoginNode  SSHAccessCidr=<office-ip>/32
MonitoringStack=Prometheus-LoginNode
```

| Feature | Verified | Result |
|---|---|---|
| **Multi-AZ subnets** | 3 private subnets, 3 distinct AZs, non-overlapping `/18` CIDRs (`10.1.0.0/18`, `10.1.64.0/18`, `10.1.128.0/18`); compute nodes actually scheduled into the additional-AZ subnets (`10.1.x`) | ✅ |
| **SSHAccessCidr** | `LoginAccessSecurityGroup` opens **port 22 only** (443 absent since `GrafanaAccessCidr` empty) from the given CIDR, attached to the login node only; direct `ssh` to the login public IP succeeds | ✅ |
| **MonitoringStack=Prometheus-LoginNode** | 6 login containers up (prometheus/grafana/nginx/cloudwatch-exporter/node-exporter/pushgateway) | ✅ |
| **OpenLDAP server (boot-time, automatic)** | `slapd` active, DB on `/home/ldap-db`, OUs + `clusterusers` gid 3000 created, admin password in SSM; `ldap-add-user` helper auto-installed to `/usr/local/bin`; apt/dpkg-lock wait survived first-boot unattended-upgrades | ✅ |
| **`directory-role` tag discovery** | login tagged `directory-role=server` (independent of `monitoring-role`); compute clients discover the server by that tag, scoped to `pcs-cluster-id` | ✅ |
| **Compute SSSD client (boot-time, automatic)** | fresh compute nodes auto-install SSSD + sssd-tools, resolve `testuser1` via the login LDAP — no manual step | ✅ |
| **Slurm job as LDAP user** | `srun` as `testuser1` runs with `uid=10001` on the compute node | ✅ |
| **Multi-node UID consistency** | two different compute nodes both resolve `testuser1`→`uid=10001` | ✅ |
| **User delete propagation** | `ldapdelete` + `sss_cache -E` removes the user from `getent` immediately (sssd-tools now installed) | ✅ |
| **Template lint** | `validate-template` passes on all 8 edited `assets/*.yaml` (incl. 3 GPU templates + 2 IAM stacks) | ✅ |
| **`OnDemandEfaInterfaceCount` (0/1/2 collapse)** | `=1` → hpc6a launch template has 1 EFA NIC; `=2` → hpc8a has 2 EFA NICs (`DeviceIndex 0/1`, `InterfaceType: efa`). Real EFA traffic on hpc8a (2 nodes, `FI_PROVIDER=efa`): `osu_bw` peak **26.3 GB/s** (~210 Gbps, 1 pair); `osu_mbw_mr` peak **42.9 GB/s** (~343 Gbps, 16 pairs/node multi-rail) | ✅ |
| **Slurm managed accounting + multi-user (Test 12)** | on a `ManagedAccounting=enabled` cluster: LDAP users alice/bob registered in `sacctmgr` (account=ml-team) as root admin; jobs submitted as each LDAP user complete and `sacct -a` records them under the correct `User`+`Account` (`scontrol show job`: `UserId=alice(10001) Account=ml-team`) | ✅ |

---

## Region coverage

The templates fetch nested stacks + boot scripts from a single S3 bucket
(`S3BucketName`) and resolve the PCS-Ready DLAMI from SSM per region. Validated
deploys (deploy-all: prereqs + cluster + login + CPU, monitoring on) across:

| Region | Deploy | Login + monitoring | FSx Lustre `/fsx` + OpenZFS `/home` | Notes |
|---|---|---|---|---|
| **us-east-1** | ✅ CREATE_COMPLETE | ✅ 6 containers | ✅ 1.2T Lustre + OpenZFS NFS | template bucket's home region |
| **us-east-2** | ✅ CREATE_COMPLETE | ✅ 6 containers | ✅ | primary test region; also EFA hpc6a/hpc8a + multi-user/accounting |
| **us-west-2** | ✅ CREATE_COMPLETE | ✅ 6 containers | ✅ | |
| **ap-northeast-1** | ✅ CREATE_COMPLETE | ✅ | ✅ | Tokyo |
| **ap-south-1** | ✅ CREATE_COMPLETE | ✅ 6 containers | ✅ | Mumbai; p6-b200 ×4 GPU (8 GPU/node, 8 EFA NICs), B200 driver 595.71.05 |

### All PCS launch regions

AWS PCS is available in the 18 regions below (per
`/aws/service/global-infrastructure/services/pcs/regions`). The 5 marked
**tested** above were validated end-to-end; the rest are expected to work (the
templates resolve the PCS-Ready DLAMI from SSM and FSx deployment types per
region) but have not been run. Confirm `LustreDeploymentType` /
`OpenZFSDeploymentType` availability in a new region before relying on the
defaults (`PERSISTENT_2` / `SINGLE_AZ_HA_2`).

| Region | Status |
|---|---|
| us-east-1 (N. Virginia) | ✅ tested |
| us-east-2 (Ohio) | ✅ tested |
| us-west-2 (Oregon) | ✅ tested |
| ap-northeast-1 (Tokyo) | ✅ tested |
| ap-south-1 (Mumbai) | ✅ tested (GPU/B200) |
| ap-northeast-3 (Osaka) | ⬜ not run |
| ap-southeast-1 (Singapore) | ⬜ not run |
| ap-southeast-2 (Sydney) | ⬜ not run |
| eu-central-1 (Frankfurt) | ⬜ not run |
| eu-north-1 (Stockholm) | ⬜ not run |
| eu-south-1 (Milan) | ⬜ not run |
| eu-south-2 (Spain) | ⬜ not run |
| eu-west-1 (Ireland) | ⬜ not run |
| eu-west-2 (London) | ⬜ not run |
| eu-west-3 (Paris) | ⬜ not run |
| sa-east-1 (São Paulo) | ⬜ not run |
| us-gov-east-1 (GovCloud East) | ⬜ not run |
| us-gov-west-1 (GovCloud West) | ⬜ not run |

### p6-b200 ×4 NCCL all_reduce (ap-south-1, 32 GPU)

NCCL `all_reduce_perf` over EFA (Pyxis container `public.ecr.aws/hpc-cloud/nccl-tests`,
4× p6-b200.48xlarge = 32 B200 GPUs, NCCL auto-selected NICs):

- **NET/OFI provider: `efa`, fabric `efa-direct`, 8 NICs per node** (all EFA NICs in use)
- **Peak bus bandwidth: 377.4 GB/s** (16 GiB message, in/out-of-place)
- Avg bus bandwidth across the 8 B → 16 GiB sweep: 106 GB/s; 0 wrong / 0 out-of-bounds

| Message size | busbw (GB/s) |
|---|---|
| 128 MiB | 295 |
| 256 MiB | 325 |
| 1 GiB | 367 |
| 4 GiB | 375 |
| 16 GiB | **377** |

The curve is healthy (monotonic, latency-bound small → bandwidth-bound large,
asymptotes at 377). With p6-b200's 8×EFA (≈800 Gbps/node node-to-node) this is a
reasonable effective all_reduce busbw — EFA's `efa-direct` fabric is engaged on
all 8 NICs. It is **not a peak-tuned number**: NICs were auto-selected (no
`NCCL_ALGO`/topology-aware tuning) and the test ran from a general-purpose
nccl-tests image. The goal here was to confirm the templates establish
multi-node GPU communication over EFA, not to chase the hardware ceiling; for a
peak run use a B200-tuned NCCL build + the `topology-aware-nccl-tests` sbatch.

> Pre-merge gotcha hit here: the GPU nodes booted without Enroot/Pyxis (the
> `PostInstallScriptUrl` 404, see caveat below). After installing
> `install-enroot-pyxis.sh` manually on the **login node and all compute nodes**
> and `systemctl restart slurmd` on the compute nodes (so the freshly-dropped
> `spank_pyxis.so` loads), `--container-image` jobs ran. A correct
> `PostInstallScriptUrl` at deploy time avoids all of this.

### FSDP distributed training (ap-south-1, p6-b200)

Ran the repo's canonical `3.test_cases/pytorch/FSDP/slurm/llama3_2_1b-training.sbatch`
unchanged (venv per `create_venv.sh`, `srun … torchrun --nproc_per_node=8`,
`--dataset=allenai/c4`).

- **The distributed-training stack the architecture is responsible for works.**
  Across 4× p6-b200 (32 GPUs): NCCL c10d rendezvous, FSDP2 wrap of the Llama-3
  1B model (`1,154,549,760` params), and optimizer creation complete on every
  rank. Each node exposes all 8 B200 GPUs to its `torchrun` (`nvidia-smi -L` = 8/node).
- **The training step loop could not be reached because of an external
  HuggingFace rate-limit on `allenai/c4`**, not a cluster/template issue. Every
  rank's dataloader does `load_dataset("allenai/c4", streaming=True)`, which
  hits the c4 *tree* API (1024-shard listing); HF returns `429 Too Many Requests`.
  Confirmed it is purely external by exhausting the mitigations:
  - HF token (authenticated) — still 429
  - shared `HF_HOME=/fsx` cache + single-process prefetch of shards — still 429
    (streaming re-lists shards via the tree API regardless of cache)
  - `HF_HUB_OFFLINE=1` — flips to `OfflineModeIsEnabled` (streaming needs the Hub)
  - long `HF_HUB_ETAG_TIMEOUT`/`DOWNLOAD_TIMEOUT` + the loader's 20× retry — still 429
  - dropping to a single node (8 concurrent readers instead of 32) — still 429
- **Conclusion:** the cluster is training-ready (NCCL/EFA/FSDP all proven; NCCL
  all_reduce hit 377 GB/s above). Driving `allenai/c4` through the streaming
  loader at scale needs an HF account with a higher rate limit, an HF mirror, or
  pre-tokenized data staged on `/fsx` (non-streaming) — an operational/data
  choice, independent of these templates.

Cross-region note: nested-stack `TemplateURL` and the in-instance
`aws s3 cp` of boot scripts both work against an S3 bucket in a **different**
region (S3 global namespace; no `--region` needed) — verified ap-south-1 →
us-east-1 bucket. The PCS-Ready DLAMI SSM parameter
(`/aws/service/pcs/ami/dlami-base-ubuntu2404/x86_64/latest/ami-id`) resolves in
every region tested. `OpenZFSDeploymentType=SINGLE_AZ_HA_2` and
`LustreDeploymentType=PERSISTENT_2` (the defaults) were available in all five.

> **Pre-merge caveat:** `PostInstallScriptUrl` defaults to the `awslabs/main`
> GitHub-raw URL, which 404s until PR #1120 merges — so pre-merge test deploys
> must override it (point at the test bucket / fork) or compute nodes boot
> without Enroot/Pyxis (`/var/log/pcs-post-install.log` shows exit 127). The
> cluster still reaches CREATE_COMPLETE; only the container runtime is missing.

---

## Canonical assets reused (not duplicated)

| Workload | Source in this repo | PCS-specific delta |
|---|---|---|
| NCCL `all_reduce` | [`micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch`](../../../micro-benchmarks/nccl-tests) | Partition name, Enroot import on login node |
| FSDP Llama-2 7B | [`3.test_cases/pytorch/FSDP`](../../../3.test_cases/pytorch/FSDP) | Cache on `/fsx`, 2 nodes |
| GPU Health Check | [`4.validation_and_observability/2.gpu-cluster-healthcheck`](../../../4.validation_and_observability/2.gpu-cluster-healthcheck) | sbatch wrapper, partition name |

---

## Notes

- All Slurm commands run as `ubuntu` from the login node (SSM or SSH)
- Slurm binaries: `export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH`
- Template lint: `aws cloudformation validate-template --template-body file://assets/<name>.yaml`
- Regression criteria (storage): >10% degradation blocks the change

---

## Cleanup

Delete the stack from the CloudFormation console (select → **Delete**) or:

```bash
aws cloudformation delete-stack --stack-name <stack-name>
aws cloudformation wait stack-delete-complete --stack-name <stack-name>
```

Nested stacks (and FSx) are deleted automatically — back up FSx data first.

If DELETE_FAILED on CNG stacks (PCS timing dependency), delete PCS CNGs first:
```bash
CLUSTER_ID=<id>
for cng in $(aws pcs list-compute-node-groups --cluster-identifier $CLUSTER_ID --query 'computeNodeGroups[].id' --output text); do
  aws pcs delete-compute-node-group --cluster-identifier $CLUSTER_ID --compute-node-group-identifier $cng
done
sleep 60
aws cloudformation delete-stack --stack-name <stack-name>
```
