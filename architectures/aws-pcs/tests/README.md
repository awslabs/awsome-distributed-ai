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
| 7, 7b | **Training** | FSDP Llama-2 7B (HF-streamed) + Megatron-LM GPT-3 (TP/PP/DP, local data) | [`training-test.md`](./training-test.md) | GPU PRs |
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
(`S3BucketName`) and resolve the PCS-Ready DLAMI from SSM per region.

Columns: **Deploy** (deploy-all → CREATE_COMPLETE), **Mon** (6 monitoring
containers up on the login node), **Storage** (FSx Lustre `/fsx` + OpenZFS
`/home` created & mounted — with the OpenZFS deployment type that worked),
**Pyxis** (Enroot/Pyxis container job runs), **GPU** (multi-NIC GPU CNG tested,
incl. capacity reservation type), **Verified** (date, UTC).

| Region | Deploy | Mon | Storage (OpenZFS type) | Pyxis | GPU (CB/ODCR) | Verified |
|---|---|---|---|---|---|---|
| **us-east-1** (N. Virginia) | ✅ | ✅ | ✅ `SINGLE_AZ_HA_2` | ✅ | — | 2026-06-17 |
| **us-east-2** (Ohio) | ✅ | ✅ | ✅ `SINGLE_AZ_HA_2` | ✅ | EFA hpc6a(1)/hpc8a(2) — CPU/EFA | 2026-06-17 |
| **us-west-2** (Oregon) | ✅ | ✅ | ✅ `SINGLE_AZ_HA_2` | ✅ | — | 2026-06-17 |
| **ap-northeast-1** (Tokyo) | ✅ | ✅ | ✅ `SINGLE_AZ_HA_2` | ✅ | — | 2026-06-17 |
| **ap-south-1** (Mumbai) | ✅ | ✅ | ✅ `SINGLE_AZ_HA_2` | ✅ | ✅ **p6-b200 ×4** (Capacity Block) | 2026-06-17 |
| **ap-northeast-3** (Osaka) | ✅ | ✅ | ✅ `SINGLE_AZ_1` ¹ | ✅ | — | 2026-06-17 |
| **ap-southeast-1** (Singapore) | ✅ | ✅ | ✅ `SINGLE_AZ_HA_2` | ✅ | — | 2026-06-17 |
| **ap-southeast-2** (Sydney) | ✅ | ✅ | ✅ `SINGLE_AZ_HA_2` | ✅ | — | 2026-06-17 |
| **eu-central-1** (Frankfurt) | ✅ | ✅ | ✅ `SINGLE_AZ_HA_2` | ✅ | — | 2026-06-17 |
| **eu-north-1** (Stockholm) | ✅ | ✅ | ✅ `SINGLE_AZ_HA_2` | ✅ | — | 2026-06-17 |

¹ **Osaka (ap-northeast-3) does not support the default `OpenZFSDeploymentType=SINGLE_AZ_HA_2`**
— the default deploy fails at the OpenZFS filesystem with
`Invalid deploymentType (BadRequest)`. Deploy there with
`OpenZFSDeploymentType=SINGLE_AZ_1` (+ a valid `HomeThroughput`, e.g. 256). This
is the documented "not available in every Region" case (see the parameter's
description in `ml-cluster-prerequisites.yaml`); `SINGLE_AZ_1` is available in
all regions and is the safe fallback.

### Remaining PCS launch regions (not yet run)

AWS PCS is available in 18 regions total (per
`/aws/service/global-infrastructure/services/pcs/regions`). Beyond the 10 tested
above, these are expected to work but have not been run — confirm
`LustreDeploymentType` / `OpenZFSDeploymentType` availability before relying on
the defaults (`PERSISTENT_2` / `SINGLE_AZ_HA_2`), as Osaka above shows:
`eu-south-1` (Milan), `eu-south-2` (Spain), `eu-west-1` (Ireland), `eu-west-2`
(London), `eu-west-3` (Paris), `sa-east-1` (São Paulo), `us-gov-east-1`,
`us-gov-west-1` (GovCloud).

The ap-south-1 (Mumbai) row was exercised most deeply — p6-b200 ×4 GPU. The
detailed results live with their tests, not here: NCCL all_reduce numbers in
[compute-test.md](./compute-test.md#test-6-nccl-multi-node-efa), distributed
training (and the HuggingFace rate-limit caveat) in
[training-test.md](./training-test.md#test-7-fsdp-sample-training), GPU health
check in [gpu-healthcheck-test.md](./gpu-healthcheck-test.md).

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
| Megatron-LM GPT-3 (TP/PP/DP) | [`3.test_cases/megatron/megatron-lm`](../../../3.test_cases/megatron/megatron-lm) | Import `.sqsh` to `/fsx`, data under `/fsx/gpt2/`, 4 nodes |
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
