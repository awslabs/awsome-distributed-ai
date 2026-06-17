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
| 1-3, 8 | **Infrastructure** | Monitoring stack, container runtime (first-boot + AMI build), template lint | [`infra-test.md`](./infra-test.md) | Every PR |
| 4-6 | **Compute** | CPU queue, GPU families (G/P5/P6), NCCL multi-node EFA | [`compute-test.md`](./compute-test.md) | Every PR |
| 7 | **Training** | FSDP Llama-2 7B distributed training | [`training-test.md`](./training-test.md) | GPU PRs |
| 9 | **HPC EFA** | EFA on CPU instances (hpc6a/hpc7a/hpc8a), OSU benchmarks | [`hpc-efa-test.md`](./hpc-efa-test.md) | EFA wiring changes |
| 10 | **Storage** | FSx health check + performance regression test (noatime benchmark) | [`storage-test.md`](./storage-test.md) | FSx / mount changes |
| 11-12 | **Multi-user** | OpenLDAP directory + Slurm managed accounting | [`multi-user-test.md`](./multi-user-test.md) | Directory / accounting changes |
| 13 | **GPU health** | GPU Cluster Health Check suite (DCGM, EFA, NVLink, NCCL thresholds) | [`gpu-healthcheck-test.md`](./gpu-healthcheck-test.md) | GPU CNG deploys |

---

## Quick-start: single-cluster shortcut

A single `pcs-ml-cluster-deploy-all.yaml` deploy with `MonitoringStack=Prometheus-LoginNode`,
`DeployOnDemandCNG=true`, and `DeployPseriesCNG=true` exercises Tests 1–7 in
one cluster. Tests 8-13 are separate paths run only when their inputs change.

---

## Major-update PR — configurations run end-to-end on real hardware

The major-update PR (IAM policies, multi-user OpenLDAP, `MonitoringStack` rename,
`SSHAccessCidr`/`GrafanaAccessCidr`, multi-AZ subnets, VPCName auto-derive +
EFA auto-count) was validated end-to-end in us-east-2 with a single
`deploy-all` cluster:

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
