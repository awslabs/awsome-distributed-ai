# AWS PCS — Test & Validation Guide

Test procedures for validating an AWS PCS cluster deployed from the templates
in [`../assets`](../assets).

For operational guidance (Slurm version trade-offs, AMI pinning, monitoring,
FSx tuning), see [`../docs/OPERATIONS.md`](../docs/OPERATIONS.md).

---

## Pre-merge test matrix

Run this **complete set** before merging template or script changes.

Unless a test says otherwise, commands run from the login node (SSM or SSH) as
`ubuntu`, the PCS-Ready DLAMI default user (the multi-user tests submit as
LDAP users).

| # | Category | Tests | File | When to run |
|---|---|---|---|---|
| 0 | **Docs lint** | `bash tests/lint-docs.sh` — no stale/renamed param refs in docs, every deploy-all param documented, README anchors resolve (the docs counterpart of template-lint; runs in seconds, no AWS) | [`lint-docs.sh`](./lint-docs.sh) | Every PR (esp. param renames) |
| 1-3, 8 | **Infrastructure** | Template lint, monitoring stack, container runtime (first-boot + AMI build) | [`infra-test.md`](./infra-test.md) | Every PR |
| 4-6 | **Compute** | CPU queue, GPU families (G/P5/P6), NCCL multi-node EFA | [`compute-test.md`](./compute-test.md) | Every PR |
| 7, 7b | **Training** | FSDP Llama-2 7B (HF-streamed) + Megatron-LM GPT-3 (TP/PP/DP, local data) | [`training-test.md`](./training-test.md) | GPU PRs |
| 9 | **HPC EFA** | EFA on CPU instances (hpc6a/hpc7a/hpc8a), OSU benchmarks | [`hpc-efa-test.md`](./hpc-efa-test.md) | EFA wiring changes |
| 10 | **Storage** | FSx health check + performance regression test (noatime benchmark) | [`storage-test.md`](./storage-test.md) | FSx / mount changes |
| 11-12 | **Multi-user** | OpenLDAP directory + Slurm managed accounting | [`multi-user-test.md`](./multi-user-test.md) | Directory / accounting changes |
| 13 | **GPU health** | GPU Cluster Health Check suite (DCGM, EFA, NVLink, NCCL thresholds) | [`gpu-healthcheck-test.md`](./gpu-healthcheck-test.md) | GPU CNG deploys |
| 14 | **IAM** | cluster-admin deploys+deletes (no `iam:CreatePolicy`); cluster-user is SSM-login-only and can't read the LDAP password | [`iam-test.md`](./iam-test.md) | IAM policy changes |

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

The table lists **all 20 regions where AWS PCS is available** (per
`/aws/service/global-infrastructure/services/pcs/regions`) — 16 verified,
4 not yet run.

| Region (AZ ID) ¹ | Verified ² | Storage ³ (Lustre<br>/ OpenZFS) | GPU verified ⁴ | *(ref) CBML GPUs offered* ⁵ | Date (UTC) |
|---|---|---|---|---|---|
| **N. Virginia**<br>(`use1-az1`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_HA_2` | ✅ | P6-B300, P6-B200, P5, P5e, P5en, P4d, P4de | 2026-06-17 |
| **Ohio**<br>(`use2-az3`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_HA_2` | ✅ | P6-B200, P5, P5e, P5en, P4d | 2026-06-17 |
| **Oregon**<br>(`usw2-az3`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_HA_2` | ✅ P6-B300 | P6-B300, P6-B200, P5, P5e, P5en, P4d, P4de | 2026-06-17 |
| **Tokyo**<br>(`apne1-az1`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_HA_2` | — | P5, P5e, P5en | 2026-06-17 |
| **Mumbai**<br>(`aps1-az2`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_HA_2` | ✅ P6-B200 | P6-B200, P5, P5e, P5en | 2026-06-17 |
| **Osaka**<br>(`apne3-az2`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_1` ⁶ | — | — | 2026-06-17 |
| **Singapore**<br>(`apse1-az1`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_HA_2` | — | — | 2026-06-17 |
| **Sydney**<br>(`apse2-az2`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_HA_2` | — | P5, P5e, P5en | 2026-06-17 |
| **Frankfurt**<br>(`euc1-az2`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_HA_2` | — | — | 2026-06-17 |
| **Stockholm**<br>(`eun1-az2`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_HA_2` | — | P5, P5e, P5en | 2026-06-17 |
| **Ireland**<br>(`euw1-az1`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_HA_2` | — | — | 2026-07-07 |
| **Spain**<br>(`eus2-az1`) | ✅<br>(m7i/c7i) ⁷ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_HA_2` | — | P5en | 2026-07-07 |
| **London**<br>(`euw2-az2`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_1` ⁶ | — | P5, P5e, P5en | 2026-07-08 |
| **Paris**<br>(`euw3-az1`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_1` ⁶ | — | — | 2026-07-08 |
| **São Paulo**<br>(`sae1-az1`) | ✅ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_1` ⁶ | — | P5, P5e | 2026-07-08 |
| **Jakarta**<br>(`apse3-az1`) | ✅<br>(c7i) ⁷ | ✅ `PERSISTENT_2`<br>✅ `SINGLE_AZ_1` ⁶ | — | P5, P5e, P5en | 2026-07-08 |
| **Cape Town**<br>(af-south-1) | — | — | — | — | not yet run |
| **Milan**<br>(eu-south-1) | — | — | — | — | not yet run |
| **GovCloud US-East**<br>(us-gov-east-1) | — | — | — | P6-B300, P6-B200 | not yet run |
| **GovCloud US-West**<br>(us-gov-west-1) | — | — | — | P6-B200 | not yet run |

¹ **Region (AZ ID)** — the [AZ ID](https://docs.aws.amazon.com/ram/latest/userguide/working-with-az-ids.html)
of the `PrimarySubnetAZ` the run used. AZ IDs identify the same physical
location in every account, unlike AZ *names* such as `us-east-1a`, whose
mapping is randomized per account.

² **Verified (E2E)** — the `deploy-all` stack reaches CREATE_COMPLETE, the 6 monitoring
containers come up on the login node, and an Enroot/Pyxis container job runs
(`srun --container-image=ubuntu:22.04`). These three always pass or fail
together, so they are reported as one column.

³ **Storage** — FSx Lustre `/fsx` and FSx OpenZFS `/home` created & mounted,
with the deployment type that worked (Lustre on the first line of each cell,
OpenZFS on the second). Deployment-type support varies by region — check the
official per-region tables before deploying:
[FSx for Lustre deployment types](https://docs.aws.amazon.com/fsx/latest/LustreGuide/using-fsx-lustre.html)
and [FSx for OpenZFS availability by Region](https://docs.aws.amazon.com/fsx/latest/OpenZFSGuide/available-aws-regions.html).

⁴ **GPU verified** — a GPU CNG launched and ran on reserved capacity (Capacity
Block or ODCR); the instance family is noted where a test file records it.
**A "—" does NOT mean GPUs are unsupported there.** It only means a
reserved-capacity GPU run has not been exercised in that region yet — GPU
capacity is scarce and expensive, so GPU runs are done opportunistically where
capacity was purchased, not per-region. The GPU CNG templates are
region-agnostic; wherever the *(ref) CBML GPUs offered* column (or an ODCR)
provides capacity, they are expected to work as-is.

⁵ ***(ref) CBML GPUs offered*** — supplemental reference, not a test result:
P-family instance types purchasable as
[Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)
in that region, per the EC2 docs as of 2026-07-08. PCS supports CBML with
P6-B300 / P6-B200 / P5en / P5e / P5 / P4d.

⁶ **These regions do not support the default `OpenZFSDeploymentType=SINGLE_AZ_HA_2`**
(ap-northeast-3, eu-west-2, eu-west-3, sa-east-1, ap-southeast-3) — the default
deploy fails at the OpenZFS filesystem with `Invalid deploymentType (BadRequest)`.
Deploy there with `OpenZFSDeploymentType=SINGLE_AZ_1` (+ a valid
`HomeThroughput`, e.g. 256). This is the documented "not available in every
Region" case (see the parameter's description in `ml-cluster-prerequisites.yaml`);
`SINGLE_AZ_1` is available in all regions and is the safe fallback. FSx Lustre
showed no such regional variation — `LustreDeploymentType=PERSISTENT_2` (the
default) worked in all 16 regions, as the Lustre column shows.

⁷ **The default instance types are not offered in every region** — the
parenthesized types in the Verified column are what the run used instead:
eu-south-2 has no m6i/c6i (`LoginNodeInstanceType=m7i.4xlarge`,
`OnDemandInstanceType=c7i.4xlarge`), and ap-southeast-3 has no c6i
(`OnDemandInstanceType=c7i.4xlarge`). Rows without a parenthesis ran the
defaults. Check with
`aws ec2 describe-instance-type-offerings --location-type availability-zone`
before deploying.

---

## Canonical assets reused (not duplicated)

| Workload | Source in this repo | PCS-specific delta |
|---|---|---|
| NCCL `all_reduce` | [`micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch`](../../../micro-benchmarks/nccl-tests) | Partition name, Enroot import on login node |
| FSDP Llama-2 7B | [`3.test_cases/pytorch/FSDP`](../../../3.test_cases/pytorch/FSDP) | Cache on `/fsx`, 2 nodes |
| Megatron-LM GPT-3 (TP/PP/DP) | [`3.test_cases/megatron/megatron-lm`](../../../3.test_cases/megatron/megatron-lm) | Import `.sqsh` to `/fsx`, data under `/fsx/gpt2/`, 4 nodes |
| GPU Health Check | [`4.validation_and_observability/2.gpu-cluster-healthcheck`](../../../4.validation_and_observability/2.gpu-cluster-healthcheck) | sbatch wrapper, partition name |

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
