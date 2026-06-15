# AWS PCS — Test & Validation Guide

This directory is a single guide for validating an AWS PCS cluster deployed from the
templates in [`../assets`](../assets). Each test below lists **what to run** and the
**expected result**.

For non-test operational guidance (Slurm version trade-offs, AMI single-version rule,
`MonitoringVersion` migration, the `DcgmExporterImage` default and when to change it,
AMI pinning, FSx deployment-type ↔ throughput coupling), see
[`../docs/OPERATIONS.md`](../docs/OPERATIONS.md).

Rather than ship its own copies, this guide reuses the repository's canonical benchmark
and training assets and documents only the **PCS-specific deltas** (queue/partition names,
running the Enroot import on the login node, putting caches on `/fsx`):

| Stage | Canonical asset to use | PCS-specific delta documented here |
|---|---|---|
| NCCL `all_reduce` over EFA | [`micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch`](../../../micro-benchmarks/nccl-tests) | [Test 6](#test-6-nccl-multi-node-efa) — import on the login node, partition name |
| FSDP Llama-2 7B training | [`3.test_cases/pytorch/FSDP`](../../../3.test_cases/pytorch/FSDP) | [Test 7](#test-7-fsdp-sample-training) — venv **or** Enroot container; cache on `/fsx`, 2 nodes |
| GPU sanity (nvidia-smi) | (one `srun` line, no script) | [Test 5](#test-5-p5p6-gpu-multi-nic) |

All Slurm commands run as the **`ubuntu`** user from the login node (SSM session or
SSH). Slurm binaries are only on `PATH` in a login shell — over SSM/SSH wrap commands
as `bash -lc "sinfo; squeue"`.

---

## Pre-merge full test matrix (run before opening / updating the PR)

Run this **complete set** before merging template or script changes. Several real bugs
only appeared in specific combinations (a 25.05-only `MetricsType` rejection, a Pyxis
SPANK plugin built for the wrong Slurm version, a first-boot post-install failure), so
do **not** assume a single 25.11 GPU run covers everything.

| # | Dimension | What to cover | Why it matters |
|---|---|---|---|
| 1 | **Every Slurm version** | Deploy one cluster per `SlurmVersion` `AllowedValues` value (**25.05 and 25.11**) and run a Pyxis `--container-image` job on each | `MetricsType` is 25.11-only (25.05 cluster create fails if it's set unconditionally); the Pyxis SPANK plugin is ABI-locked to its Slurm version, so a wrong-version `spank_pyxis.so` stops slurmd. **Any change to `scripts/install-enroot-pyxis.sh` MUST be retested on all supported versions.** |
| 2 | **First-boot container runtime install** | Default `PostInstallScriptUrl` runs `install-enroot-pyxis.sh` at first boot; verify Enroot/Pyxis lands and a `--container-image` job works → [Test 2](#test-2-enrootpyxis-container-runtime-first-boot-install) | The default path used by every cluster that doesn't override `AmiId`. Bugs in `install-enroot-pyxis.sh` only show up on a clean first boot. |
| 3 | **First-boot from a clean deploy** | Validate on a **freshly deployed** cluster, not a node you hand-patched | Post-install runs during cloud-init *before* slurmd/profile.d/controller exist; bugs there (e.g. version detection, `set -e` aborts) only show on a clean first boot, not after a live re-run. |
| 4 | **CPU queue** | `DeployOnDemandCNG=true`; Pyxis container job on `cpu1` | Baseline; also the cheapest way to exercise items 1–3 without GPU capacity. |
| 5 | **Each GPU family** (as capacity allows) | `p5`/`p5e`/`p5en`, `p6-b200`, `p6-b300`: nvidia-smi, NCCL all_reduce, FSDP | EFA NIC layout and the dcgm-exporter image differ per family (see notes). |
| 6 | **Monitoring** | 6 login containers up, all Prometheus targets healthy, GPU dashboards populate on every supported GPU family with the default `DcgmExporterImage` (DCGM 4.5.2 by digest) | — |
| 7 | **Template lint** | `aws cloudformation validate-template` on every edited `assets/*.yaml` | Catches structural errors before a deploy round-trip. |
| 8 | **Pre-baked AMI build path** (when touched) | If `pcs-ready-dlami-with-enroot-pyxis.yaml` or any code it bakes (`scripts/install-enroot-pyxis.sh`) changes: build an AMI per supported `SlurmVersion`, then deploy a cluster with `AmiId=<ami-xxx>` + `PostInstallScriptUrl=""` and run a container job → [Test 8](#test-8-pre-baked-ami-build-standalone-dlami-template) | Independent path: the cluster stack does NOT run Image Builder, so an `install-enroot-pyxis.sh` fix is only in the AMI after a rebuild. The AMI is single-Slurm-version by design — `SlurmVersion` on the DLAMI stack must match the cluster's `SlurmVersion` (the SPANK plugin is ABI-locked). **Skip this row only if neither the AMI build template nor the install script changed.** |
| 9 | **CPU EFA path** (when touched) | If `add-cng.yaml`'s EFA wiring (`EnableEfa` / `EfaInterfaceCount` / `PlacementGroupName`) or the deploy-all forwarding (`OnDemand{EnableEfa,EfaInterfaceCount,PlacementGroupName}`) changes: deploy with `OnDemandEnableEfa=true` on at least one EFA-capable HPC type (e.g. hpc7a.96xlarge, 2 NICs), verify `lspci`/`fi_info` show the EFA NICs, and run a 2-node OSU `osu_mbw_mr` → [Test 9](#test-9-efa-on-cpu-hpc-instances-hpc6a--hpc7a--hpc8a) | EFA enables a different LaunchTemplate shape (`NetworkInterfaces` block with `InterfaceType=efa`, mutually exclusive with `SecurityGroupIds`); a regression here only shows on a real EFA deploy, not template-validate. **Skip this row only if no EFA-related wiring was touched.** |
| 10 | **FSx storage health + performance** | (A) Both filesystems mount with correct options; read/write sanity; FSx-side params match CFN. (B) Performance regression/improvement test: stat IOPS, sequential read/write BW, multi-node concurrent stat, flock correctness → [Test 10](#test-10-fsx-storage-health-and-performance) | Run Part A after every deploy; run Part B before/after any Lustre-related change (mount options, lctl tunables, stripe, FSxLustreEnableEfa, Lustre version). A regression >10% on any metric should block the change. |

Tests 1–10 below are the per-item how-to. The single-cluster shortcut (one deploy that
covers monitoring + CPU + one GPU family) is fine for iterating; rows 8 and 9 are
separate paths run only when their inputs change. The full matrix above is the bar
for **merge**.

---

## Coverage matrix

What this guide covers, and the template/parameter that exercises it:

| Dimension | Options to test | How |
|---|---|---|
| **Monitoring** | enabled (default) | `DeployMonitoring=true` → [Test 1](#test-1-monitoring-stack) |
| **Container runtime — default first-boot path** | UserData install on every node | Default `PostInstallScriptUrl` (no AMI override) → [Test 2](#test-2-enrootpyxis-container-runtime-first-boot-install) |
| **CPU nodes** | `c6i`/`c7i` etc. | `DeployOnDemandCNG=true` → [Test 3](#test-3-cpu-queue) |
| **Single-NIC GPU** | `g5`/`g6` | On-Demand CNG with a G-series type → [Test 4](#test-4-g-series-gpu-single-nic) |
| **Multi-NIC GPU** | `p5`/`p5e`/`p5en`, `p6-b200`, `p6-b300` | `DeployPseriesCNG=true` + `PseriesInstanceType` → [Test 5](#test-5-p5p6-gpu-multi-nic) |
| **NCCL / EFA** | 2-node all_reduce | [Test 6](#test-6-nccl-multi-node-efa) |
| **Sample training** | FSDP Llama-2 7B | [Test 7](#test-7-fsdp-sample-training) |
| **Container runtime — pre-baked AMI path** | Standalone DLAMI build + cluster pinned to its output | `pcs-ready-dlami-with-enroot-pyxis.yaml` (separate stack) → cluster with `AmiId=ami-xxx` + `PostInstallScriptUrl=""` → [Test 8](#test-8-pre-baked-ami-build-standalone-dlami-template) |
| **EFA on CPU HPC instances** | hpc6a (1 NIC), hpc7a / hpc8a (2 NIC) | `OnDemandEnableEfa=true` + `OnDemandEfaInterfaceCount=1\|2` (deploy-all) or `EnableEfa=true` + `EfaInterfaceCount=1\|2` (modular `add-cng.yaml`); auto-creates a per-CNG cluster placement group, override with `OnDemandPlacementGroupName` / `PlacementGroupName` to share → [Test 9](#test-9-efa-on-cpu-hpc-instances-hpc6a--hpc7a--hpc8a) |
| **FSx storage health + performance** | Lustre + OpenZFS mounts, read/write, FSx-side parameters honored; performance regression/improvement test for any Lustre-related change | Default deploy already mounts both filesystems; verify + benchmark → [Test 10](#test-10-fsx-storage-health-and-performance) |

A single `pcs-ml-cluster-deploy-all.yaml` deploy with `DeployMonitoring=true`,
`DeployOnDemandCNG=true`, and `DeployPseriesCNG=true` exercises Tests 1–7 in one cluster.
Test 8 is a **separate** flow (independent stack for the AMI, then a cluster pinned to
its output); run it when `pcs-ready-dlami-with-enroot-pyxis.yaml` or
`scripts/install-enroot-pyxis.sh` change. See [the README](../README.md#5-usage-examples)
for deploy commands; set `PseriesInstanceType` to the GPU family you want to validate.

---

## Verified configurations

Key configurations validated on real hardware with these templates (representative
results; exact bandwidth/throughput vary with NCCL/EFA versions and message size):

| Config | Region | Capacity | Monitoring | NCCL all_reduce (2-node peak busbw) | FSDP Llama-2 7B (2-node) |
|---|---|---|---|---|---|
| **2× p6-b200.48xlarge** (16× B200) | us-west-2 | Capacity Block | ✅ v2.9.1, 16 GPUs in Grafana | **~654 GB/s** @16 GiB (EFA, `found 8 nics`, `#wrong 0`) | **~223 TFLOPS/GPU, ~86k tok/s** |
| **2× p6-b300.48xlarge** (16× B300) | us-west-2 | Capacity Block | ✅ v2.9.1, 16 B300 GPUs in Grafana with the default `DcgmExporterImage` (DCGM 4.5.2 by digest) ‡ | **~751 GB/s** @64 GiB (EFA, `found 16 nics`, `#wrong 0`) † | **~205 TFLOPS/GPU, ~79k tok/s** (venv); **~193 TFLOPS/GPU** (container) |
| **2× p5.48xlarge** (16× H100) | us-east-2 | Capacity Block | ✅ | **~480 GB/s** (EFA, `found 32 nics`, `#wrong 0`) | ~60 TFLOPS/GPU |
| **Slurm 25.05, CPU + PostInstall** | us-west-2 | On-Demand | ✅ v2.9.1 (no `slurm_openmetrics` job §) | first-boot Pyxis OK, `srun --container-image=ubuntu:22.04` clean | n/a |
| **Slurm 25.11, CPU + PostInstall** | us-west-2 | On-Demand | ✅ v2.9.1 incl. Slurm OpenMetrics | first-boot Pyxis OK | n/a |
| **Login + CPU (`c6i`)** | us-west-2 / us-east-* | On-Demand | ✅ all targets up | n/a | n/a |
| **Grafana public access** (login-only SG) | us-west-2 | — | ✅ reachable at `https://<login-public-ip>/grafana/` from the allowed CIDR | — | — |

### HPC EFA on CPU instances (`OnDemandEnableEfa=true`)

OSU MPI micro-benchmarks 7.4 on 2 nodes, Slurm 25.11, AWS Open MPI 4.1.7
(`/opt/amazon/openmpi`), libfabric provider `efa`. Tuned env: `FI_PROVIDER=efa`,
huge page on (default), `FI_EFA_FORK_SAFE=1`, `OMPI_MCA_pml=cm`, `OMPI_MCA_mtl=ofi`,
`OMPI_MCA_mtl_ofi_provider_include=efa`. All deploys used `OnDemandEnableEfa=true`
with the matching `OnDemandEfaInterfaceCount`; the cluster placement group was
auto-created per-CNG by `add-cng.yaml`.

| Instance | NICs | Spec aggregate ¶ | `osu_latency` 1B | `osu_bw` peak (1 pair) | `osu_bibw` peak (1 pair) | `osu_mbw_mr` peak (16 pair × 2 nodes) | `osu_allreduce` 1B (32 ranks) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **hpc6a.48xlarge** | 1 | 100 Gbps | 13.78 µs | 96.3 Gbps (12.04 GB/s) | 152 Gbps †† | **97.6 Gbps** (97.6% of spec) | 27.5 µs |
| **hpc7a.96xlarge** | 2 | 300 Gbps | 14.27 µs | 95.6 Gbps (11.95 GB/s) | 156 Gbps | **263 Gbps** (87.6% of spec) | 23.4 µs |
| **hpc8a.96xlarge** | 2 | 300 Gbps | **10.31 µs** ★ | **210 Gbps** (26.27 GB/s) | **363 Gbps** †† | **341 Gbps** ‡‡ | **17.4 µs** ★ |

¶ AWS docs ([HPC instance specs](https://docs.aws.amazon.com/ec2/latest/instancetypes/hpc.html))
"Baseline / Burst bandwidth (Gbps)" column. The notes also state "you must attach at
least 2 ENIs, to separate network cards, to achieve [aggregate] throughput" with a
per-ENI cap of 150 Gbps for hpc7a and 170 Gbps for hpc6id; **per-ENI cap for hpc8a is
not stated in AWS docs**.

†† `osu_bibw` measures bidirectional bandwidth on a single pair (simultaneous
send+recv). It can exceed the uni-directional spec because it counts both
directions; not directly comparable to the "300 Gbps" aggregate spec.

‡‡ hpc8a's `osu_mbw_mr` reading exceeds the docs' aggregate spec (300 Gbps). The
reading is reproducible across two independent runs (job 6 = 42.68 GB/s, job 7 =
43.85 GB/s), so it isn't a one-off artifact. Possible explanations include (a) AWS
docs spec is per-direction sustained while `osu_mbw_mr` reports forward-direction
peak, (b) Nitro v6 / EFAv4 efficiency on hpc8a leaves headroom above the published
number, (c) ack/control traffic in reverse direction inflates the receiver-side
counter view. **NIC-level Prometheus counters** (`node_amazonefa_tx_bytes` and
`rx_bytes`, available via the v2.7+ `efa-metrics.sh` textfile collector) can resolve
which is the case for a given run; in a 30s-update collector the wire-level peak
shows below the OSU-reported peak because the OSU phases for each message size are
shorter than the textfile sample window.

★ hpc8a is fastest on every metric. The Nitro v6 / EFAv4 generation lift over hpc7a
(Nitro v4) shows up most clearly in latency (~30% better) and single-pair bandwidth
(~2× — single pair on hpc7a hits the 150 Gbps per-ENI cap, hpc8a does not appear to
hit one).

**Tuning notes** (collected during these runs):
- **`osu_bw` (single pair) understates aggregate fabric bandwidth.** A single MPI
  pair uses one libfabric endpoint; on instances with 2 NICs, the second NIC is idle.
  Use `osu_mbw_mr -np 32 -N 16` for the realistic aggregate number; `osu_bw` only
  matches the per-ENI cap.
- **Disabling huge pages costs throughput.** An earlier hpc7a run with
  `FI_EFA_USE_HUGE_PAGE=0` showed 68 Gbps `osu_bw`; leaving the variable unset (= 1
  default) brought it to 95.6 Gbps on the same instance.
- **Open MPI 4.1.7 on the PCS-Ready DLAMI has no `pml=ofi`.** Set
  `OMPI_MCA_pml=cm` and `OMPI_MCA_mtl=ofi` instead — the `cm` PML dispatches to the
  OFI MTL. Setting `OMPI_MCA_pml=ofi` directly fails with "mca_pml_base_open()
  failed → Returned 'Not found' (-13)".
- **EFA monitoring works with the default v2.9.1 stack.** The `efa-metrics.sh`
  textfile collector emits `node_amazonefa_*` series for `tx_bytes`, `rx_bytes`,
  `recv_bytes`, `send_bytes`, RDMA `read/write` bytes/work-requests, retransmits,
  and unresponsive-remote-endpoint events. The Compute Node Details Grafana
  dashboard has dedicated EFA panels for these. Sampling cadence is 30 sec
  (textfile-collector timer), so short OSU sub-tests show below their wall-clock peak.

### Stack creation times (measured, deploy-all)

Wall-clock from `aws cloudformation create-stack` to top-level `CREATE_COMPLETE`, on a
warmed account in us-west-2 (no first-time-in-region provisioning). Useful for sizing
how long a deploy round-trip takes during a review cycle.

| Configuration | First-boot path | Time |
|---|---|---|
| 25.05, CPU only | Default `PostInstallScriptUrl` (Enroot/Pyxis at first boot) | **~24m** |
| 25.11, CPU only | Default `PostInstallScriptUrl` (Enroot/Pyxis at first boot) | **~31m** |
| 25.11, GPU + CPU (Capacity Block) | Default `PostInstallScriptUrl` (Enroot/Pyxis at first boot) | **~44m** |
| 25.05/25.11, CPU only, **pre-baked AMI** | Custom AMI built **separately** via `pcs-ready-dlami-with-enroot-pyxis.yaml` (~30m one-time), then cluster deployed with `AmiId=<ami-xxx>` + `PostInstallScriptUrl=""` | AMI build ~30m + cluster ~25m |

Notes: Prerequisites (VPC + dual FSx) is the long-pole on the cluster path (~20-25m);
the default first-boot path adds the per-boot Enroot/Pyxis install (~2-3m). Pre-baking
the AMI is a separate ~30m one-time job — its cost amortizes when you redeploy or
scale clusters that share the AMI. GPU adds the P-series CNG and the GPU node first boot.

Notes:
- **Deploy path:** all of the above came up from `pcs-ml-cluster-deploy-all.yaml` with
  the default first-boot Enroot/Pyxis installer + `DeployMonitoring=true`.
- **Container runtime:** validated via first-boot UserData install (the default); the
  pre-baked-AMI path uses the same `install-enroot-pyxis.sh` baked into the DLAMI by
  `pcs-ready-dlami-with-enroot-pyxis.yaml`.
- **Pre-baked AMI path validated** (us-west-2, CPU-only): build the AMI with the
  standalone DLAMI template, take its `DLAMIforPCSAmiId` output, deploy the cluster
  with `AmiId=<ami-xxx>` and `PostInstallScriptUrl=""`. ImageBuilder bakes Enroot 3.5.0
  + per-version Pyxis into the DLAMI; login/compute nodes boot ready, no first-boot
  install. The AMI is single-Slurm-version on purpose — see
  [docs/OPERATIONS.md §2](../docs/OPERATIONS.md#2-container-runtime-postinstall-vs-ami-build).
- **EFA interface count** is derived from the instance type (p5/p5e = 32, p5en = 16,
  p6-b200 = 8, p6-b300 = 16-of-17); see [README GPU compute](../README.md#gpu-compute-p5p6).
- **FSDP runs both ways** — validated with a shared-`/fsx` **venv** (~200 TFLOPS/GPU) and
  with an **Enroot/Pyxis container** (`CONTAINER_IMAGE=/fsx/pytorch-fsdp.sqsh`, ~193
  TFLOPS/GPU) on the same 2× p6-b300. See Test 7 for both. **FSDP loss** stays at ln(vocab)
  in this smoke test — a known dataloader/vocab quirk of the test case, not a cluster issue.
- **† B300 NCCL bandwidth scales past 16 GiB.** A 2-node / **16 GiB** all_reduce reaches only
  ~654 GB/s busbw — it doesn't saturate all 16 EFA cards. Re-measured with larger messages on
  2× p6-b300: **64 GiB → ~751 GB/s** busbw (`found 16 nics`, `#wrong 0`), still climbing, so
  16 GiB was indeed unsaturated. **128 GiB and 256 GiB OOM** (the all_reduce buffer exceeds
  B300 GPU memory), so ~64 GiB is the practical max single-buffer size here; for a true peak,
  scale to more nodes rather than larger buffers.
- **‡ B300 GPU metrics work with the default `DcgmExporterImage`** — a DCGM 4.5.2
  build pinned by digest (`nvcr.io/nvidia/k8s/dcgm-exporter@sha256:a7ad6547…`),
  validated on 2× p6-b300. Override only if you need to pin to a different DCGM build;
  see [docs/OPERATIONS.md §3.1](../docs/OPERATIONS.md#31-dcgmexporterimage-the-default-and-when-to-change-it).
- **§ Slurm OpenMetrics is 25.11+ only.** On 25.05 the Slurm dashboards stay empty (the
  rest of monitoring works fine). See
  [docs/OPERATIONS.md §1](../docs/OPERATIONS.md#1-slurm-version-selection).

---

## Test 1: Monitoring stack

With `DeployMonitoring=true` (default), Prometheus/Grafana/exporters install on the
login node and DCGM/node exporters on the compute nodes.

```bash
# On the login node:
docker ps --format "table {{.Names}}\t{{.Status}}"          # login: prometheus, grafana, nginx, cloudwatch-exporter, node-exporter, pushgateway
tail -5 /var/log/monitoring-install.log                     # ends "...complete (exit 0)"
ls -la /opt/aws-parallelcluster-monitoring                  # installed on node-local /opt (NOT /home)
curl -s http://localhost:9090/api/v1/targets | \
  python3 -c 'import sys,json;[print(t["labels"].get("instance"),t["health"]) for t in json.load(sys.stdin)["data"]["activeTargets"]]'
curl -s http://localhost:6817/metrics | head               # Slurm OpenMetrics
```

**Expected:** the six login containers are `Up`; install log exits 0; the tree is under
`/opt`; all Prometheus targets `up`; Slurm OpenMetrics returns Prometheus-format text.
For dashboard access (SSM port-forward or public CIDR) see
[README §8 Monitoring](../README.md#8-monitoring). Use `MonitoringVersion=v2.9.1`+ on PCS.

---

## Test 2: Enroot/Pyxis container runtime (first-boot install)

This is the default path used by every cluster that doesn't override `AmiId`:
`PostInstallScriptUrl` runs `install-enroot-pyxis.sh` once on each node at first boot.
The pre-baked-AMI path is validated separately as [Test 8](#test-8-pre-baked-ami-build-standalone-dlami-template).

Deploy `pcs-ml-cluster-deploy-all.yaml` with both `AmiId` and `PostInstallScriptUrl`
left at their defaults (so SSM auto-resolves the latest PCS-Ready DLAMI and
post-install runs the Enroot/Pyxis installer), then on any node:

```bash
which enroot                                                       # /usr/bin/enroot
ls /opt/aws/pcs/scheduler/slurm-*/lib/slurm/spank_pyxis.so         # per-version Pyxis SPANK plugin
cat /etc/aws/pcs/scheduler/slurm-*/plugstack.conf.d/pyxis.conf     # points at the matching .so
tail -1 /var/log/pcs-post-install.log                              # "...completed (exit 0)"
```

**Expected:** `enroot` on `PATH`; a `spank_pyxis.so` under the **cluster's** Slurm version
dir, and the plugstack `pyxis.conf` referencing that exact path; post-install log exits 0.
The Test 1/6/7 container jobs are the functional proof that Pyxis works.

> **⚠️ Regression-test rule for `scripts/install-enroot-pyxis.sh`.** This script has bitten
> us repeatedly in ways a single 25.11 GPU run does not catch. **Any change to it MUST be
> retested across the full matrix at the top of this guide**, specifically:
> - **All supported Slurm versions** (25.05 **and** 25.11). The Pyxis SPANK plugin is
>   ABI-locked to its Slurm version — a plugin built for the wrong version stops slurmd from
>   starting (`Incompatible Slurm plugin version`). The script builds Pyxis for the version
>   passed in `PCS_SLURM_VERSION` and installs the `.so` to a per-version path; a regression
>   here only shows on the *other* version.
> - **The pre-baked AMI path too** ([Test 8](#test-8-pre-baked-ami-build-standalone-dlami-template)).
>   `pcs-ready-dlami-with-enroot-pyxis.yaml` carries its **own copy** of the Enroot/Pyxis
>   steps in its Image Builder UserData — editing `install-enroot-pyxis.sh` does **not**
>   change the AMI path until you rebuild. Build an AMI per supported `SlurmVersion`,
>   deploy a cluster pinned to it (`AmiId=<ami-xxx>` + `PostInstallScriptUrl=""`), and
>   run a container job.
> - **On a clean first boot**, not a hand-patched node — post-install runs before
>   slurmd/profile.d/controller exist, and several bugs only appear there.

---

## Test 3: CPU queue

Deployed by default as `cpu1` (`DeployOnDemandCNG=true`, `c6i.4xlarge`, 0–4 dynamic).

```bash
sinfo                                          # cpu1 partition present, nodes idle~
srun --partition=cpu1 --nodes=1 hostname       # a node powers up and runs
```

**Expected:** `cpu1` shows in `sinfo`; a dynamically-scaled node launches and the job
returns its hostname.

---

## Test 4: G-series GPU (single NIC)

Single-NIC GPU instances (`g5`/`g6`) use `add-cng.yaml` — deploy them as the On-Demand
CNG, e.g. `OnDemandInstanceType=g6.12xlarge`, `OnDemandQueueName=gpu-g6` (see
[README Example 2](../README.md#5-usage-examples)).

```bash
srun --partition=gpu-g6 --nodes=1 --gres=gpu:1 \
  --container-image=docker://nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**Expected:** the container runs and `nvidia-smi` lists the node's GPU(s) — confirms
single-NIC GPU + Pyxis on a G-series queue.

---

## Test 5: P5/P6 GPU (multi-NIC)

Multi-NIC GPU node groups, selected automatically by `PseriesInstanceType`
(`p5.48xlarge`/`p5e`/`p5en` → `add-cng-p5.yaml`; `p6-b200.48xlarge` → `add-cng-p6-b200`;
`p6-b300.48xlarge` → `add-cng-p6-b300`). The EFA interface count is derived from the
type — no parameter to set. A one-line interactive `srun` is enough for a GPU sanity
check (no batch script needed); set `--partition` to your GPU queue:

```bash
srun --partition=gpu-p6b200 --nodes=1 --gres=gpu:8 \
  --container-image=docker://nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**Expected:** `nvidia-smi` lists 8 GPUs of the expected model (H100 / H200 / B200 /
B300). This confirms the multi-NIC launch template booted and Pyxis works on the GPU
node. EFA itself is exercised by Test 6.

---

## Test 6: NCCL multi-node (EFA)

2-node × 8-GPU `all_reduce_perf` over EFA, using the repo's canonical launcher
[`micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch`](../../../micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch)
(it reads `$IMAGE`, default `/fsx/nccl-tests.sqsh`). Only two PCS-specific deltas:

1. **Import the image on the login node** — `enroot import` builds its overlayfs on the
   node-local root disk (the login node has a 300 GiB root via `RootVolumeSize`); FSx
   Lustre can't host that overlay, so only the resulting `.sqsh` goes to shared `/fsx`.
   Pin a specific image tag for reproducible numbers (don't use `latest`):

   ```bash
   # On the login node (direct, not a batch job). enroot URI form is
   # docker://[REGISTRY#]REPO:TAG — the registry needs a '#', or it 401s on Docker Hub.
   TAG=cuda12.8.1-efa1.43.2-ofiv1.16.3-ncclv2.27.7-1-testsv2.16.9
   enroot import -o /fsx/nccl-tests.sqsh "docker://public.ecr.aws#hpc-cloud/nccl-tests:${TAG}"
   ```

2. **Submit with your GPU queue** — set the partition to the queue you deployed
   (`gpu-p5` / `gpu-p6b200` / `gpu-p6b300`); the canonical script defaults to 2 nodes,
   8 tasks/node:

   ```bash
   cd /fsx && sbatch --partition=gpu-p6b200 \
     /fsx/awsome-distributed-ai/micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch
   ```

**Expected** (in `nccl-all_reduce_perf_<jobid>.out`):
- EFA is the provider: `NET/OFI Selected provider is efa, fabric is efa-direct (found N nics)`
  (N = EFA interface count: 32 for p5/p5e, 16 for p5en, 8 for p6-b200, 16 for p6-b300).
- Correctness: `# Out of bounds values : 0 OK`, every size `#wrong: 0`.
- `busbw` rises with message size — on 2× p6-b300, ~654 GB/s at 16 GiB but **~751 GB/s at
  64 GiB** (16 cards aren't saturated at 16 GiB); ~480 GB/s on 2× p5.

> **Sizing the sweep on B300.** The canonical script sweeps to `-e 16G`. On p6-b300 that
> under-reports peak bandwidth — raise it to `-e 64G` (edit the `all_reduce_perf` line) to
> see the cards saturate. **Don't go to 128 GiB/256 GiB**: the all_reduce buffer exceeds
> B300 GPU memory and the job is OOM-killed. For a higher peak, add nodes, not buffer size.

---

## Test 7: FSDP sample training

A short FSDP Llama-2 7B run, using the repo's canonical training case
[`3.test_cases/pytorch/FSDP`](../../../3.test_cases/pytorch/FSDP) and its
[`slurm/llama2_7b-training.sbatch`](../../../3.test_cases/pytorch/FSDP/slurm/llama2_7b-training.sbatch).
Follow that case's README; the only PCS-specific deltas are where things live on the
shared filesystems and the node count:

1. **Build the venv on shared `/fsx`** so every compute node sees it (the canonical
   `slurm/create_venv.sh` creates `./env` in place — run it from a `/fsx` checkout):

   ```bash
   cd /fsx && git clone --depth 1 https://github.com/awslabs/awsome-distributed-ai.git
   cd /fsx/awsome-distributed-ai/3.test_cases/pytorch/FSDP/slurm && bash create_venv.sh
   export HF_TOKEN=hf_xxx          # gated Llama-2 tokenizer
   ```

2. **Keep the HuggingFace cache on `/fsx` (Lustre), not `/home` (NFS).** Concurrent rank
   file-locking on NFS throws `OSError: [Errno 116] Stale file handle`; export
   `HF_HOME=/fsx/.hf-cache` before submitting.

3. **Submit 2 nodes** (the canonical sbatch defaults to 4) on your GPU queue. The venv must
   be on `PATH` for `torchrun` to resolve on every node — point `PATH` at the shared venv via
   `--export` (the canonical sbatch doesn't `activate` it):

   ```bash
   sbatch --nodes=2 --partition=gpu-p6b200 \
     --export=ALL,PATH=/fsx/awsome-distributed-ai/3.test_cases/pytorch/FSDP/slurm/env/bin:$PATH,HF_HOME=/fsx/.hf-cache \
     llama2_7b-training.sbatch
   ```

### Option B — run it in an Enroot/Pyxis container instead of the venv

The same canonical sbatch switches to container mode when `CONTAINER_IMAGE` is set (it adds
`--container-image`/`--container-mounts` and runs `./train.py` inside). Build the image once
on the login node and submit with `CONTAINER_IMAGE` — no venv needed:

```bash
# On the login node (300 GiB root disk + Docker), build + import to /fsx:
cd /fsx/awsome-distributed-ai/3.test_cases/pytorch/FSDP
sudo docker build -t fsdp:pytorch -f Dockerfile .
enroot import -o /fsx/pytorch-fsdp.sqsh dockerd://fsdp:pytorch

# Submit (container mode; mounts $(pwd) into /fsx inside the container):
cd slurm && sbatch --nodes=2 --partition=gpu-p6b300 \
  --export=ALL,CONTAINER_IMAGE=/fsx/pytorch-fsdp.sqsh,HF_HOME=/fsx/.hf-cache,HF_TOKEN=hf_xxx \
  llama2_7b-training.sbatch
```

> If the Dockerfile's `FROM` tag (a `public.ecr.aws/hpc-cloud/nccl-tests` tag) has been
> rotated out of the registry, substitute a current tag from that repo before building.

**Expected** (in `logs/llama2_7b-FSDP_<jobid>.out`), either path:
- NCCL initializes over EFA (`found N nics`) and training logs ~100 steps + a validation
  step, saving checkpoints under `./checkpoints`.
- Throughput per step, e.g. on 2× p6-b300 **~200 TFLOPS/GPU, ~77k tokens/s** (venv) /
  **~193 TFLOPS/GPU** (container); ~60 TFLOPS on 2× p5/H100. (Loss is constant at ln(vocab)
  in this smoke test — a known dataloader/vocab quirk of the test case, not a cluster problem.)

> **Multi-NIC tip:** the canonical sbatch already sets `NCCL_SOCKET_IFNAME=^docker,lo,veth,eth`
> (NCCL auto-selects). Do **not** pin a single interface on P5/P6 — all NICs share one
> subnet and pinning one breaks the cross-node NCCL bootstrap ring.

---

## Test 8: Pre-baked AMI build (standalone DLAMI template)

**When to run:** when `pcs-ready-dlami-with-enroot-pyxis.yaml` or any code it bakes
in (`scripts/install-enroot-pyxis.sh`) changes — the cluster stack does NOT run
Image Builder, so a fix to the install script is only in the AMI after a rebuild.
Skip this test if you only touched the cluster templates.

This is an **independent flow**, not a deploy-all parameter: build an AMI with the
standalone template, then deploy a cluster pinned to that AMI ID with
`PostInstallScriptUrl=""` so nothing else runs at boot.

The AMI is **single-Slurm-version by design** (Pyxis SPANK plugin ABI is
version-locked) — so when you run this test, run it for **every supported
`SlurmVersion`** that the install-script change could affect (typically both 25.05
and 25.11).

### Step 1 — build the AMI (~30 min one-time per Slurm version)

```bash
SLURM_VERSION=25.11   # repeat for 25.05 if relevant

aws cloudformation create-stack \
  --stack-name pcs-dlami-${SLURM_VERSION/./} \
  --template-url https://midaisuk-llm-dev.s3.amazonaws.com/templates/pcs-ready-dlami-with-enroot-pyxis.yaml \
  --parameters ParameterKey=SlurmVersion,ParameterValue=${SLURM_VERSION} \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --profile claude --region us-west-2

aws cloudformation wait stack-create-complete \
  --stack-name pcs-dlami-${SLURM_VERSION/./} \
  --profile claude --region us-west-2

AMI_ID=$(aws cloudformation describe-stacks \
  --stack-name pcs-dlami-${SLURM_VERSION/./} \
  --query 'Stacks[0].Outputs[?OutputKey==`DLAMIforPCSAmiId`].OutputValue' \
  --output text --profile claude --region us-west-2)
echo "$AMI_ID"   # ami-0xxxxxxxxxxxxxxxx
```

### Step 2 — deploy a cluster pinned to that AMI

```bash
aws cloudformation create-stack \
  --stack-name pcs-amitest-${SLURM_VERSION/./} \
  --template-url https://midaisuk-llm-dev.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-west-2a \
    ParameterKey=SlurmVersion,ParameterValue=${SLURM_VERSION} \
    ParameterKey=AmiId,ParameterValue=${AMI_ID} \
    ParameterKey=PostInstallScriptUrl,ParameterValue= \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --profile claude --region us-west-2
```

### Step 3 — verify the bake landed and a container job runs

On any node from the new cluster:

```bash
which enroot                                                       # /usr/bin/enroot (pre-baked)
ls /opt/aws/pcs/scheduler/slurm-${SLURM_VERSION}/lib/slurm/spank_pyxis.so  # built for matching Slurm
cat /etc/aws/pcs/scheduler/slurm-${SLURM_VERSION}/plugstack.conf.d/pyxis.conf  # references the .so
test ! -s /var/log/pcs-post-install.log && echo "post-install did not run (PostInstallScriptUrl='')"
```

Then a container job through the login node (same form as Test 2):

```bash
srun --partition=cpu1 --nodes=1 --ntasks=1 \
  --container-image=ubuntu:22.04 bash -c "echo PYXIS_FROM_AMI_OK"
```

**Expected:** `enroot` and the per-version Pyxis files exist **without** the
post-install hook running (because `PostInstallScriptUrl=""`); the container job
prints `PYXIS_FROM_AMI_OK`. Slurmd starts cleanly (no `Incompatible Slurm plugin
version` in `journalctl -u slurmd`).

### Step 4 — clean up

```bash
aws cloudformation delete-stack --stack-name pcs-amitest-${SLURM_VERSION/./} --profile claude --region us-west-2
aws cloudformation delete-stack --stack-name pcs-dlami-${SLURM_VERSION/./}   --profile claude --region us-west-2
```

The DLAMI stack's AMI itself is **not** automatically deregistered when the stack is
deleted — if you need to free its EBS snapshots, deregister the AMI manually
(`aws ec2 deregister-image --image-id $AMI_ID`) and delete the snapshot.

---

## Test 9: EFA on CPU HPC instances (hpc6a / hpc7a / hpc8a)

**When to run:** when `add-cng.yaml`'s EFA wiring (`EnableEfa`, `EfaInterfaceCount`,
`PlacementGroupName`) or the deploy-all forwarding params (`OnDemandEnableEfa`,
`OnDemandEfaInterfaceCount`, `OnDemandPlacementGroupName`) change. Skip if only
GPU/monitoring/AMI paths were touched.

This validates that the on-demand CPU CNG actually launches with EFA NICs in a
cluster placement group, and that MPI / libfabric over EFA works end-to-end. The
verified-configurations table above documents the bandwidth numbers; this section
documents the **how-to** so a contributor can reproduce.

### Step 1 — deploy with EFA on the CPU CNG

```bash
# hpc7a / hpc8a have 2 EFA NICs; hpc6a has 1.
INSTANCE_TYPE=hpc7a.96xlarge
EFA_NICS=2

# AZ availability is region-specific — confirm with describe-instance-type-offerings:
# hpc7a is in us-east-2b (and others); hpc8a is in us-east-2b / eu-north-1 / ap-northeast-1;
# hpc6a is in us-east-2 (b) / us-west-2 / eu-west-1 etc. AZ MUST contain the type.
AWS_AZ=us-east-2b

aws cloudformation create-stack \
  --stack-name pcs-hpc-efa \
  --region us-east-2 \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=$AWS_AZ \
    ParameterKey=DeployOnDemandCNG,ParameterValue=true \
    ParameterKey=OnDemandInstanceType,ParameterValue=$INSTANCE_TYPE \
    ParameterKey=OnDemandCngName,ParameterValue=hpc \
    ParameterKey=OnDemandQueueName,ParameterValue=hpc \
    ParameterKey=OnDemandMinCount,ParameterValue=0 \
    ParameterKey=OnDemandMaxCount,ParameterValue=2 \
    ParameterKey=OnDemandEnableEfa,ParameterValue=true \
    ParameterKey=OnDemandEfaInterfaceCount,ParameterValue=$EFA_NICS \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

The `OnDemandCNGStack` nested stack auto-creates a cluster placement group and
exposes the name as a stack output (`PlacementGroupName`). To share the PG across
multiple CNGs (heterogeneous tightly-coupled jobs), pass
`OnDemandPlacementGroupName=<existing-pg-name>` instead.

### Step 2 — verify EFA visibility on a compute node

After the stack reaches CREATE_COMPLETE and an `srun` has woken up a node:

```bash
# On a compute node (via Slurm srun from login):
srun -p hpc -N 1 -n 1 bash -c '
  /opt/amazon/efa/bin/fi_info -p efa | head -20  # provider=efa, FI_EP_RDM
  lspci | grep -iE "EFA|Elastic"                 # 2 EFA + 2 ENA on hpc7a/hpc8a
'
```

Expected (hpc7a / hpc8a):
- `fi_info -p efa`: shows `efa-direct` and `efa` fabrics on `rdmap36s0` and
  `rdmap42s0` (or matching device names).
- `lspci`: 2 lines `Elastic Fabric Adapter` (or `Device efa3` on hpc8a) + 2 lines
  `Elastic Network Adapter`.

### Step 3 — OSU MPI micro-benchmarks

Build OSU 7.4 on `/fsx` once (shared across compute nodes) using the
PCS-Ready DLAMI's `/opt/amazon/openmpi`:

```bash
mkdir -p /fsx/osu && cd /fsx/osu
curl -fL -o osu.tgz https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.4.tar.gz
tar xf osu.tgz && cd osu-micro-benchmarks-7.4
PATH=/opt/amazon/openmpi/bin:$PATH ./configure CC=mpicc CXX=mpicxx --prefix=/fsx/osu
PATH=/opt/amazon/openmpi/bin:$PATH make -j8
```

Submit a 2-node sbatch with the AWS-tuned EFA env (the canonical reference
parameters, see "Tuning notes" in [Verified configurations](#hpc-efa-on-cpu-instances-ondemandenableefatrue)):

```bash
cat > /fsx/osu/osu-bench.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=osu-efa
#SBATCH --partition=hpc
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --output=/fsx/osu/logs/%x_%j.out

set -ex
OSU=/fsx/osu/osu-micro-benchmarks-7.4
export PATH=/opt/amazon/openmpi/bin:$PATH

export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1                # huge page stays at the default (=1, on)
export OMPI_MCA_pml=cm                    # AWS Open MPI 4.1.7 has no pml=ofi
export OMPI_MCA_mtl=ofi
export OMPI_MCA_mtl_ofi_provider_include=efa
export OMPI_MCA_btl=^openib,tcp

MPI_X="-x FI_PROVIDER -x FI_EFA_FORK_SAFE \
       -x OMPI_MCA_pml -x OMPI_MCA_mtl -x OMPI_MCA_mtl_ofi_provider_include \
       -x OMPI_MCA_btl"

mpirun -np 2  -N 1  $MPI_X $OSU/c/mpi/pt2pt/standard/osu_latency
mpirun -np 2  -N 1  $MPI_X $OSU/c/mpi/pt2pt/standard/osu_bw
mpirun -np 2  -N 1  $MPI_X $OSU/c/mpi/pt2pt/standard/osu_bibw
mpirun -np 32 -N 16 $MPI_X $OSU/c/mpi/pt2pt/standard/osu_mbw_mr
mpirun -np 32 -N 16 $MPI_X $OSU/c/mpi/collective/blocking/osu_allreduce
EOF

mkdir -p /fsx/osu/logs
sbatch -p hpc /fsx/osu/osu-bench.sbatch
```

The reference numbers per instance type are in
[Verified configurations](#hpc-efa-on-cpu-instances-ondemandenableefatrue) above.

### Step 4 — observe NIC-level traffic in Grafana (optional)

The monitoring stack's Compute Node Details dashboard has dedicated EFA panels
sourced from the `node_amazonefa_*` metrics produced by the v2.7+
`efa-metrics.sh` textfile collector:

- RDMA Read / Write Throughput
- SRD Retransmitted Packets
- Work-Request Errors

For per-NIC `tx_bytes` / `rx_bytes` rate during a benchmark, query Prometheus
directly:

```promql
rate(node_amazonefa_tx_bytes[30s]) * 8 / 1e9   # Gbps per (instance, device)
sum by (instance) (rate(node_amazonefa_tx_bytes[30s])) * 8 / 1e9  # both NICs
```

(The textfile collector cadence is 30s; rates over windows shorter than that are
zeros most of the time. OSU sub-tests are also short — 10–30s each — so wall-clock
peak in Prometheus typically reads below the OSU-reported peak.)

### Step 5 — clean up

When done with EFA testing, just delete the CNG stack (or the whole deploy-all
stack). The auto-created cluster placement group is owned by the CNG stack and
is removed automatically. Slurm puts the EFA compute nodes to sleep on idle
(`SuspendTime`), so leaving the cluster up between benchmarks does not keep the
hpc7a/hpc8a instances running.

---

## Test 10: FSx storage health and performance

Run this test whenever Lustre-related template changes are made (mount options,
`lctl` tunables, stripe configuration, `FSxLustreEnableEfa`, Lustre version,
or any UserData change that touches `/fsx`). The test has two parts:

- **Part A — Health check**: confirms the filesystem mounts, is usable, and
  matches what CFN asked for. Run after every deploy; fast (~2 min).
- **Part B — Performance baseline + regression/improvement test**: measures
  storage throughput under controlled conditions. Run before and after any
  Lustre performance-related change to detect regressions or confirm
  improvements.

---

### Part A — Health check

#### A1. Filesystems mounted on every node

On the login node and at least one compute node (via `srun`):

```bash
mount | grep -E ' /home | /fsx '
df -h /home /fsx
cat /proc/mounts | grep lustre   # confirm mount options (noatime, flock, lazystatfs)
```

Expected:
- `/fsx` mounted as type `lustre`, options include `noatime`, `flock`, `lazystatfs`
- `/home` mounted as type `nfs`, options include `nconnect=16,rsize=1048576,wsize=1048576`
- Both `df -h` show `Avail` > 0

Troubleshooting: if a mount is missing, check `/var/log/cloud-init-output.log`.

#### A2. Read/write sanity

```bash
# /fsx (Lustre)
dd if=/dev/zero of=/fsx/.healthcheck bs=1M count=1024 conv=fsync 2>&1 | tail -1
dd if=/fsx/.healthcheck of=/dev/null bs=1M 2>&1 | tail -1
rm /fsx/.healthcheck

# /home (OpenZFS)
dd if=/dev/zero of=/home/ubuntu/.healthcheck bs=1M count=100 conv=fsync 2>&1 | tail -1
dd if=/home/ubuntu/.healthcheck of=/dev/null bs=1M 2>&1 | tail -1
rm /home/ubuntu/.healthcheck
```

#### A3. FSx-side parameters match CFN inputs

```bash
FSX_ID=$(aws cloudformation describe-stacks --stack-name <stack> \
  --query 'Stacks[0].Outputs[?OutputKey==`FSxLustreFilesystemId`].OutputValue' \
  --region <region> --output text)
aws fsx describe-file-systems --file-system-ids "$FSX_ID" --region <region> \
  --query 'FileSystems[0].[StorageCapacity,StorageType,LustreConfiguration.[DeploymentType,PerUnitStorageThroughput,DataCompressionType,EfaEnabled,MetadataConfiguration.Mode]]' \
  --output text
```

Expected defaults: `1200 | SSD | PERSISTENT_2 | 250 | LZ4 | False | AUTOMATIC`

When `FSxLustreEnableEfa=true`: `EfaEnabled = True`, `Capacity >= 19200` (at
PerUnitStorageThroughput=250). A CFN Rule on both the prerequisites and
deploy-all templates fails the stack at create time when combined with
PERSISTENT_1.

#### A4. Storage dashboard in Grafana

Open Grafana → Storage dashboard. Verify `/fsx` panels (Throughput, IOPS,
Free Capacity) populate within ~5 min of a workload. The `dd` from A2 is
enough to seed values.

---

### Part B — Performance baseline and regression test

Use this procedure to:
1. Record a **baseline** (before a change)
2. Apply the change (mount option, lctl tunable, stripe config, etc.)
3. Record the **after** measurement
4. Compare — confirm no regression and quantify improvement

#### B1. Preparation (run once per test cluster)

```bash
# 10K small files for metadata testing (stat / readdir)
STAT_DIR=/fsx/perf-bench/stat-10k
mkdir -p $STAT_DIR
seq 1 10000 | xargs -P 64 -I{} touch $STAT_DIR/file_{}

# 10K × 4KB files for smallfile read testing
SF_DIR=/fsx/perf-bench/smallfile-10k
mkdir -p $SF_DIR
seq 1 10000 | xargs -P 64 -I{} dd if=/dev/urandom of=$SF_DIR/file_{} bs=4096 count=1 2>/dev/null

# 4GB sequential file for throughput testing
dd if=/dev/zero of=/fsx/perf-bench/seq-4g bs=1M count=4096 oflag=direct
```

#### B2. Single-node benchmarks (login node)

```bash
RESULTS=/fsx/perf-bench/results-$(date +%Y%m%d-%H%M)-${LABEL:-baseline}
mkdir -p $RESULTS

# (1) df latency — measures statfs() performance
for i in $(seq 1 100); do
  ts_s=$(date +%s%N); df /fsx >/dev/null; ts_e=$(date +%s%N)
  echo $(( (ts_e - ts_s) / 1000000 ))
done > $RESULTS/df_latency_ms.txt
echo "df median: $(sort -n $RESULTS/df_latency_ms.txt | awk 'NR==50{print}') ms"

# (2) stat 10K files — measures metadata read (MDS) throughput
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
ts_s=$(date +%s%N)
find /fsx/perf-bench/stat-10k -type f | xargs stat -c '%s' >/dev/null
ts_e=$(date +%s%N)
echo "stat 10K: $(( (ts_e - ts_s) / 1000000 )) ms" | tee $RESULTS/stat_10k.txt

# (3) smallfile 10K × 4KB read — measures many-file open+read (Python imports, HF cache pattern)
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
ts_s=$(date +%s%N)
find /fsx/perf-bench/smallfile-10k -type f -exec cat {} + >/dev/null
ts_e=$(date +%s%N)
echo "smallfile 10K read: $(( (ts_e - ts_s) / 1000000 )) ms" | tee $RESULTS/smallfile_read.txt

# (4) Sequential read 4GB — measures bulk data throughput (OST bandwidth)
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
dd if=/fsx/perf-bench/seq-4g of=/dev/null bs=1M 2>&1 | tee $RESULTS/dd_read.txt

# (5) Sequential write 4GB
dd if=/dev/zero of=/fsx/perf-bench/seq-write-4g bs=1M count=4096 conv=fsync 2>&1 | tee $RESULTS/dd_write.txt
rm -f /fsx/perf-bench/seq-write-4g
```

#### B3. Multi-node benchmarks (compute nodes, via srun)

These are the most sensitive tests for detecting regressions under concurrent
load — the typical ML training scenario.

```bash
# Prerequisite: at least 4 compute nodes available
# sinfo -N should show 4 nodes idle or idle~

# (6) Multi-node stat — N nodes × M procs concurrent stat of 10K files
#     This is the primary regression indicator for metadata changes.
srun -N 4 --ntasks-per-node=1 -p cpu1 bash -c \
  'echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null'
sleep 3
ts_s=$(date +%s%N)
srun -N 4 --ntasks-per-node=4 -p cpu1 bash -c \
  "find /fsx/perf-bench/stat-10k -type f | xargs stat -c '%s' >/dev/null"
ts_e=$(date +%s%N)
echo "multi-node stat 16p: $(( (ts_e - ts_s) / 1000000 )) ms" | tee $RESULTS/multi_stat.txt

# (7) Multi-node sequential read — all nodes read the same 4GB file
srun -N 4 --ntasks-per-node=1 -p cpu1 bash -c \
  'echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null'
sleep 3
ts_s=$(date +%s%N)
srun -N 4 --ntasks-per-node=1 -p cpu1 bash -c \
  "dd if=/fsx/perf-bench/seq-4g of=/dev/null bs=1M 2>/dev/null"
ts_e=$(date +%s%N)
echo "multi-node read 4N: $(( (ts_e - ts_s) / 1000000 )) ms" | tee $RESULTS/multi_read.txt

# (8) flock correctness — concurrent lock serialization
LOCK=/fsx/perf-bench/locktest
rm -f $LOCK
(flock -x 200; echo "lock1 $(date +%s%N)"; sleep 0.1; echo "unlock1 $(date +%s%N)") 200>$LOCK &
(sleep 0.01; flock -x 200; echo "lock2 $(date +%s%N)"; sleep 0.1) 200>$LOCK &
wait
echo "flock: PASS" | tee $RESULTS/flock.txt
```

#### B4. A/B comparison for mount options or lctl changes

To compare two configurations (e.g. `noatime` vs `relatime`), use
`mount -o remount` to switch without a full redeploy:

```bash
# Switch ALL nodes (login + compute) to config A
sudo mount -o remount,relatime /fsx
srun -N 4 --ntasks-per-node=1 -p cpu1 bash -c 'sudo mount -o remount,relatime /fsx'
# Run B2 + B3 with LABEL=before

# Switch ALL nodes to config B
sudo mount -o remount,noatime /fsx
srun -N 4 --ntasks-per-node=1 -p cpu1 bash -c 'sudo mount -o remount,noatime /fsx'
# Run B2 + B3 with LABEL=after
```

For `lctl` changes: apply the tunable, drop caches, re-run. No remount needed.

#### B5. Interpreting results

| Metric | What it measures | Sensitive to |
|---|---|---|
| df latency | `statfs()` → all OSTs | `lazystatfs` mount option; OST count |
| stat 10K | MDS metadata read (stat RPCs) | `noatime`; `mdc.*.max_rpcs_in_flight`; `statahead_max` |
| smallfile 10K read | open + read + close × many files | `noatime`; `mdc.*.max_rpcs_in_flight`; read-ahead |
| dd seq read | Single-stream bulk data throughput | `osc.*.max_rpcs_in_flight`; `max_pages_per_rpc`; stripe; provisioned throughput |
| dd seq write | Single-stream write throughput | `osc.*.max_dirty_mb`; `max_rpcs_in_flight`; stripe |
| multi-node stat | Concurrent MDS load under contention | `noatime` (scales with node count); `mdc.*` tunables |
| multi-node read | Aggregate read bandwidth | Provisioned throughput; node count × per-client BW |
| flock | Locking correctness | `flock` mount option |

**Regression criteria**: a >10% degradation on any metric (excluding `df`
latency which is <5 ms and subject to jitter) should block the change until
investigated.

**Expected magnitudes for common changes**:
- `noatime` addition: stat -0% to -5% single-node; -4% to -30% multi-node
  (scales with concurrency)
- `mdc.*.max_rpcs_in_flight` 8→64: stat -30% to -60%; smallfile -20% to -50%
- `osc.*.max_rpcs_in_flight` 8→64: dd read +50% to +200% (if provisioned
  throughput allows)
- `lfs setstripe -c -1 -S 16M`: dd read/write +100% to +400% (if OSTs > 2)

---

### Baseline results (2026-06-11, us-east-2, 1.2 TiB PERSISTENT_2 / 2 OST, c6i.4xlarge)

#### Single-node

| Metric | Value | Notes |
|---|---|---|
| df latency (median) | 1 ms | lazystatfs server-default |
| stat 10K files | 1851 ms | 5,400 stat ops/sec |
| smallfile 10K × 4KB read | 9374 ms | 1,067 files/sec |
| dd sequential read 4GB | 620 MB/s | Near provisioned limit (1.2 TiB × 250 MB/s/TiB ÷ 1024 ≈ 293 MB/s theoretical baseline; burst to 620 due to FSx credit system) |
| dd sequential write 4GB | 613 MB/s | |

#### Multi-node (4 × c6i.4xlarge)

| Metric | relatime | noatime | Delta |
|---|---|---|---|
| 16-stream stat 10K files | 5033 ms | 4812 ms | **-4.4%** |

#### Interpretation

On a small filesystem (2 OSTs, MDS far from saturation), single-node deltas
are in the noise. Multi-node shows the beginning of MDS contention relief from
`noatime`. At production scale (64+ nodes, 10+ OSTs), improvements from
`noatime` + `mdc` tunables are expected to be 10–30× larger.

These numbers serve as the **regression baseline** for this filesystem size.
When running the same tests on a larger filesystem (e.g. 19200 GiB / ~16 OSTs
for EFA testing), record a new baseline — absolute numbers will differ but the
relative before/after comparison remains valid.

---

## Cleanup

Delete the stack from the CloudFormation console (select → **Delete**) or:

```bash
aws cloudformation delete-stack --stack-name <stack-name>
aws cloudformation wait stack-delete-complete --stack-name <stack-name>
```

Nested stacks (and FSx) are deleted automatically — back up FSx data first. A Capacity
Block keeps billing for its full reserved window regardless and is not released by stack
deletion.

## References

- [AWS PCS Documentation](https://docs.aws.amazon.com/pcs/)
- [aws-parallelcluster-monitoring](https://github.com/aws-samples/aws-parallelcluster-monitoring)
- [Slurm OpenMetrics](https://slurm.schedmd.com/rest.html#openmetrics) · [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)
- [AI/ML for AWS PCS Workshop](https://catalog.workshops.aws/ml-on-pcs/)
