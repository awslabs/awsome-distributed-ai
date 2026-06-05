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
| 2 | **Container runtime, both paths** | (a) `BuildAMI=false` + `PostInstallScriptUrl` (first-boot install); (b) **`BuildAMI=true`** (pre-baked AMI via `pcs-ready-dlami-with-enroot-pyxis.yaml`) | The AMI-build path **duplicates** the Enroot/Pyxis logic in its own UserData (a fix to `install-enroot-pyxis.sh` is **not** automatically in the AMI path), and the **AMI is single-Slurm-version** by design — `SlurmVersion` on the DLAMI stack must match the cluster's `SlurmVersion` (the SPANK plugin is ABI-locked, so a 25.11 AMI used by a 25.05 cluster crashes slurmd). Build an AMI and boot a node from it, run a container job. **Any Enroot/Pyxis change MUST also be verified through a BuildAMI=true run, on each Slurm version.** |
| 3 | **First-boot from a clean deploy** | Validate on a **freshly deployed** cluster, not a node you hand-patched | Post-install runs during cloud-init *before* slurmd/profile.d/controller exist; bugs there (e.g. version detection, `set -e` aborts) only show on a clean first boot, not after a live re-run. |
| 4 | **CPU queue** | `DeployOnDemandCNG=true`; Pyxis container job on `cpu1` | Baseline; also the cheapest way to exercise items 1–3 without GPU capacity. |
| 5 | **Each GPU family** (as capacity allows) | `p5`/`p5e`/`p5en`, `p6-b200`, `p6-b300`: nvidia-smi, NCCL all_reduce, FSDP | EFA NIC layout and the dcgm-exporter image differ per family (see notes). |
| 6 | **Monitoring** | 6 login containers up, all Prometheus targets healthy, GPU dashboards populate on every supported GPU family with the default `DcgmExporterImage` (DCGM 4.5.2 by digest) | — |
| 7 | **Template lint** | `aws cloudformation validate-template` on every edited `assets/*.yaml` | Catches structural errors before a deploy round-trip. |

Tests 1–7 below are the per-item how-to. The single-cluster shortcut (one deploy that
covers monitoring + CPU + one GPU family) is fine for iterating, but the matrix above is
the bar for **merge**.

---

## Coverage matrix

What this guide covers, and the template/parameter that exercises it:

| Dimension | Options to test | How |
|---|---|---|
| **Monitoring** | enabled (default) | `DeployMonitoring=true` → [Test 1](#test-1-monitoring-stack) |
| **Container runtime** | (a) first-boot UserData, (b) pre-baked AMI | (a) `BuildAMI=false` + `PostInstallScriptUrl` (default); (b) `BuildAMI=true` → [Test 2](#test-2-enrootpyxis-container-runtime) |
| **CPU nodes** | `c6i`/`c7i` etc. | `DeployOnDemandCNG=true` → [Test 3](#test-3-cpu-queue) |
| **Single-NIC GPU** | `g5`/`g6` | On-Demand CNG with a G-series type → [Test 4](#test-4-g-series-gpu-single-nic) |
| **Multi-NIC GPU** | `p5`/`p5e`/`p5en`, `p6-b200`, `p6-b300` | `DeployPseriesCNG=true` + `PseriesInstanceType` → [Test 5](#test-5-p5p6-gpu-multi-nic) |
| **NCCL / EFA** | 2-node all_reduce | [Test 6](#test-6-nccl-multi-node-efa) |
| **Sample training** | FSDP Llama-2 7B | [Test 7](#test-7-fsdp-sample-training) |

A single `pcs-ml-cluster-deploy-all.yaml` deploy with `DeployMonitoring=true`,
`DeployOnDemandCNG=true`, and `DeployPseriesCNG=true` exercises Tests 1–7 in one cluster.
See [the README](../README.md#5-usage-examples) for deploy commands; set
`PseriesInstanceType` to the GPU family you want to validate.

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

### Stack creation times (measured, deploy-all)

Wall-clock from `aws cloudformation create-stack` to top-level `CREATE_COMPLETE`, on a
warmed account in us-west-2 (no first-time-in-region provisioning). Useful for sizing
how long a deploy round-trip takes during a review cycle.

| Configuration | First-boot path | Time |
|---|---|---|
| 25.05, CPU only | `BuildAMI=false` + `PostInstallScriptUrl` | **~24m** |
| 25.11, CPU only | `BuildAMI=false` + `PostInstallScriptUrl` | **~31m** |
| 25.11, GPU + CPU (Capacity Block) | `BuildAMI=false` + `PostInstallScriptUrl` | **~44m** |
| 25.05/25.11, CPU only | `BuildAMI=true` (Image Builder runs alongside prereqs) | **~32m** |

Notes: Prerequisites (VPC + dual FSx) is the long-pole on every path (~20-25m); the
PostInstall path adds the per-boot Enroot/Pyxis install (~2-3m), and the BuildAMI path
adds Image Builder (~20-25m) but **runs in parallel with prereqs**, so the wall-clock
delta vs PostInstall is much smaller than ImageBuilder's standalone time. GPU adds the
P-series CNG and the GPU node first boot.

Notes:
- **Deploy path:** all of the above came up from `pcs-ml-cluster-deploy-all.yaml`
  (`BuildAMI=false`, Enroot/Pyxis via `PostInstallScriptUrl`, `DeployMonitoring=true`).
- **Container runtime:** validated via first-boot UserData install (the default); the
  pre-baked-AMI path (`BuildAMI=true`) shares the same `install-enroot-pyxis.sh`.
- **`BuildAMI=true` path validated** (us-west-2, CPU-only, `PostInstallScriptUrl=""`):
  ImageBuilder bakes Enroot 3.5.0 + per-version Pyxis into the DLAMI; login/compute nodes
  boot ready, no first-boot install. The AMI is single-Slurm-version on purpose — see
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

## Test 2: Enroot/Pyxis container runtime

Container support can be provided two ways — validate whichever you deployed:

- **(a) First-boot UserData** (`BuildAMI=false` + `PostInstallScriptUrl`, the default).
- **(b) Pre-baked custom AMI** (`BuildAMI=true`, via `pcs-ready-dlami-with-enroot-pyxis.yaml`).

> **Cleanest path for (b).** `PostInstallScriptUrl` defaults to the Enroot/Pyxis installer
> and deploy-all passes it to the node groups regardless of `BuildAMI` (it's a generic
> first-boot hook, not forced empty). When testing path (b), deploy with
> **`BuildAMI=true` AND `PostInstallScriptUrl=""`** so nodes don't re-run the installer at
> boot. Leaving the default is not fatal — the installer is idempotent and skips what's
> already baked into the AMI — but `""` gives the cleanest boot. For path (a) keep
> `BuildAMI=false` and the default `PostInstallScriptUrl`.

```bash
# On any node (login or compute):
which enroot                                                       # /usr/bin/enroot
ls /opt/aws/pcs/scheduler/slurm-*/lib/slurm/spank_pyxis.so         # per-version Pyxis SPANK plugin
cat /etc/aws/pcs/scheduler/slurm-*/plugstack.conf.d/pyxis.conf     # points at the matching .so
tail -1 /var/log/pcs-post-install.log                              # (a) only: "...completed (exit 0)"
```

**Expected:** `enroot` on `PATH`; a `spank_pyxis.so` under the **cluster's** Slurm version
dir, and the plugstack `pyxis.conf` referencing that exact path. For (a), the post-install
log exits 0. The Test 1/6/7 container jobs are the functional proof that Pyxis works.

> **⚠️ Regression-test rule for `scripts/install-enroot-pyxis.sh`.** This script has bitten
> us repeatedly in ways a single 25.11 GPU run does not catch. **Any change to it MUST be
> retested across the full matrix at the top of this guide**, specifically:
> - **All supported Slurm versions** (25.05 **and** 25.11). The Pyxis SPANK plugin is
>   ABI-locked to its Slurm version — a plugin built for the wrong version stops slurmd from
>   starting (`Incompatible Slurm plugin version`). The script builds Pyxis for the version
>   passed in `PCS_SLURM_VERSION` and installs the `.so` to a per-version path; a regression
>   here only shows on the *other* version.
> - **The `BuildAMI=true` path too.** `pcs-ready-dlami-with-enroot-pyxis.yaml` carries its
>   **own copy** of the Enroot/Pyxis steps in its Image Builder UserData — editing
>   `install-enroot-pyxis.sh` does **not** change the AMI path. Build an AMI and boot a node
>   from it, then run a container job.
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
