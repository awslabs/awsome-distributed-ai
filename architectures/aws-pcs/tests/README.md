# AWS PCS — Test & Validation Guide

This directory is a single guide for validating an AWS PCS cluster deployed from the
templates in [`../assets`](../assets). Each test below lists **what to run** and the
**expected result**.

Rather than ship its own copies, this guide reuses the repository's canonical benchmark
and training assets and documents only the **PCS-specific deltas** (queue/partition names,
running the Enroot import on the login node, putting caches on `/fsx`):

| Stage | Canonical asset to use | PCS-specific delta documented here |
|---|---|---|
| NCCL `all_reduce` over EFA | [`micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch`](../../../micro-benchmarks/nccl-tests) | [Test 6](#test-6-nccl-multi-node-efa) — import on the login node, partition name |
| FSDP Llama-2 7B training | [`3.test_cases/pytorch/FSDP`](../../../3.test_cases/pytorch/FSDP) | [Test 7](#test-7-fsdp-sample-training) — venv/cache on `/fsx`, 2 nodes |
| GPU sanity (nvidia-smi) | (one `srun` line, no script) | [Test 5](#test-5-p5p6-gpu-multi-nic) |

All Slurm commands run as the **`ubuntu`** user from the login node (SSM session or
SSH). Slurm binaries are only on `PATH` in a login shell — over SSM/SSH wrap commands
as `bash -lc "sinfo; squeue"`.

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
| **2× p6-b200.48xlarge** (16× B200) | us-west-2 | Capacity Block | ✅ v2.6.5, 16 GPUs in Grafana | **~654 GB/s** (EFA, `found 8 nics`, `#wrong 0`) | **~223 TFLOPS/GPU, ~86k tok/s** |
| **2× p6-b300.48xlarge** (16× B300) | us-west-2 | Capacity Block | ⚠️ stack up, but **GPU metrics empty** ‡ | **~760 GB/s** busbw (EFA, `found 16 nics`, `#wrong 0`) † | **~195 TFLOPS/GPU, ~75k tok/s** |
| **2× p5.48xlarge** (16× H100) | us-east-2 | Capacity Block | ✅ | **~480 GB/s** (EFA, `found 32 nics`, `#wrong 0`) | ~60 TFLOPS/GPU |
| **Login + CPU (`c6i`)** | us-west-2 / us-east-* | On-Demand | ✅ all targets up | n/a | n/a |
| **Grafana public access** (login-only SG) | us-west-2 | — | ✅ reachable at `https://<login-public-ip>/grafana/` from the allowed CIDR | — | — |

Notes:
- **Deploy path:** all of the above came up from `pcs-ml-cluster-deploy-all.yaml`
  (`BuildAMI=false`, Enroot/Pyxis via `PostInstallScriptUrl`, `DeployMonitoring=true`).
- **Container runtime:** validated via first-boot UserData install (the default); the
  pre-baked-AMI path (`BuildAMI=true`) shares the same `install-enroot-pyxis.sh`.
- **`BuildAMI=true` path validated** (us-west-2, CPU-only deploy-all, `PostInstallScriptUrl=""`):
  ImageBuilder produced a custom AMI (`EnrootPyxisInstaller` component); the login/compute
  nodes booted from it with **`enroot 3.5.0` + Pyxis baked in** (post-install log =
  `No post-install script configured` — confirming **no first-boot double-install**), the
  monitoring stack up, and a `cpu1` Pyxis container job (`ubuntu:22.04`) ran clean
  (`pyxis: imported docker image`). **Set `PostInstallScriptUrl=""` when `BuildAMI=true`** to
  avoid re-running the installer at boot — see [README container runtime](../README.md#container-runtime-enrootpyxis).
- **EFA interface count** is derived from the instance type (p5/p5e = 32, p5en = 16,
  p6-b200 = 8, p6-b300 = 16-of-17); see [README GPU compute](../README.md#gpu-compute-p5p6).
- **FSDP loss** stays at ln(vocab) in this smoke test — a known dataloader/vocab quirk of
  the test case, not a cluster issue.
- **† B300 NCCL bandwidth is not yet conclusive.** B300 has 2× the EFA cards of B200
  (16 vs 8) but the 2-node busbw was only ~1.16× higher — a 2-node / 16 GiB all_reduce
  likely doesn't saturate all 16 cards. Re-test with **larger message sizes (up to ~64 GiB)
  and more nodes (4–8+)** before treating ~760 GB/s as B300's peak network bandwidth.
- **‡ B300 GPU metrics don't populate yet.** The monitoring stack (`v2.6.5`) pins
  `dcgm-exporter` to DCGM 4.2.0, which doesn't support B300 (needs ≥ 4.4.0); the pin exists
  because newer NVCR tags can't be pulled on Docker 29.x. The stack and all non-GPU metrics
  come up fine, but the GPU dashboards stay empty on **p6-b300** until the dcgm image can be
  bumped. Tracked upstream:
  [aws-parallelcluster-monitoring#50](https://github.com/aws-samples/aws-parallelcluster-monitoring/issues/50).
  All other GPU types (p5/p5e/p5en/p6-b200) report GPU metrics normally.

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
[README §8 Monitoring](../README.md#8-monitoring). Use `MonitoringVersion=v2.6.5`+ on PCS.

---

## Test 2: Enroot/Pyxis container runtime

Container support can be provided two ways — validate whichever you deployed:

- **(a) First-boot UserData** (`BuildAMI=false` + `PostInstallScriptUrl`, the default).
- **(b) Pre-baked custom AMI** (`BuildAMI=true`, via `pcs-ready-dlami-with-enroot-pyxis.yaml`).

> **Use one, not both.** `PostInstallScriptUrl` defaults to the Enroot/Pyxis installer and
> deploy-all passes it to the node groups regardless of `BuildAMI`. When testing path (b),
> deploy with **`BuildAMI=true` AND `PostInstallScriptUrl=""`** — otherwise every node both
> boots the pre-baked AMI and re-runs the installer at first boot (wasted time / possible
> conflict). For path (a) keep `BuildAMI=false` and the default `PostInstallScriptUrl`.

```bash
# On any node (login or compute):
which enroot                                  # /usr/bin/enroot
ls /usr/local/lib/slurm/spank_pyxis.so        # Pyxis SPANK plugin present
tail -1 /var/log/pcs-post-install.log         # (a) only: "...completed (exit 0)"
```

**Expected:** `enroot` on `PATH` and `spank_pyxis.so` present on every node. For (a),
the post-install log exits 0 (Pyxis plugstack changes need a `slurmd` restart to take
effect, which PCS handles at boot). The Test 1/6/7 container jobs are the functional
proof that Pyxis works.

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
- `busbw` scales up to the interconnect bandwidth — e.g. **~760 GB/s** peak at 16 GiB on
  2× p6-b300; ~480 GB/s on 2× p5. (Cross-node bandwidth high and no errors is the check.)

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

3. **Submit 2 nodes** (the canonical sbatch defaults to 4) on your GPU queue:

   ```bash
   sbatch --nodes=2 --partition=gpu-p6b200 llama2_7b-training.sbatch
   ```

**Expected** (in `logs/llama2_7b-FSDP_<jobid>.out`):
- NCCL initializes over EFA (`found N nics`) and training logs ~100 steps + a validation
  step, saving checkpoints under `./checkpoints`.
- Throughput is printed per step, e.g. **~195 TFLOPS/GPU, ~75k tokens/s** on 2× p6-b300
  (~60 TFLOPS on 2× p5/H100). (Loss is constant at ln(vocab) in this smoke test — a
  known dataloader/vocab quirk of the test case, not a cluster problem.)

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
