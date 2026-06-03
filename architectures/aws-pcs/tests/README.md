# AWS PCS — Test & Validation Guide

This directory is a single guide for validating an AWS PCS cluster deployed from the
templates in [`../assets`](../assets). Each test below lists **what to run** and the
**expected result**. The accompanying scripts:

| Script | Stage |
|---|---|
| [`01-nvidia-smi.sbatch`](./01-nvidia-smi.sbatch) | GPU sanity in a Pyxis container |
| [`02-import-nccl-image.sh`](./02-import-nccl-image.sh) | Import the NCCL-tests image (run on the login node) |
| [`02-nccl-tests.sbatch`](./02-nccl-tests.sbatch) | 2-node NCCL `all_reduce_perf` over EFA |
| [`03-fsdp-llama2.sbatch`](./03-fsdp-llama2.sbatch) | 2-node PyTorch FSDP Llama-2 7B smoke test |

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
| **2× p6-b300.48xlarge** (16× B300) | us-west-2 | Capacity Block | ✅ v2.6.5, 16 GPUs in Grafana | **~760 GB/s** busbw (EFA, `found 16 nics`, `#wrong 0`) † | **~195 TFLOPS/GPU, ~75k tok/s** |
| **2× p5.48xlarge** (16× H100) | us-east-2 | Capacity Block | ✅ | **~480 GB/s** (EFA, `found 32 nics`, `#wrong 0`) | ~60 TFLOPS/GPU |
| **Login + CPU (`c6i`)** | us-west-2 / us-east-* | On-Demand | ✅ all targets up | n/a | n/a |
| **Grafana public access** (login-only SG) | us-west-2 | — | ✅ reachable at `https://<login-public-ip>/grafana/` from the allowed CIDR | — | — |

Notes:
- **Deploy path:** all of the above came up from `pcs-ml-cluster-deploy-all.yaml`
  (`BuildAMI=false`, Enroot/Pyxis via `PostInstallScriptUrl`, `DeployMonitoring=true`).
- **Container runtime:** validated via first-boot UserData install (the default); the
  pre-baked-AMI path (`BuildAMI=true`) shares the same `install-enroot-pyxis.sh`.
- **EFA interface count** is derived from the instance type (p5/p5e = 32, p5en = 16,
  p6-b200 = 8, p6-b300 = 16-of-17); see [README GPU compute](../README.md#gpu-compute-p5p6).
- **FSDP loss** stays at ln(vocab) in this smoke test — a known dataloader/vocab quirk of
  the test case, not a cluster issue.
- **† B300 NCCL bandwidth is not yet conclusive.** B300 has 2× the EFA cards of B200
  (16 vs 8) but the 2-node busbw was only ~1.16× higher — a 2-node / 16 GiB all_reduce
  likely doesn't saturate all 16 cards. Re-test with **larger message sizes (up to ~64 GiB)
  and more nodes (4–8+)** before treating ~760 GB/s as B300's peak network bandwidth.

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
type — no parameter to set. Run `01-nvidia-smi.sbatch` (adjust `--partition`):

```bash
sbatch 01-nvidia-smi.sbatch        # 1 node, 8 GPUs
```

**Expected:** `nvidia-smi` lists 8 GPUs of the expected model (H100 / H200 / B200 /
B300). This confirms the multi-NIC launch template booted and Pyxis works on the GPU
node. EFA itself is exercised by Test 6.

---

## Test 6: NCCL multi-node (EFA)

2-node × 8-GPU `all_reduce_perf` over EFA. **Import the image on the login node first**
(it has a 300 GiB root disk and Enroot), then submit the job:

```bash
# On the login node (direct, not a batch job):
bash 02-import-nccl-image.sh                   # writes /fsx/nccl-tests.sqsh

# Submit the 2-node all_reduce (set --partition to your GPU queue):
sbatch 02-nccl-tests.sbatch
```

**Expected** (in `02-nccl-tests_<jobid>.out`):
- EFA is the provider: `NET/OFI Selected provider is efa, fabric is efa-direct (found N nics)`
  (N = EFA interface count: 32 for p5/p5e, 16 for p5en, 8 for p6-b200, 16 for p6-b300).
- Correctness: `# Out of bounds values : 0 OK`, every size `#wrong: 0`.
- `busbw` scales up to the interconnect bandwidth — e.g. **~760 GB/s** peak at 16 GiB on
  2× p6-b300; ~480 GB/s on 2× p5. (Cross-node bandwidth high and no errors is the check.)

---

## Test 7: FSDP sample training

2-node PyTorch FSDP Llama-2 7B smoke test, using the repo's
[`3.test_cases/pytorch/FSDP`](../../../3.test_cases/pytorch/FSDP) case.

**One-time setup on the login node** (venv on shared `/fsx`, repo cloned, HF token):

```bash
cd /fsx
git clone --depth 1 https://github.com/aws-samples/awsome-distributed-ai.git
python3 -m venv /fsx/fsdp-env
source /fsx/fsdp-env/bin/activate
pip install -U pip wheel setuptools
pip install -r /fsx/awsome-distributed-ai/3.test_cases/pytorch/FSDP/src/requirements.txt
export HF_TOKEN=hf_xxx        # gated tokenizer; keep HF_HOME on /fsx (set in the sbatch)
```

Then submit (adjust `--partition`; the script pins `HF_HOME=/fsx/.hf-cache`):

```bash
sbatch 03-fsdp-llama2.sbatch
```

**Expected** (in `03-fsdp-llama2_<jobid>.out`):
- NCCL initializes over EFA (`found N nics`) and training logs 100 steps + a validation
  step, then saves checkpoints (`llama_v2-50steps`, `llama_v2-100steps`).
- Throughput is printed per step, e.g. **~195 TFLOPS/GPU, ~75k tokens/s** on 2× p6-b300
  (~60 TFLOPS on 2× p5/H100). (Loss is constant at ln(vocab) in this smoke test — a
  known dataloader/vocab quirk of the test case, not a cluster problem.)

> **Multi-NIC tip:** keep `NCCL_SOCKET_IFNAME=^docker,lo` (NCCL auto-selects). Do **not**
> pin a single interface on P5/P6 — all NICs share one subnet and pinning one breaks the
> cross-node NCCL bootstrap ring.

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
