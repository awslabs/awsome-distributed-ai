# P5 GPU Training Validation on AWS PCS

End-to-end validation of the P5 (p5.48xlarge, 8×H100) GPU node group on this PCS
cluster, using Enroot/Pyxis for containers. Three stages, smallest-first:

1. **GPU sanity** — `nvidia-smi` inside a Pyxis container on a GPU node.
2. **Multi-node NCCL** — `all_reduce_perf` across 2 nodes (16 GPUs) over EFA, to
   confirm inter-node GPU communication works.
3. **Megatron-LM Llama-2** — a tiny 2-node Llama-2 pretraining run (5 iters) from
   the upstream [megatron-lm/slurm/llama2](https://github.com/awslabs/awsome-distributed-ai/tree/main/3.test_cases/megatron/megatron-lm/slurm)
   sample, to confirm a real training stack runs distributed.

## Cluster context (recorded at authoring time)

| Item | Value |
|------|-------|
| PCS cluster | `pcs_d3g5oj8y3j` (Slurm 25.11) |
| GPU partition | `gpu-p5` — nodes `gpu-p5-[1-2]`, 8×H100 each (16 GPUs) |
| Capacity Block | `cr-0af82458b684f2c57` (us-east-2a, ends 2026-06-01 11:30 UTC) |
| Login node | `m6i.xlarge` (2 vCPU — **do not build big containers here**) |
| Shared storage | `/fsx` (FSx Lustre, ~1.2T) and `/home` (FSx OpenZFS), both cluster-wide |
| Containers | Enroot 3.5.0 + Pyxis (slurm-25.11), Docker 29.5.2 on login node |

All Slurm commands run as the `ubuntu` user. Submit from the login node (SSM
session or SSH).

## Working directory

Use a shared-filesystem working dir so all nodes see the scripts and `.sqsh`
images. Copy this directory to `/fsx`:

```bash
mkdir -p /fsx/validation
cp -r <this-dir>/* /fsx/validation/
cd /fsx/validation
```

---

## Stage 1 — GPU sanity (`nvidia-smi` via Pyxis)

No image build needed; pulls a public CUDA image on the fly.

```bash
sbatch 01-nvidia-smi.sbatch          # 1 node, 8 GPUs
# or fully inline, no script:
srun --partition=gpu-p5 --nodes=1 --ntasks=1 --gres=gpu:8 \
     --container-image=docker://nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**Pass:** `nvidia-smi` lists 8× H100 GPUs. Check `01-*.out`.

---

## Stage 2 — Multi-node NCCL all-reduce (EFA)

Builds the upstream `nccl-tests` container once, imports to a `.sqsh`, then runs
`all_reduce_perf` on 2 nodes × 8 GPUs over EFA.

```bash
# 2a. Build + import the nccl-tests image (run on a GPU node, NOT the login node)
sbatch 02-build-nccl-tests.sbatch     # writes /fsx/validation/nccl-tests.sqsh

# 2b. Run the 2-node all-reduce
sbatch 02-nccl-tests.sbatch           # 2 nodes, 16 GPUs
```

**Pass criteria:**
- Job completes without NCCL errors.
- `NCCL_DEBUG=INFO` log shows the EFA provider in use
  (`NET/OFI Selected provider is efa` or `Using network AWS Libfabric`).
- `all_reduce_perf` busbw climbs into the expected range for H100+EFA at large
  message sizes (tens to ~hundreds of GB/s by 1–16 GB); the key check is that
  cross-node bandwidth is high and the test reports no errors (`#wrong: 0`).

Source: `micro-benchmarks/nccl-tests/` in this repo.

---

## Stage 3 — Megatron-LM Llama-2 (tiny 2-node run)

Adapted from `3.test_cases/megatron/megatron-lm/slurm/llama2`. Differences from
upstream, to keep it small and runnable on 2 nodes:

- Model preset overridden to **llama2-7b** (`TP=1, PP=1`) instead of the default
  70b — fits comfortably on 16 H100s.
- `--train-iters 5`, validation disabled (`--split 100,0,0`) — a smoke test, not
  a real training run.
- W&B logging removed (no external account needed).

### 3a. Build the Megatron container (on a GPU node)

The image is ~20 GB; building on the `m6i.xlarge` login node is very slow. Build
on a GPU node via the batch job below (it runs `docker build` + `enroot import`):

```bash
sbatch 03-build-megatron.sbatch       # writes /fsx/validation/aws-megatron-lm.sqsh
```

> If `docker` is not available on compute nodes, fall back to building on the
> login node (`cd 3.test_cases/megatron/megatron-lm && make`) and accept the
> slower build, or build the `.sqsh` once and reuse it.

### 3b. Tokenizer + data

Llama-2 needs the HF tokenizer (`tokenizer.model`). It is gated — download from
<https://huggingface.co/meta-llama/Llama-2-7b-hf> and place `tokenizer.model`
(and `tokenizer.json`) under `/fsx/validation/llama2/`. Then preprocess a small
sample corpus:

```bash
mkdir -p /fsx/validation/llama2
# copy tokenizer.model / tokenizer.json into /fsx/validation/llama2/
wget -P /fsx/validation/llama2 \
  https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
xz -d /fsx/validation/llama2/oscar-1GB.jsonl.xz

sbatch 03-data-preproc.sbatch         # 1 node; writes my-llama2_text_document.*
```

### 3c. Pretrain (2 nodes, 5 iters)

```bash
sbatch 03-pretrain-llama2.sbatch      # 2 nodes, 16 GPUs, llama2-7b, 5 iters
```

**Pass:** the job logs 5 training iterations with decreasing/known loss values
and prints throughput (`--log-throughput`), no NCCL/CUDA OOM errors. Check
`03-pretrain-*.out`.

---

## Order of operations / dependencies

```
Stage 1 (nvidia-smi)        ── no build, run anytime nodes are up
Stage 2  build → run        ── nccl-tests.sqsh required before 02-nccl-tests
Stage 3  build → tokenizer  ── aws-megatron-lm.sqsh + tokenizer.model required
         → data-preproc     ── my-llama2_text_document required before pretrain
         → pretrain
```

## Cleanup / cost note

The Capacity Block ends **2026-06-01 11:30 UTC**; the 2 p5 nodes bill for the
whole window regardless. Set the `gpu-p5` node group back to Min=0 when done to
release nodes early (capacity remains reserved until block end):

```bash
# scale the CNG stack back down
aws cloudformation update-stack --stack-name pcs-p5-gpu --region us-east-2 \
  --profile claude --use-previous-template \
  --parameters ParameterKey=MinCount,ParameterValue=0 ParameterKey=MaxCount,ParameterValue=2 \
    $(for p in CapacityReservationId ClusterId ClusterName CngName QueueName InstanceType \
       NetworkInterfaceCount SubnetId AmiId ClusterSecurityGroupId IamProfileArn \
       FSxLustreFilesystemId FSxLustreFilesystemMountName FSxOpenZFSFilesystemId \
       DeployMonitoring MonitoringRole PostInstallScriptUrl PostInstallScriptArgs; \
       do printf 'ParameterKey=%s,UsePreviousValue=true ' "$p"; done) \
  --capabilities CAPABILITY_IAM
```
