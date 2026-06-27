# Cosmos 3 on AWS — Physical AI Flywheel sample

A runnable sample for the **Cosmos 3 Physical AI flywheel** on AWS: generate →
post-train → eval. [NVIDIA Cosmos 3](https://huggingface.co/collections/nvidia/cosmos3)
is a family of **omnimodal world models** for Physical AI, openly licensed under
OpenMDW-1.1 — the same model family is both the synthetic-data generator and the policy
you post-train. This sample post-trains an action policy on **turnkey public LeRobot v3
datasets** (e.g. `lerobot/droid_100`, BridgeData2, LIBERO) using each dataset's native
action vectors, and wraps the whole loop in an **Amazon EKS / SageMaker
HyperPod** deployment (same control plane; HyperPod-specific notes are called out where
they apply).

Every stage has been **validated end-to-end on `p5en.48xlarge` (8× H200) with real
checkpoints**: synthetic data generation (SDG), action-policy post-training, Nano vision
supervised fine-tuning (SFT), Super-64B vision Low-Rank Adaptation (LoRA), and policy-server eval. See the
[Flywheel stages](#flywheel-stages-generate--post-train--eval) table below for the
per-stage manifest, config, and checkpoint detail.

> **Image:** the sample runs on an **AWS Deep Learning Container (DLC)-based image**
> (`Dockerfile`, tagged `:dlc` in Amazon Elastic Container Registry (ECR)), which is
> built with a matched Elastic Fabric Adapter (EFA) / NVIDIA Collective Communications
> Library (NCCL) stack for multi-node training (see [Why the AWS DLC base image](#why-the-aws-dlc-base-image-multi-node-efanccl) below).

## Architecture

The [`cosmos-framework`](https://github.com/NVIDIA/cosmos-framework) is NVIDIA's
training/inference code (what you run), while the **Cosmos 3 models** are the weights it
trains and serves ([`Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano), 16B on a dense 8B Qwen3-VL backbone;
[`Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super), 64B on a 32B backbone; plus task variants like [`Cosmos3-Nano-Policy-DROID`](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID)).
Architecturally, Cosmos 3 is a **Mixture-of-Transformers** omnimodal world model — a
dual-tower design (an autoregressive *reasoner* tower + a diffusion *generator* tower,
joined by dual-stream attention) that jointly processes and generates language, image,
video, and action.

The sample is shown from two views: the **flywheel** is the conceptual loop, and the
**AWS architecture** is how that loop maps onto EKS/HyperPod.

**Flywheel (conceptual loop).** Read it clockwise as a self-reinforcing data engine:
**(1)** ingest and curate real-world Physical AI data (DROID, BridgeData2, AV sensor
logs) into a corpus on S3/FSx → **(2)** the Cosmos3-Super (64B) teacher *generates*
synthetic data (e.g. text-to-image (T2I), image-to-video (I2V), or video-to-video (V2V) via [vLLM-Omni](https://github.com/vllm-project/vllm-omni)) to augment it → **(3)** that synthetic +
real corpus *post-trains* a deployable Cosmos3-Nano (16B) policy (action-policy + vision
SFT, plus Super LoRA adapters) → **(4)** the policy is *evaluated* closed-loop, and its
failures become new generation targets that re-enter the corpus. The per-stage manifest,
config, and checkpoint for each step are in [Flywheel stages](#flywheel-stages-generate--post-train--eval) below.

![Physical AI data flywheel — NVIDIA Cosmos 3 on AWS](diagrams/cosmos3-flywheel.drawio.svg)

**AWS architecture (deployment).** This view shows how the above stages land on
infrastructure. A single **Amazon EKS / SageMaker HyperPod** control plane schedules all
three GPU workloads —
**post-train** (JobSet/PyTorchJob, Fully Sharded Data Parallel v2 (FSDP2) + Ulysses Context Parallelism (CP)), **generation** (vLLM-Omni server, Classifier-Free Guidance (CFG) × Ulysses × Hybrid Sharded Data Parallel (HSDP)), and **eval/serve** (policy-server Deployment) — onto a `p5en.48xlarge`
node pool (8× NVIDIA H200 each, 16 EFA NICs per node). A three-tier storage path backs them — Amazon S3 data
lake → FSx for Lustre hot tier (via an S3→FSx data repository association (DRA)) → local NVMe scratch for the
generation write path — with the [Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1) safety filter, Data Center GPU Manager (DCGM) based GPU
observability (scraped by your own Prometheus/monitoring), and Distributed Checkpoint (DCP) based
resilience spanning all stages. The storage, observability, and resilience details each
have their own section below.

![Cosmos 3 model factory — AWS architecture](diagrams/cosmos3-aws-architecture.drawio.svg)

### Parallelism: training vs. generation

Training and generation are **different engines** with different parallelism
knobs.

| Stage | Engine | Parallelism |
|-------|--------|-------------|
| Post-train | `cosmos-framework` | **FSDP2 + Ulysses CP** (`context_parallel_shard_degree=2` on Super) |
| Generate | `vllm-omni` | **CFG-parallel × Ulysses × HSDP** (`--cfg-parallel-size`, `--ulysses-degree`, `--use-hsdp --hsdp-shard-size`) |

- **CFG (Classifier-Free Guidance)** is a diffusion-*sampling* technique rather than
  a distributed-systems axis: each denoising step runs one conditioned and one
  unconditioned forward pass and blends them to steer toward the prompt. *CFG-parallel*
  places those two passes on different GPUs so that they run concurrently.
 - **Ulysses** is a sequence-parallelism algorithm and is the common term across both
   engines: `cosmos-framework` exposes it as its **context-parallel (CP)** axis
   (`context_parallel_shard_degree`), while vLLM-Omni exposes the same algorithm as
   `--ulysses-degree`. It shards the long video-plus-text token sequence across GPUs,
   using an all-to-all to swap attention heads.
- **HSDP (Hybrid Sharded Data Parallel)** shards the weights within a group and
  replicates them across groups.
- **FSDP2** fully shards each parameter across all ranks, with no replicate dimension.

**Training-side deep dive (when to reach for each).** Cosmos3 trains with FSDP2
(`fully_shard` on a `DeviceMesh`, per-parameter DTensor sharding), not legacy FSDP1.
Two data-parallel knobs live in `[model.parallelism]`: `data_parallel_shard_degree`
(`-1` = shard each param across all ranks → pure FSDP) and
`data_parallel_replicate_degree` (`1` = none). Set replicate > 1 to get **HSDP** (shard
within a group, replicate across groups) — useful when cross-cluster all-gather traffic
becomes the bottleneck at large node counts. **Ulysses CP** is an orthogonal axis to
data-parallel sharding: it is implemented on
DTensor `redistribute` (not Ring Attention), with an all-to-all to shard attention
*heads* inside attention. Cosmos3 needs it because long packed sequences (video latents +
text + action, ~45k tokens) make attention activation memory the limiter. **Constraint:**
because it scatters heads, `cp_degree` must divide the attention head count (and KV heads
under GQA) — the practical ceiling on CP. The **Super (64B)** recipe sets
`context_parallel_shard_degree=2`; **Nano** uses `1` (off).

### Observability

This sample relies on the **NVIDIA GPU-Operator's DCGM exporter** (a prerequisite, not
shipped here) for hardware-saturation signals — **SM-active** (the fraction of time the
GPU's streaming multiprocessors (SM) are executing work), high-bandwidth memory (HBM)
bandwidth, and TensorCore-active (`DCGM_FI_DEV_*`). Scrape the exporter with Prometheus
(or your cluster's existing monitoring); on SageMaker HyperPod the observability add-on
emits the same metrics to Amazon Managed Prometheus.

## What's here

### Action-policy code (`src/cosmos3_aws/action/`)
| File | Purpose |
|------|---------|
| `lerobot_v3_action_dataset.py` | `LeRobotV3ActionDataset` — wraps the **official** `lerobot.datasets.LeRobotDataset` (v3 chunk-loading, video decode, windowing). Works on any conformant **turnkey public** v3 dataset (droid_100, BridgeData2, LIBERO). Native action (no fabricated forward kinematics (FK)). |
| `public_lerobot_sft_dataset.py` | `get_action_public_lerobot_sft_dataset` — factory that wraps `LeRobotV3ActionDataset` in `cosmos-framework`'s `ActionSFTDataset` + `ActionTransformPipeline`, mirroring the in-tree `get_action_droid_sft_dataset`. |
| `action_policy_public_lerobot_experiment.py` | The `action_policy_public_lerobot` experiment (Hydra ConfigStore via import). Mirrors `cosmos-framework`'s `action_policy_droid_nano` recipe (FusedAdam/LambdaLinear, action-head skip-on-load) but feeds the public dataset. |
| `launch_droid_policy.py` | Import-launcher: imports the experiment (firing `cs.store`) then runs the `cosmos-framework` train flow. |
| `*_test.py` | Unit tests: `LeRobotV3ActionDataset` (integration, image-only) + `get_action_public_lerobot_sft_dataset` (monkeypatched, GPU-free). |

### TOMLs (`toml/`) and manifests (`kubernetes/`)
| File | Purpose |
|------|---------|
| `toml/droid_policy_smoke.toml` | Action-policy smoke (10 iters, `ckpt_type=dummy`). |
| `toml/droid_policy.toml` | **Action-policy real** post-train (`ckpt_type=dcp`, Cosmos3-Nano warm-start). |
| `toml/vision_sft_nano.toml` / `vision_sft_super.toml` | **Nano** / **Super** vision SFT (+ LoRA). |
| `kubernetes/train-multi-node-dlc.yaml` | Distributed **action-policy** post-training on the **AWS DLC image** (EFA / Remote Direct Memory Access (RDMA), real Distributed Checkpoint (DCP)). Parameterized via `env_vars`. |
| `kubernetes/train-vision-sft.yaml` | **Vision SFT** (Nano) + Cosmos3-Super LoRA (parameterized `SFT_TOML`+`DATASET_PATH`). |
| `kubernetes/generate-vllm-omni-super.yaml` | **Generation (SDG)** — Super video-to-video (V2V) via the official `vllm/vllm-omni:cosmos3` engine (a separate stack from the training image). |
| `kubernetes/serve-policy.yaml` | **Policy-server eval** (Deployment + Service). |
| `storage/storage-fsx-efa-sc.yaml` | EFA-enabled PERSISTENT_2 FSx for Lustre StorageClass + PersistentVolumeClaim (PVC). Shared by both deployment paths. |
| `storage/storage-fsx-dra.yaml` | **S3→FSx-Lustre Data Repository Association (DRA)** hydration — the recommended data plane (in-region S3 origin → FSx hot tier). Shared by both deployment paths. |
| `build-push.sh` | Build the **AWS DLC image** (`Dockerfile`) and push to ECR via `docker buildx` (see **Deployment**). |

> **Training and inference take different CLIs — don't conflate them.** Training
> (`scripts/train.py`, the SFT experiments) takes `--sft-toml <file>` plus **Hydra-style
> dotted overrides** after `--` (e.g. `trainer.max_iter=10 ckpt_type=dcp`).
> Inference/generation (`scripts/inference.py`) is **tyro** with **flat flags**
> (`--checkpoint-path`, `-o/--output-dir`, `--benchmark`) — not the Hydra dotted form.
> For generation inputs, use the canonical `inputs/omni/*.json` from `cosmos-framework`
> (carrying `"model_mode": "image2video"` etc.); the `inference/defaults/<task>/sample_args.json`
> files are *overlays*, not standalone inputs.

### Why JobSet for multi-node training

The `kubernetes/` training manifests launch the multi-node gang as a
[**JobSet**](https://jobset.sigs.k8s.io/) (`jobset.x-k8s.io`), the Kubernetes
SIG-Batch (Special Interest Group) API purpose-built as a unified primitive for distributed ML training and
HPC workloads. We chose it over a raw `batch/v1` Job because its `failurePolicy:
{restartStrategy: Recreate}` gives clean gang-restart semantics (all ranks recreated
together so NCCL re-forms), and over a framework-specific operator Custom Resource Definition (CRD) because it
needs no Kubeflow training operator on the cluster — it builds directly on the
upstream Job API. The API is
currently `v1alpha2`, so install the JobSet controller on your cluster before applying
these manifests. The HyperPod variants under `hyperpod-eks/` instead use the Kubeflow
`PyTorchJob` that the HyperPod Helm chart provides, because the managed auto-resume
annotations attach to that CRD.

### Why a wrapper instead of the in-tree `DROIDLeRobotDataset`

The `cosmos-framework` in-tree `DROIDLeRobotDataset` (and its `get_action_droid_sft_dataset`
factory) reads native joint actions, but it expects a **prepared DROID v3 "success
split"**: the LeRobot v2→v3 conversion and success-filtering are run out-of-band and
that dataset is not publicly released, and the class hard-codes NVIDIA's DROID column
and camera names. A user who only has turnkey public Hub datasets cannot run it as-is.
`LeRobotV3ActionDataset` closes that gap: it wraps the official `lerobot` library, so
any conformant public v3 dataset (`lerobot/droid_100`, BridgeData2, LIBERO) works
directly with its **native** action vector — no out-of-band prep, no hard-coded
columns. The model's domain-aware action projection + `max_action_dim` padding handle
heterogeneous action widths. The
`get_action_public_lerobot_sft_dataset` factory then feeds it through `cosmos-framework`'s
own `ActionSFTDataset` + `ActionTransformPipeline`, so the per-sample transform and the
training recipe are otherwise identical to the in-tree path.

**Vision SFT datasets** are video+caption: the experiment reads
`${DATASET_PATH}/train/video_dataset_file.jsonl` (each row = a clip path + caption),
so point `DATASET_PATH` at an `sft_dataset_*` directory. Note that `cosmos-framework` uses a
**packing dataloader with a fixed per-rank token budget** (`max_num_tokens_after_packing`),
which means adding GPUs scales *global* tokens/iter roughly linearly — so per-step time,
not iterations/hour, is the per-GPU throughput signal when reasoning about scale-out.

## Prerequisites

**Hardware:** `p5en.48xlarge` nodes (8× NVIDIA H200 each, 16 EFA NICs per node). The
training manifests are multi-node-capable; node count is set via `NUM_NODES` (default
1). **FSx for Lustre** with room for the model checkpoints — a weights-only DCP
checkpoint is roughly 2 bytes/param (≈32 GB for Nano 16B, ≈128 GB for Super 64B), and
full training-state checkpoints that include optimizer state are larger.

You need an **Amazon EKS or SageMaker HyperPod (EKS)** cluster that can schedule those
nodes, with: the **NVIDIA GPU Operator** and **EFA device plugin** (so pods can request
`nvidia.com/gpu` and `vpc.amazonaws.com/efa`); **FSx for Lustre** mounted via a
Persistent Volume Claim (PVC), created by the manifests in [`storage/`](./storage/); the **JobSet controller** (EKS path) or the Kubeflow **PyTorchJob**
operator from the HyperPod Helm chart (HyperPod path); the **AWS DLC image** built and
pushed to your ECR (`build-push.sh`); and a **Hugging Face token** Secret whose account
has accepted the licenses for the gated checkpoints used here (notably
[`nvidia/Cosmos-Guardrail1`](https://huggingface.co/nvidia/Cosmos-Guardrail1)). See [`1.architectures/`](../../../1.architectures/) for
cluster setup. The path-specific prerequisite checklists are in
[`kubernetes/README.md`](kubernetes/README.md) and
[`hyperpod-eks/README.md`](hyperpod-eks/README.md).

## Deployment

Both deployment paths — `kubernetes/` (EKS) and `hyperpod-eks/` (SageMaker
HyperPod) — are parameterized Kubernetes manifests rendered with `envsubst` from
`env_vars` and run on `p5en.48xlarge`. Build and push the image to your ECR, then
render and apply a manifest. Example, action-policy multi-node:

> **Use the [`a8m/envsubst`](https://github.com/a8m/envsubst) variant**, **not** GNU gettext's
> `envsubst`. These manifests embed inline shell, and write in-container runtime
> variables as `$$NAME` (e.g. `$$NODE_RANK`, `$$MASTER`, `$$RANK`,
> `$$MASTER_ADDR`): a8m collapses `$$NAME` → a literal `$NAME` that the pod's bash
> expands at run time, while template vars (`${IMAGE_URI}`, `${NAMESPACE}`, …) are
> substituted from `env_vars`. GNU envsubst does **not** honor the `$$` escape (it
> would eat those runtime vars) and does not support the `${VAR:-default}`
> expansion the manifests use. Install it:
>
> ```bash
> # prebuilt binary (uname picks the right asset), or: go install github.com/a8m/envsubst/cmd/envsubst@latest
> curl -L "https://github.com/a8m/envsubst/releases/download/v1.4.3/envsubst-$(uname -s)-$(uname -m)" \
>   -o /usr/local/bin/envsubst && chmod +x /usr/local/bin/envsubst
> ```

```bash
cp env_vars.example env_vars   # then edit env_vars for your cluster (registry, FSx paths, HF token)
set -a; . ./env_vars; set +a
./build-push.sh                # build the AWS DLC image (Dockerfile) → ECR (docker buildx, linux/amd64)
# the manifests read the HF token from a Secret named hf-token (key: token):
kubectl create secret generic hf-token -n "$NAMESPACE" --from-literal=token="$HF_TOKEN"
# a8m/envsubst: substitutes ${TEMPLATE} vars, leaves $$RUNTIME vars as literal $RUNTIME, then apply:
envsubst < kubernetes/train-multi-node-dlc.yaml | kubectl apply -f -   # 2× p5en, EFA/RDMA
```

> **Smoke vs real (env-driven, no YAML edit):** the defaults run a smoke test
> (`POLICY_TOML=droid_policy_smoke.toml`, `CKPT_TYPE=dummy`, random init). For a real
> warm-start, set `POLICY_TOML=droid_policy.toml` + `CKPT_TYPE=dcp` and stage the
> Cosmos3-Nano DCP checkpoint at `BASE_CHECKPOINT_PATH` (see the warm-start recipe below).

> **Build host:** the AWS DLC image is x86_64-only (linux64 FFmpeg build + amd64-only CUDA
> wheels), so `build-push.sh` targets `linux/amd64`. On an ARM/Apple-Silicon host
> the build runs under slow QEMU emulation — build on an x86_64 host (or an
> in-region EC2 instance near ECR) for faster builds.

### Synthetic data generation (SDG)

Generation runs on a **separate image** — the official `vllm/vllm-omni:cosmos3`
engine, not the DLC training image ([vLLM-Omni](https://github.com/vllm-project/vllm-omni) is a cp312/vLLM-0.23 stack). Mirror it into your
ECR (or pull it directly if your nodes have egress), then bring up the server and
send video-to-video requests. The manifest header carries the authoritative
variable list; the short version:

```bash
set -a; . ./env_vars; set +a
# IMAGE_URI must point at the vllm-omni image (NOT the DLC training image):
export IMAGE_URI=<acct>.dkr.ecr.<region>.amazonaws.com/vllm-omni:cosmos3
envsubst < kubernetes/generate-vllm-omni-super.yaml | kubectl apply -f -

# once the server is Ready, port-forward and POST a V2V request:
kubectl port-forward svc/cosmos3-vllm-omni 8000:8000
curl -F input_reference=@clip.mp4 http://localhost:8000/v1/videos/sync
```

Parallelism flags (`CFG_PARALLEL × ULYSSES`) must multiply to `GPUS_PER_NODE`.
Guardrails are toggled **per request** (`extra_params.guardrails`), not via a
server flag. Delete the Job when generation is done. (HyperPod-EKS variant:
`hyperpod-eks/generate-vllm-omni-super.yaml`, same flow + node-health scheduling.)

> The generation and policy-serving paths pull **`nvidia/Cosmos-Guardrail1`** at
> startup — a **gated (auto-approval) HF repo**. Accept its license once on the account
> behind your HF token, or startup fails on the download.

## Flywheel stages (generate → post-train → eval)

The table below lists the five stages, with the manifest, configuration, and checkpoint
that each one uses.

| Stage | Manifest / TOML | Checkpoint | Notes |
|------|------|------|--------|
| **Generation (SDG)** | `kubernetes/generate-vllm-omni-super.yaml` | `Cosmos3-Super` (V2V) | Super V2V via the official `vllm/vllm-omni:cosmos3` engine; guardrails per-request (`extra_params.guardrails`). |
| **Action-policy** (post-train) | `toml/droid_policy.toml` + `kubernetes/train-multi-node-dlc.yaml` | warm-start `Cosmos3-Nano` (HF→DCP) | `strict_resume=False`; action heads initialize fresh while the rest warm-starts. |
| **Vision SFT** (Nano) | `toml/vision_sft_nano.toml` + `kubernetes/train-vision-sft.yaml` | warm-start `Cosmos3-Nano` (HF→DCP) | Reads a video+caption dataset (e.g. public `nvidia/BridgeData2-Subset-Synthetic-Captions`) at `${DATASET_PATH}/train/video_dataset_file.jsonl`. |
| **Vision LoRA** (Super 64B) | `toml/vision_sft_super.toml` + `kubernetes/train-vision-sft.yaml` | warm-start `Cosmos3-Super` (HF→DCP) | CP=2; LoRA-only (base frozen). Reuses the vision-SFT dataset. |
| **Policy-server eval** | `kubernetes/serve-policy.yaml` | `Cosmos3-Nano-Policy-DROID` | Serves on `:8000` (`GET /info`, `POST /predict`). Requires (1) the `Cosmos-Guardrail1` HF license accepted on the token's account, and (2) the checkpoint as a **local dir** (a bare `org/repo` HF id is not resolved — pre-download to FSx). |

### Storage & checkpoint I/O

The storage path shipped in this sample is **FSx for Lustre with EFA**
(`storage/storage-fsx-efa-sc.yaml`) fronting an in-region S3 origin via a Data Repository
Association (`storage/storage-fsx-dra.yaml`) — a shared, hydrate-on-demand hot tier across
nodes. The data path is read-heavy during training and the checkpoint path uses
**Distributed Checkpoint (DCP)**; both run on FSx in the shipped manifests. (Benchmark
the read-path backends and checkpoint cadence on your own cluster for your dataset and
failure profile.)

#### Real action-policy warm-start recipe (HF → DCP)
The `cosmos-framework` `checkpoint.load_path` consumes **DCP**, but the `Cosmos3-Nano` Hugging Face (HF)
repo ships Diffusers/safetensors. Convert once (CPU job is fine):
```
python -m cosmos_framework.scripts.convert_model_to_dcp --checkpoint-path Cosmos3-Nano -o $BASE_CHECKPOINT_PATH
```
then run with `ckpt_type=dcp` and `checkpoint.load_path=$BASE_CHECKPOINT_PATH`
(the loader appends `/model`). The `action_policy_public_lerobot` experiment sets
`strict_resume=False` and skips the action heads on load, so they initialize fresh
while the rest warm-starts. Note that the DCP **asynchronous** checkpoint save pins roughly the model
size in CPU shared memory, so set the pod's `/dev/shm` `sizeLimit` generously (256Gi is
used here, and p5en has about 2 TiB of RAM).

### Endurance & fault recovery

The shipped training manifests (action-policy and vision SFT) are built for long runs:
they write periodic **asynchronous DCP checkpoints** and recover from worker failures
without manual intervention. The relevant mechanics:

| Aspect | How |
|------|------|
| **Fault recovery** | The JobSet uses `failurePolicy: {maxRestarts: N, restartStrategy: Recreate}`. On a worker failure all pods are recreated and `cosmos-framework` auto-resumes from the latest DCP checkpoint (`latest_checkpoint.txt`, written by the async background process once a checkpoint fully completes). |
| **GPU saturation** | The NVIDIA GPU-Operator DCGM exporter surfaces SM-active / HBM bandwidth / TensorCore-active (see [Observability](#observability)); on SageMaker HyperPod the observability add-on emits the same metrics to Amazon Managed Prometheus (see `hyperpod-eks/`). |
| **MFU (optional)** | `cosmos-framework`'s `MFUCallback` emits MFU only through Weights & Biases (W&B); the shipped configs set `job.wandb_mode=disabled`. A hosted W&B account is not required — `job.wandb_mode=offline` logs MFU to a local `wandb/` datastore. This sample does not publish an MFU figure; if you capture it, cross-check against DCGM SM-active. |

Recovery has two phases: **pod reschedule** and **DCP reload + catch-up**. On
SageMaker HyperPod, node auto-replacement + job auto-resume target the **reschedule**
phase; the **reload + catch-up** phase is handled by `cosmos-framework` and behaves the same across
EKS and HyperPod. Size your checkpoint interval to your failure tolerance (the standard
Young/Daly tradeoff between checkpoint overhead and re-done work on restart).

### Why the AWS DLC base image (multi-node EFA/NCCL)

Multi-node training needs the container's `aws-ofi-nccl` plugin to be Application Binary Interface (ABI) compatible
with the NCCL that `cosmos-framework`'s torch wheel ships. If they are mismatched, cross-node
NCCL-over-EFA fails with `aws-ofi-nccl initialization failed` /
`fi_getinfo() No data available` **even when EFA itself is fully functional**
(`uverbs0-15` present, `ulimit -l unlimited`, `fi_info -p efa` → `FI_PROTO_EFA`) — the
failure is the plugin/NCCL version matrix, not the fabric.

This is why the sample builds on the **AWS Deep Learning Container** rather than an NGC
PyTorch base. The DLC ships **torch 2.10.0+cu130** — the exact wheel `cosmos-framework` pins,
carrying **NCCL 2.28.9** — together with an AWS-tuned, version-**matched** EFA stack
(EFA 1.47.0 → libfabric 2.4 + aws-ofi-nccl 1.18.0). Because the DLC's `aws-ofi-nccl` is
built against that same NCCL, there is no ABI mismatch and **no plugin rebuild or
`NCCL_NET_PLUGIN` workaround is needed**. (An NGC base validates single-node but hits the
mismatch on multi-node; its only single-node escape, `NCCL_NET_PLUGIN=none`, is NVLink-only
and forces TCP across nodes — not a multi-node answer.)

On the DLC image, a multi-node run logs NCCL loading the matched plugin and selecting
EFA with GPUDirect RDMA — for example:
```
NET/OFI Initializing aws-ofi-nccl 1.18.0 ... Using Libfabric version 2.4
NET/OFI Selected provider is efa, fabric is efa-direct (found 16 nics)
NET/OFI Using transport protocol RDMA
Connected all rings, use ring PXN 0 GDR 1   # GPUDirect RDMA over EFA
```
The training manifest's diagnostic preamble prints these EFA/NCCL/plugin lines before
training starts, so a run yields both the transport proof (`NET/OFI` = EFA, not
`NET/Socket` = TCP) and the result in one place. The base-image choice is load-bearing —
verify the transport in your own logs rather than assuming it.

#### Video-decode dependencies in the image (DROID camera MP4s)

torchcodec decodes the DROID camera MP4s, and two build-time packaging requirements
follow from running it on the AWS DLC base. Both are handled in `Dockerfile`:
1. **FFmpeg version.** Ubuntu 22.04 apt ships FFmpeg 4.4 (`libavutil.so.56`), but
   torchcodec 0.10.0 needs FFmpeg 5–8. The image bakes a prebuilt **shared
   FFmpeg 8** (`libavutil.so.60`) onto `/usr/local/lib` + `ldconfig` — deliberately
   not via `LD_LIBRARY_PATH`, to leave the DLC's EFA stack untouched.
2. **Shared `libpython3.13.so.1.0`.** torchcodec's custom-ops `.so` links it, but the
   AWS DLC builds system Python without `--enable-shared`. `cosmos-framework`'s
   `cosmos-dependencies` wheels are **cp313-only** (an AL2023 cp312 base does not work —
   `flash-attn` has no cp312 wheel). The image therefore builds the venv on
   **uv-managed CPython 3.13**, which bundles a shared `libpython3.13.so.1.0`, then
   exposes it via `ldconfig`.

> Note: the multi-node manifest uses `imagePullPolicy: Always` — the `:dlc` tag is
> mutable and nodes cache layers, so `Always` guarantees a rebuilt image is re-pulled.

## Unit tests

The tests target the **data-pipeline** code — the dataset wrapper and SFT factory,
which hold the sample's custom, unit-testable logic (action-vector handling, windowing,
factory wiring). The rest of `src/` is declarative experiment/launch configuration that
runs end-to-end via the manifests rather than in isolation. Both test files run inside
the container image, since they import `cosmos-framework` (not available in CPU-only CI):

- `public_lerobot_sft_dataset_test.py` — GPU-free and monkeypatched: it stubs the
  `cosmos-framework` classes and verifies the factory wiring without touching real data.
- `lerobot_v3_action_dataset_test.py` — an integration test that exercises the wrapper
  + windowing against a small staged public LeRobot v3 dataset (needs
  `DROID_DATASET_PATH` and FFmpeg).

```bash
export PYTHONPATH=$PWD/src
export DROID_DATASET_PATH=/path/to/droid_lerobot_v3   # or a small lerobot/droid_100 clone
python -m pytest src/cosmos3_aws/action/ -q
```

## Software versions

The multi-node EFA/NCCL story depends on a version-matched stack (see
[Why the AWS DLC base image](#why-the-aws-dlc-base-image-multi-node-efanccl) above and
the `Dockerfile` header). The pinned, validated versions:

| Component | Version |
|-----------|---------|
| AWS DLC base (`DLC_TAG`) | `2.10.0-gpu-py313-cu130-ubuntu22.04-ec2` (PyTorch 2.10.0+cu130, Python 3.13, CUDA 13.0, Ubuntu 22.04) |
| `cosmos-framework` (`COSMOS_FRAMEWORK_REF`) | `90cd348877c37b888942c988b631eb1611bf2950` |
| NCCL | 2.28.9 (from the `cosmos-framework` venv's `nvidia-nccl-cu13`) |
| EFA installer | 1.47.0 (libfabric 2.4, aws-ofi-nccl 1.18.0, gdrcopy 2.5.1 — baked into the DLC) |
| FFmpeg (shared, for torchcodec) | 8.x (pinned BtbN build, SHA-256 verified) |
| torchcodec | 0.10.0 |
| `uv` | 0.10.8 |
| Generation engine | `vllm/vllm-omni:cosmos3` (separate image from the training stack) |

## References

- NVIDIA Cosmos 3 model collection — [huggingface.co/collections/nvidia/cosmos3](https://huggingface.co/collections/nvidia/cosmos3)
- Cosmos 3 technical report — [research.nvidia.com/labs/cosmos-lab/cosmos3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)
- JobSet (`jobset.x-k8s.io`) — [jobset.sigs.k8s.io](https://jobset.sigs.k8s.io/)
- EKS cluster architectures — [`1.architectures/4.amazon-eks`](../../../1.architectures/4.amazon-eks)
- SageMaker HyperPod EKS architectures — [`1.architectures/7.sagemaker-hyperpod-eks`](../../../1.architectures/7.sagemaker-hyperpod-eks)

## Security

See [CONTRIBUTING](https://github.com/awslabs/awsome-distributed-ai/blob/main/CONTRIBUTING.md#security-issue-notifications)
for more information. Credentials (Hugging Face tokens, etc.) flow through
Kubernetes Secrets referenced by `secretKeyRef`, never committed to rendered
YAML; `env_vars` is gitignored.

## License

This project is licensed under the MIT-0 License. See the LICENSE file.
