<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Continue SFT of DreamZero (14B World-Action Model) on LIBERO on Amazon EKS

DreamZero is a **16.48B-parameter World-Action Model (WAM)** — a Wan-based
video-diffusion Diffusion Transformer (DiT) that *jointly* denoises future video
frames and future robot actions in a shared causal self-attention space. The
model predicts both what will happen (video) and what to do (actions); the video
prediction acts as a computational scaffold for action reasoning.

This walkthrough packages the canonical customer workflow as a set of reusable
Kubernetes manifests that run on Amazon EKS: take the released
[`GEAR-Dreams/DreamZero-DROID`](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)
14B checkpoint (pretrained on DROID, a Franka arm), continue **supervised
fine-tuning (SFT)** on a *new* embodiment's data (LIBERO, `libero_sim`), then
**evaluate the result in the LIBERO simulator** and render in-sim rollout videos.
There is no native LIBERO 14B checkpoint upstream — warm-starting from DROID is
the point.

Upstream projects:
[github.com/RLinf/RLinf](https://github.com/RLinf/RLinf) (training framework) and
[github.com/RLinf/dreamzero](https://github.com/RLinf/dreamzero) (the `groot`
package that provides the WAM model code).

> **Validation scope (read this first).** The pipeline below was validated
> end-to-end on EKS with a **1-step** SFT run. That proves the *infrastructure
> and the pipeline* — image build, multi-node EFA/NCCL, FSDP2 sharded
> checkpointing, DCP→`.pt` conversion, and LIBERO simulator eval — **not** task
> accuracy. A 1-step checkpoint yields `eval/success_once = 0.0`, which is
> expected. Real accuracy requires a multi-step training run (the pipeline
> supports it — raise `runner.max_steps`; see step 4). No success numbers or loss
> curves are fabricated here.

The recipe has six moving parts, each a self-contained manifest in this
directory:

1. A one-shot **model/dataset download Job** (`model-download.yaml`) that stages
   the DreamZero-DROID checkpoint, the `umt5-xxl` tokenizer, and the
   `physical-intelligence/libero` dataset onto a shared FSx for Lustre volume.
2. A **metadata-generation Job** (`generate-metadata.yaml`) that produces the
   `libero_sim` normalization statistics the DROID checkpoint does not ship.
3. A **multi-node SFT RayJob** (`dreamzero-sft.yaml`) that fans FSDP2 training
   across 2× `p5en.48xlarge` (16× H200) via KubeRay.
4. A **checkpoint-conversion Job** (`convert-checkpoint.yaml`) that consolidates
   the sharded FSDP DCP checkpoint into a single `.pt` on CPU.
5. A **LIBERO simulator eval Job** (`dreamzero-eval.yaml`) that drives the sim,
   reports `eval/success_once`, and writes in-sim rollout videos.

All steps share a single FSx for Lustre `PersistentVolumeClaim` named
`fsx-claim`, mounted at `/fsx`, that holds the models, dataset, and checkpoints.

## Architecture

### Infrastructure / training topology

![DreamZero SFT infrastructure](../../diagrams/infra-dreamzero-sft.drawio.svg)

Multi-node SFT runs as a **KubeRay `RayJob`** (run-to-completion) with an embedded
`RayCluster`: one Ray **head** pod and one Ray **worker** pod, each landing on its
own `p5en.48xlarge` node (8× H200 141 GB) → **16 GPUs total**. The KubeRay
operator brings the Ray cluster up, then runs the Ray-agnostic launcher
(`run_dreamzero_sft_eks.sh`) as the head `entrypoint`; RLinf's `Cluster` (Ray)
scheduler fans **FSDP2 `full_shard`** training across all 16 GPUs (the 16.48B
model is *sharded*, not replicated). There is **no** torchrun, DeepSpeed, or
manual head election. Gradient sync flows over **NCCL on EFA RDMA** (libfabric
2.4 / aws-ofi-nccl 1.18, GPUDirect RDMA). Pod anti-affinity guarantees one pod
per physical node; a topology-spread constraint prefers co-location under the
same network layer for lowest NCCL latency. `shutdownAfterJobFinishes: true`
tears the RayCluster down when training ends.

### The World-Action Model

![DreamZero WAM](../../diagrams/dreamzero-wam.drawio.svg)

During training, video frames and actions are each encoded and corrupted with
noise via flow-matching interpolation. The noisy video latent tokens and an
**Action Register** (noisy action tokens + state-encoder output) participate
together in blockwise causal self-attention — this is the "joint" in Joint
Video-Action. UMT5-XXL text embeddings and CLIP image embeddings condition the
DiT via cross-attention. Two output heads predict the velocity field: one for
video (dynamics loss) and one for actions (action loss), weighted equally. The
video loss is not auxiliary — it is the mechanism by which the model learns
physics (gravity, contact, object permanence). At deployment the robot consumes
only the action channel.

| Component | Role |
|-----------|------|
| DiT backbone (14B-class WAM) | Shared denoising over video + action tokens (dim 5120, 40 layers) |
| Action Register | Noisy action tokens + state-encoder output; participates in joint attention |
| UMT5-XXL | Encodes task-instruction text (`google/umt5-xxl` tokenizer) |
| CLIP / image conditioning | Encodes the observation frame for cross-attention conditioning |
| Wan VAE | Compresses/decodes video latents |
| Action Decoder | Projects denoised action tokens to per-embodiment joint positions |

## Hardware requirements

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU nodes | **2× `p5en.48xlarge`** | 8× NVIDIA H200 (141 GB) each = **16 GPUs**. The 16.48B model is FSDP2-sharded across all 16. |
| EFA | **16 EFA NICs per node** | High-bandwidth RDMA for NCCL allreduce. Single-node eval also requests 16. |
| Shared storage | **FSx for Lustre, ≥250 GB free** | A 14B FSDP **DCP checkpoint is ~140–206 GB** (16 shards incl. full optimizer state). A full filesystem truncates `torch.save` mid-write → corrupt, unreadable shards. |
| Eval | 1× `p5en.48xlarge` (8× H200) | Eval pins `cluster.num_nodes=1`; the 14B model is sharded across one node's 8 GPUs with single-node FSDP. |

## Prerequisites

### 1. An EKS cluster that can provision 2× `p5en.48xlarge` with EFA

You need an Amazon EKS cluster with GPU autoscaling (e.g. Karpenter) able to
launch **2× `p5en.48xlarge`** nodes, each with **8× H200** GPUs and **16 EFA
NICs**, plus the NVIDIA GPU Operator (or device plugin) and EFA device plugin so
pods can request `nvidia.com/gpu` and `vpc.amazonaws.com/efa`. Cluster-creation
references live in
[`1.architectures/4.amazon-eks`](../../../../../1.architectures/4.amazon-eks).

Point your local kubeconfig at the cluster and confirm it is reachable:

```bash
aws eks update-kubeconfig --name <EKS_CLUSTER_NAME> --region <AWS_REGION>
kubectl config current-context
```

### 2. KubeRay operator

Multi-node SFT is a `ray.io/v1` `RayJob`, so the KubeRay operator must be
installed:

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator \
  --version 1.6.0 \
  -n kuberay-system --create-namespace
```

Verify the operator is running:

```bash
kubectl get pods -n kuberay-system
```

### 3. FSx for Lustre shared storage (`fsx-claim` at `/fsx`)

Every step mounts a `PersistentVolumeClaim` named **`fsx-claim`** at `/fsx`. If
your cluster already exposes such a PVC (many EKS GPU cluster templates ship one),
confirm it is `Bound` with `ReadWriteMany` access and has ≥250 GB free:

```bash
kubectl get pvc fsx-claim -n "$NAMESPACE"
# NAME        STATUS   VOLUME      CAPACITY   ACCESS MODES   STORAGECLASS  AGE
# fsx-claim   Bound    fsx-pv...   1.2Ti      RWX            fsx-sc        3d
```

If you do **not** already have one, this directory ships two optional manifests
(both require the `fsx.csi.aws.com` CSI driver):

- **Dynamic provisioning** — `storage/pvc-fsx-lustre-dynamic.yaml` creates a
  `StorageClass` + `fsx-claim` PVC and provisions a fresh filesystem. Fill in
  `FSX_SUBNET_ID` and `FSX_SECURITY_GROUP_IDS` for your cluster's VPC:

  ```bash
  export FSX_SUBNET_ID=subnet-0abc... FSX_SECURITY_GROUP_IDS=sg-0def...
  envsubst < storage/pvc-fsx-lustre-dynamic.yaml | kubectl apply -f -
  ```

- **Static binding** — `storage/pv-fsx-lustre-static.yaml` binds `fsx-claim` to
  an existing FSx for Lustre filesystem. Fill in `FSX_FILESYSTEM_ID`,
  `FSX_DNS_NAME`, and `FSX_MOUNT_NAME`:

  ```bash
  export FSX_FILESYSTEM_ID=fs-0... FSX_DNS_NAME=fs-0....fsx.<region>.amazonaws.com FSX_MOUNT_NAME=abcd1234
  envsubst < storage/pv-fsx-lustre-static.yaml | kubectl apply -f -
  ```

### 4. A `training-sa` ServiceAccount with credentials for HF + S3

The SFT, eval, convert, and metadata pods run as the **`training-sa`**
ServiceAccount. Give it IRSA or EKS Pod Identity so pods can reach Hugging Face
and (if you stage to/from S3) Amazon S3. For example, with IRSA via `eksctl`:

```bash
eksctl create iamserviceaccount \
  --name training-sa \
  --namespace "$NAMESPACE" \
  --cluster <EKS_CLUSTER_NAME> \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
  --approve
```

Or attach a Pod Identity association to an existing ServiceAccount:

```bash
kubectl create serviceaccount training-sa -n "$NAMESPACE"
aws eks create-pod-identity-association \
  --cluster-name <EKS_CLUSTER_NAME> \
  --namespace "$NAMESPACE" \
  --service-account training-sa \
  --role-arn arn:aws:iam::<account>:role/<training-sa-role>
```

> **Hugging Face auth is optional.** The default repos
> (`GEAR-Dreams/DreamZero-DROID`, `google/umt5-xxl`,
> `physical-intelligence/libero`) are **public** and download anonymously. Only
> if you must authenticate to a gated repo, create the `hf-token` Secret from
> `secret.example.yaml`:
>
> ```bash
> kubectl -n "$NAMESPACE" create secret generic hf-token --from-literal=HF_TOKEN=hf_xxx
> ```

### 5. Local tooling: `kubectl`, `helm`, and the `a8m/envsubst` variant

Several manifests in this directory embed inline shell scripts inside the YAML
(the RayJob `entrypoint`, the metadata bootstrap, the convert and eval launchers)
and are rendered with a **restricted** `envsubst` allow-list so only
`${ECR_URI}` and `${NAMESPACE}` are substituted while inline shell variables
(`${PYTHONPATH:-}`, `${DREAMZERO_PATH}`, …) are left literal.

Install [a8m/envsubst](https://github.com/a8m/envsubst) — **not** GNU gettext's
`envsubst`, which does not support the same allow-list/escape semantics:

```bash
# macOS / Linux prebuilt binary (uname picks the right asset):
curl -L "https://github.com/a8m/envsubst/releases/download/v1.4.3/envsubst-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/envsubst
chmod +x /usr/local/bin/envsubst

# Or, with Go installed:
go install github.com/a8m/envsubst/cmd/envsubst@latest
```

Verify the binary on your PATH is the a8m one:

```bash
envsubst --version
# expect: "envsubst version: vX.Y.Z" (a8m/envsubst)
# if you see "envsubst (GNU gettext-runtime)", the GNU binary is still on PATH.
```

## Configure

The committed template is `env_vars.example`. Copy it to `env_vars` (gitignored),
edit your values, then `source` it. Every step below assumes you have sourced it:

```bash
cp env_vars.example env_vars
"${EDITOR:-vi}" env_vars
source env_vars
```

The variables:

| Variable | Example | Purpose |
|----------|---------|---------|
| `ECR_URI` | `<account>.dkr.ecr.<region>.amazonaws.com/dreamzero` | Your ECR repository for the DreamZero training image (no tag). |
| `NAMESPACE` | `dreamzero` | Kubernetes namespace to deploy into. |
| `AWS_REGION` | `us-east-1` | Region of your cluster / ECR. |
| `UPSTREAM_REF` | `b3bbabb1f461` | Pinned RLinf commit baked into the image. |
| `DREAMZERO_REF` | `ab790c198fbc` | Pinned `dreamzero` (`groot`) commit baked into the image. |

## Step-by-step

Throughout, `$NAMESPACE` and `$ECR_URI` come from `env_vars`. Wherever a manifest
embeds inline shell, render it with the **restricted** allow-list
`envsubst '${ECR_URI} ${NAMESPACE}'` so the inline `${...}` shell variables are
not clobbered to empty strings.

### 1. Build and push the training image

The image is built in two stages: the upstream RLinf `embodied-libero` target
(which natively builds the dedicated **`dreamzero` venv**, RLinf PR #1272), then
an EFA overlay (`Dockerfile` at the test-case root) that layers EFA 1.47.0 /
libfabric 2.4.0 / aws-ofi-nccl 1.18.0 and the DCP-save gloo-coordinator patch. The
external `groot` package is cloned from `github.com/RLinf/dreamzero.git` and
placed on `PYTHONPATH` via `DREAMZERO_PATH=/workspace/DreamZero`.

Build the image with **local Docker + buildx**. Run from the `setup/` directory;
it reads `ECR_URI`, `AWS_REGION`, `UPSTREAM_REF`, and `DREAMZERO_REF` from
`env_vars`:

```bash
cd setup
source ../env_vars
./build-push.sh
cd ..
```

This clones the pinned `RLinf` and `dreamzero` sources, builds stage 1
(`rlinf-upstream-embodied-libero`), then builds and pushes stage 2 to
`${ECR_URI}:latest`.

### 2. Stage models + dataset to FSx

Downloads the DreamZero-DROID 14B warm-start checkpoint
(`GEAR-Dreams/DreamZero-DROID`), the `google/umt5-xxl` tokenizer, and the
`physical-intelligence/libero` dataset (LeRobot layout) — all **anonymous**. This
Job runs in a lightweight `python:3.11-slim` staging image, so plain `envsubst`
is fine:

```bash
envsubst < model-download.yaml | kubectl apply -f -
kubectl logs -f -n "$NAMESPACE" job/model-download-dreamzero
```

> **The dataset MUST be `physical-intelligence/libero`, NOT `lerobot/libero`.**
> The two repos share a name but have *different* `observation.state` / `action`
> column schemas. Only `physical-intelligence/libero` matches the `libero_sim`
> preset. Using `lerobot/libero` silently trains on the wrong column layout.

Stages to `/fsx/models/DreamZero-DROID`, `/fsx/models/umt5-xxl`, and
`/fsx/datasets/libero` (~152 GB total).

### 3. Generate the `libero_sim` normalization metadata

The DreamZero-DROID checkpoint bundles `experiment_cfg/metadata.json` for
embodiment `oxe_droid` **only**, so LIBERO SFT (`embodiment_tag: libero_sim`)
would fail with `KeyError: embodiment_tag 'libero_sim' not found`. This Job runs
upstream's `toolkits/lerobot/generate_dreamzero_metadata.py` inside the **training
image** (it needs the RLinf toolkit + the `dreamzero` venv), so use **restricted**
`envsubst`:

```bash
envsubst '${ECR_URI} ${NAMESPACE}' < generate-metadata.yaml | kubectl apply -f -
kubectl logs -f -n "$NAMESPACE" job/generate-metadata-dreamzero
# -> writes /fsx/models/metadata-libero.json (top-level key `libero_sim`)
```

The SFT and eval launchers default `METADATA_PATH` to this path, so downstream
steps pick it up automatically.

### 4. Multi-node SFT (2× `p5en.48xlarge`)

Create the launcher ConfigMap from `scripts/run_dreamzero_sft_eks.sh`, then apply
the RayJob with **restricted** `envsubst` (the RayJob `entrypoint` embeds an
inline `bash -c` block that unrestricted substitution would mangle):

```bash
kubectl -n "$NAMESPACE" create configmap dreamzero-sft-launcher \
  --from-file=run_dreamzero_sft_eks.sh=scripts/run_dreamzero_sft_eks.sh \
  --dry-run=client -o yaml | kubectl apply -f -

envsubst '${ECR_URI} ${NAMESPACE}' < dreamzero-sft.yaml | kubectl apply -f -

kubectl logs -f -n "$NAMESPACE" job/dreamzero-sft
```

The launcher drives `examples/sft/train_vla_sft.py` with config
`libero_sft_dreamzero_14b`. SFT writes a **sharded FSDP DCP** checkpoint
(`.distcp` + `.metadata`) under
`.../global_step_<N>/actor/dcp_checkpoint/`.

> **Validated smoke run uses `runner.max_steps=1`.** For a real (multi-step)
> training run, raise `runner.max_steps` (and set a checkpoint `save_interval`)
> via the launcher's `HYDRA_OVERRIDES` env var on the RayJob head/worker
> containers, e.g. `HYDRA_OVERRIDES="runner.max_steps=2000 runner.save_interval=500"`.

### 5. Convert the checkpoint (DCP shards → single `.pt`)

Eval consumes a single consolidated `.pt`. Convert the sharded DCP offline on
**CPU** (do **not** use `save_full_model_weights` — the rank-0 full-state-dict
gather stalls on the 16B model). Create the launcher ConfigMap from
`scripts/convert_checkpoint.sh`, then apply with **restricted** `envsubst`:

```bash
kubectl -n "$NAMESPACE" create configmap dreamzero-convert-launcher \
  --from-file=convert_checkpoint.sh=scripts/convert_checkpoint.sh \
  --dry-run=client -o yaml | kubectl apply -f -

envsubst '${ECR_URI} ${NAMESPACE}' < convert-checkpoint.yaml | kubectl apply -f -
kubectl logs -f -n "$NAMESPACE" job/dreamzero-convert
# -> .../global_step_1/actor/model_state_dict/full_weights.pt
```

The default `STEP=global_step_1` matches the 1-step smoke run; override the
`STEP` env in the manifest for a later checkpoint.

### 6. LIBERO simulator eval (single-node GPU)

Single-pod GPU Job (`cluster.num_nodes=1`, single-node FSDP across 8 H200). Uses
the 14B eval config `scripts/libero_spatial_eval_dreamzero_14b.yaml` (upstream
ships only a 5B eval config). Create **both** ConfigMaps — the launcher *and* the
14B eval config — then apply with **restricted** `envsubst`:

```bash
kubectl -n "$NAMESPACE" create configmap dreamzero-eval-launcher \
  --from-file=run_dreamzero_eval_eks.sh=scripts/run_dreamzero_eval_eks.sh \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NAMESPACE" create configmap dreamzero-eval-config \
  --from-file=libero_spatial_eval_dreamzero_14b.yaml=scripts/libero_spatial_eval_dreamzero_14b.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

envsubst '${ECR_URI} ${NAMESPACE}' < dreamzero-eval.yaml | kubectl apply -f -
kubectl logs -f -n "$NAMESPACE" job/dreamzero-eval
```

The launcher copies the eval config into the embodiment config dir at runtime
(mounting a ConfigMap over the dir would hide upstream config groups). The eval
reports `eval/success_once` and (with `SAVE_VIDEO=True`, the default) writes
in-sim rollout videos to `{LOG_DIR}/video/eval/seed_*/0.mp4`
(`/fsx/checkpoints/dreamzero-libero-eval/video/eval/seed_*/0.mp4`).

> The 14B config uses `total_num_envs=16`; the upstream 5B default of **128 OOMs**
> the 16.48B model co-located with the sim on 8× H200.
>
> **`eval/success_once = 0.0` is EXPECTED for a 1-step checkpoint** — the eval
> validates the machinery, not policy competence.

## File structure

```
kubernetes/libero/
├── README.md                       # This walkthrough
├── env_vars.example                # Copy to env_vars and `source` it
├── secret.example.yaml             # OPTIONAL hf-token Secret (gated repos only)
├── model-download.yaml             # Job: stage DreamZero-DROID + umt5-xxl + libero dataset
├── generate-metadata.yaml          # Job: libero_sim normalization metadata.json
├── dreamzero-sft.yaml              # KubeRay RayJob: 2-node FSDP2 SFT
├── convert-checkpoint.yaml         # Job (CPU): DCP shards -> full_weights.pt
├── dreamzero-eval.yaml             # Job (GPU): LIBERO sim eval + in-sim video
├── scripts/
│   ├── run_dreamzero_sft_eks.sh    # Multi-node SFT launcher (Ray-agnostic, FSDP2)
│   ├── convert_checkpoint.sh       # DCP -> .pt conversion launcher
│   ├── run_dreamzero_eval_eks.sh   # LIBERO simulator eval launcher
│   └── libero_spatial_eval_dreamzero_14b.yaml  # 14B eval config (upstream ships 5B only)
├── setup/
│   └── build-push.sh               # Local buildx two-stage build + push to ECR
└── storage/
    ├── pvc-fsx-lustre-dynamic.yaml # OPTIONAL: dynamically provision fsx-claim
    └── pv-fsx-lustre-static.yaml   # OPTIONAL: bind fsx-claim to an existing FSx
```

## Configuration deep-dive

A few non-obvious config decisions are baked into the launchers and the 14B eval
config:

- **`actor.model.num_action_per_block=16` (temporal alignment).**
  `libero_sft_dreamzero_14b.yaml` sets `action_horizon=16` but inherits
  `num_action_per_block=24` from the DROID model default. The mismatch trips the
  forward-pass assertion
  (`actions.shape[1] / (noise.shape[1]-1) == num_action_per_block // num_frame_per_block`;
  got `64/8=8` but expected `24//2=12`). Overriding to `16` makes `16//2=8 == 8`.
  Both the SFT launcher and the eval config set this.

- **Hydra `+` prefix for `metadata_json_path`.** The key is *commented out* in the
  config struct (not part of the Hydra schema), so it must be **added** with the
  `+` prefix: `+actor.model.metadata_json_path=/fsx/models/metadata-libero.json`.
  A plain override (without `+`) fails with "Key not in struct". Set
  `METADATA_PATH=""` in the launcher to fall back to the checkpoint's bundled
  `oxe_droid` metadata instead.

- **DCP → `.pt` conversion is mandatory; do NOT save full model weights from
  FSDP.** SFT writes a sharded DCP checkpoint; eval needs a single `.pt`. Convert
  offline on CPU (step 5). Do **not** pass
  `+actor.fsdp_config.save_full_model_weights=true` — the rank-0 full-state-dict
  gather for the 16B model stalls / never completes on 2× `p5en.48xlarge`.

- **14B eval config resolves the 14B architecture via Hydra `searchpath`.** The
  embodiment eval ships only `model/dreamzero_5b` under its config dir; the 14B
  arch (`model/dreamzero_14b.yaml`) lives in the SFT config tree. The eval config
  adds `examples/sft/config/` to the Hydra `searchpath` so
  `model/dreamzero_14b@actor.model` resolves. Component pretrained paths are left
  `null` — the backbone comes from the DreamZero-DROID safetensors + your
  `full_weights.pt`, not a 5B Wan download.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Pods crash with `ray: command not found`, or inline `${...}` vars render empty | Manifest rendered with unrestricted `envsubst`, clobbering inline shell vars | Always render manifests that embed shell with the **restricted** allow-list: `envsubst '${ECR_URI} ${NAMESPACE}' < ...`. |
| `ray: command not found` on Ray head/worker/submitter | KubeRay runs `ray` non-interactively (`~/.bashrc` not sourced); `ray` lives in the `dreamzero` venv | Already handled in `dreamzero-sft.yaml`: the venv `bin` is prepended to `PATH` on the head/worker `env` **and** on the `submitterPodTemplate`. Don't strip those `PATH` entries. |
| SFT crashes near checkpoint save: `UnpicklingError: invalid load key '\x00'` in `broadcast_object_list`, after shards are written | `dcp.save`'s finalization broadcast runs over the default NCCL/CUDA PG and races with NCCL teardown at the end of a long write (not torch-version-specific — the sync `dcp.save` path is unchanged through ≥ torch 2.8) | Fixed by the in-image `dcp-save-gloo-coordinator.patch`, which passes a dedicated gloo PG so the broadcast runs over CPU/gloo. If you somehow hit this, the **on-disk checkpoint is still valid and convertible** — proceed to step 5 (convert). |
| Convert fails with `EOFError` / `inline_container.cc unexpected pos`, corrupt shards | FSx filled up during the SFT run; `torch.save` was truncated mid-write | Ensure **≥250 GB free** on FSx before SFT (a 14B DCP checkpoint is ~140–206 GB). Free space, re-run SFT. |
| SFT fails with `KeyError: embodiment_tag 'libero_sim' not found` | The DROID checkpoint only bundles `oxe_droid` metadata | Run step 3 (`generate-metadata.yaml`) to produce `/fsx/models/metadata-libero.json` before SFT/eval. |
| Metadata generation or transforms behave wrongly / schema errors | Wrong dataset (`lerobot/libero` has a different `observation.state`/`action` schema) | The dataset **must** be `physical-intelligence/libero`. Re-stage step 2. |
| Forward-pass assertion `actions … != … // …` early in SFT/eval | `num_action_per_block` inherited DROID default of 24 | Override `actor.model.num_action_per_block=16` (the launcher and eval config already do this). |
| Eval CUDA OOM at step 0 (GPU 0 ~280 MB free) | `total_num_envs=128` (upstream 5B default) OOMs the 16.48B model co-located with the sim | Use `total_num_envs=16` (the 14B config default). For a quick smoke eval, override to `8` via `HYDRA_OVERRIDES`. |
| `eval/success_once = 0.0` | The checkpoint came from a 1-step (validation) SFT run | **Expected** for a 1-step checkpoint. Run a multi-step SFT (step 4, raise `runner.max_steps`), re-convert, re-eval for real accuracy. |

## Software versions

| Component | Version |
|-----------|---------|
| RLinf (upstream) | `b3bbabb1f461` |
| DreamZero / `groot` | `ab790c198fbc` |
| EFA installer | 1.47.0 |
| libfabric | 2.4.0 |
| aws-ofi-nccl | 1.18.0 |
| NCCL | v2.21.5-1 |
| KubeRay / Ray | 1.6.0 / 2.55.1 |
| PyTorch | 2.6.0+cu124 |
| diffusers | 0.37.1 |
| lerobot | 0.3.3 |
| torchcodec | 0.2 |

## References

- RLinf training framework — [github.com/RLinf/RLinf](https://github.com/RLinf/RLinf)
- DreamZero (`groot`) model code — [github.com/RLinf/dreamzero](https://github.com/RLinf/dreamzero)
- DreamZero-DROID checkpoint — [huggingface.co/GEAR-Dreams/DreamZero-DROID](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)
- UMT5-XXL tokenizer — [huggingface.co/google/umt5-xxl](https://huggingface.co/google/umt5-xxl)
- LIBERO dataset — [huggingface.co/datasets/physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero)
- EKS cluster architectures — [`1.architectures/4.amazon-eks`](../../../../../1.architectures/4.amazon-eks)

## Security

See [CONTRIBUTING](https://github.com/aws-samples/awsome-distributed-training/blob/main/CONTRIBUTING.md#security-issue-notifications)
for more information. Credentials (Hugging Face tokens, etc.) flow through
Kubernetes Secrets referenced by `secretKeyRef`, never committed to rendered
YAML; `env_vars` is gitignored.

## License

This project is licensed under the MIT-0 License. See the LICENSE file.
