# Cosmos 3 on Amazon EKS

This directory holds the EKS reference manifests for the Cosmos 3 Physical AI
flywheel (generate → post-train → eval). The [`../hyperpod-eks/`](../hyperpod-eks/)
directory mirrors these manifests with SageMaker HyperPod managed resilience and
node-health scheduling. For the full flywheel narrative, the architecture, and the
parallelism details, see the [top-level `README.md`](../README.md).

## Platform notes

- **Training** (`train-*.yaml`) runs as a [**JobSet**](https://jobset.sigs.k8s.io/)
  (`jobset.x-k8s.io`) with `failurePolicy: {restartStrategy: Recreate}`, which restarts
  the whole gang together on a **pod** failure so NCCL re-forms. JobSet builds directly
  on the upstream Job API and needs no Kubeflow training operator. It restarts on pod
  failures but does **not** remediate an unhealthy node — for managed node recovery, use
  the [`../hyperpod-eks/`](../hyperpod-eks/) path. After a restart, `cosmos-framework`
  resumes from the last Distributed Checkpoint (DCP) (`latest_checkpoint.txt`) rather
  than restarting from scratch.
- **Serving, generation, and storage** run as plain Deployments/Jobs and shared FSx
  storage, with no node-health scheduling on this path.

## Cluster

Use any **Amazon EKS** cluster that can schedule `p5en.48xlarge` nodes (8× H200 + 16 EFA
NICs each) — provisioned however you like (a static managed node group, or Karpenter,
backed by a Capacity Block for ML or an On-Demand Capacity Reservation). See
[`1.architectures/`](../../../../1.architectures/) for reference cluster setup (the
[EKS](../../../../1.architectures/4.amazon-eks/) and
[SageMaker HyperPod EKS](../../../../1.architectures/7.sagemaker-hyperpod-eks/) modules).

## Prerequisites

- An **Amazon EKS** cluster with GPU nodes (validated on `p5en.48xlarge`, 8× H200; see
  [Cluster](#cluster) above).
- The **NVIDIA GPU Operator** (advertises `nvidia.com/gpu`) and, for multi-node
  training, the **Elastic Fabric Adapter (EFA) device plugin** (advertises `vpc.amazonaws.com/efa`).
- The **JobSet controller** (`jobset.x-k8s.io`) installed — the multi-node training
  manifests launch the gang as a `JobSet` (see "Why JobSet" in the top README).
- An **FSx for Lustre** filesystem mounted via a PersistentVolumeClaim (PVC) named
  `fsx-claim` (datasets, checkpoints, and outputs live here), created by the shared
  manifests in [`../storage/`](../storage/).
- The **AWS Deep Learning Container (DLC) image** built and pushed to your Elastic
  Container Registry (ECR) (`../build-push.sh`), plus the `hf-token` Secret in your
  namespace (with the `nvidia/Cosmos-Guardrail1` license accepted on that account).
- The [**`a8m/envsubst`**](https://github.com/a8m/envsubst) variant (**not** GNU
  gettext's) — see the top-level [README "Deployment"](../README.md#deployment) for why
  and how to install it.

## Manifests

| File | Purpose |
|------|---------|
| `train-multi-node-dlc.yaml` | Distributed action-policy post-training (JobSet + torchrun, EFA / Remote Direct Memory Access (RDMA), real Distributed Checkpoint (DCP) warm-start). |
| `train-vision-sft.yaml` | Vision supervised fine-tuning (SFT) — Cosmos3-Nano and Super-64B LoRA (parameterized by `SFT_TOML` + `DATASET_PATH`). |
| `generate-vllm-omni-super.yaml` | Synthetic data generation (SDG) — Super video-to-video via the official `vllm/vllm-omni:cosmos3` engine (a separate image from the training stack). |
| `serve-policy.yaml` | Policy-server eval (Deployment + Service). |

Storage is shared by both deployment paths and lives in [`../storage/`](../storage/)
(`storage-fsx-efa-sc.yaml` for the EFA-enabled StorageClass + PVC, and `storage-fsx-dra.yaml`
for the optional S3→FSx-Lustre DRA data plane).

## Render & apply

All manifests are parameterized and render with the [`a8m/envsubst`](https://github.com/a8m/envsubst)
variant (**not** GNU gettext's — see the top-level [README "Deployment"](../README.md#deployment)
for why and how to install it). The manifests write in-container runtime variables as
`$$NODE_RANK` / `$$MASTER` so a8m collapses them to a literal `$NAME` the pod's bash
expands. Run these commands from the test-case root (`3.test_cases/pytorch/cosmos3/`).
If you have not already, create `env_vars` from the template and edit it for your
cluster (registry, FSx paths, Hugging Face (HF) token):

```bash
cp env_vars.example env_vars   # then edit for your cluster
set -a; . ./env_vars; set +a
# the manifests read the HF token from a Secret named hf-token (key: token):
kubectl create secret generic hf-token -n "$NAMESPACE" --from-literal=token="$HF_TOKEN"
```

**Storage (once):**

```bash
envsubst < storage/storage-fsx-efa-sc.yaml | kubectl apply -f -
envsubst < storage/storage-fsx-dra.yaml    | kubectl apply -f -   # optional S3->FSx DRA
```

**Action-policy post-training (multi-node).** Smoke vs real is **env-driven** (no YAML
edit): the defaults run a smoke (`POLICY_TOML=droid_policy_smoke.toml`,
`CKPT_TYPE=dummy`); for a real warm-start set `POLICY_TOML=droid_policy.toml` +
`CKPT_TYPE=dcp` and stage `BASE_CHECKPOINT_PATH`:

```bash
envsubst < kubernetes/train-multi-node-dlc.yaml | kubectl apply -f -
```

**Vision SFT (Nano, or Super LoRA).** This uses the same a8m render, with the per-run
variables set inline:

```bash
NUM_NODES=1 RUN_ID=nano-sft-r1 SFT_TOML=vision_sft_nano.toml \
  envsubst < kubernetes/train-vision-sft.yaml | kubectl apply -f -
```

**Synthetic data generation (SDG).** Point `IMAGE_URI` at the vLLM-Omni image (not the
DLC training image), then port-forward and POST a video-to-video request:

```bash
export IMAGE_URI=<acct>.dkr.ecr.<region>.amazonaws.com/vllm-omni:cosmos3
envsubst < kubernetes/generate-vllm-omni-super.yaml | kubectl apply -f -
kubectl port-forward svc/cosmos3-vllm-omni 8000:8000
curl -F input_reference=@clip.mp4 http://localhost:8000/v1/videos/sync
```

**Policy-server eval:**

```bash
envsubst < kubernetes/serve-policy.yaml | kubectl apply -f -
```

Tear down GPU workloads (`kubectl delete -f <manifest>`) when a run or generation
batch is done.

## Observability

This path relies on the **NVIDIA GPU-Operator's DCGM exporter** (a prerequisite, not
shipped here) for GPU-saturation signals — SM-active, high-bandwidth memory (HBM)
bandwidth, and TensorCore-active (`DCGM_FI_DEV_*`). Scrape the exporter with Prometheus
(or your cluster's existing monitoring). The HyperPod path instead emits these same
metrics to Amazon Managed Prometheus via the observability add-on (see
[`../hyperpod-eks/`](../hyperpod-eks/)).

> **Model FLOPs Utilization (MFU) capture is off by default** (`job.wandb_mode=disabled`).
> `cosmos-framework`'s `MFUCallback` emits MFU only through Weights & Biases (W&B), but a
> hosted W&B account is **not** required: set `job.wandb_mode=offline` to log MFU to a
> local `wandb/` datastore with no account, server, or network. This sample does not
> publish an MFU figure; if you capture it, cross-check the callback's value against DCGM
> SM-active.
