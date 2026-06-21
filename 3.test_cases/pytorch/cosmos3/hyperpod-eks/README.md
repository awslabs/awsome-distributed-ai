# Cosmos 3 on SageMaker HyperPod (EKS)

This directory holds the SageMaker HyperPod (EKS) reference manifests for the Cosmos 3
Physical AI flywheel (generate → post-train → eval). They use the same container (the
AWS Deep Learning Container (DLC) image), the same TOMLs, and the same torchrun launcher
as the [`../kubernetes/`](../kubernetes/) (plain-EKS) path; the only differences are the
managed resilience and node-health-aware scheduling described below. For the full
flywheel narrative, the architecture, and the parallelism details, see the
[top-level `README.md`](../README.md).

## Platform notes

- **Training** (`train-*.yaml`) runs as a Kubeflow `PyTorchJob` (the standard HyperPod
  Helm chart provides the Kubeflow training operator) with the
  `sagemaker.amazonaws.com/enable-job-auto-resume` annotation. The `../kubernetes/` set
  instead uses a `JobSet` with `failurePolicy: {restartStrategy: Recreate}`, which
  restarts the gang on a **pod** failure but does not remediate an unhealthy node.
  HyperPod covers the node layer too: its health-monitoring agent detects a bad node,
  `NodeRecovery: Automatic` replaces it, and job auto-resume restarts the job onto
  healthy capacity. In both cases `cosmos-framework` then resumes from the last Distributed
  Checkpoint (DCP) (`latest_checkpoint.txt`) rather than restarting from scratch.
- **Serving, generation, and storage** use the same workloads as the EKS set,
  with a node-health `nodeSelector` and affinity added so that pods land only on
  `Schedulable` nodes.

## Cluster

Stand up a HyperPod-EKS cluster with the official [Terraform modules](https://github.com/awslabs/awsome-distributed-ai/tree/main/1.architectures/7.sagemaker-hyperpod-eks/terraform-modules).
The modules handle the prerequisites for you: placing the cluster in the same Virtual
Private Cloud (VPC) as EKS, creating private subnets, attaching the SageMaker execution
role (`AmazonSageMakerClusterInstanceRolePolicy`), staging the lifecycle script in S3,
setting `NodeRecovery: Automatic`, and installing the Kubeflow training operator via
the standard HyperPod Helm chart. Enable the **observability add-on**
(CloudWatch Container Insights, Amazon Managed Prometheus, and Managed Grafana, with
Data Center GPU Manager (DCGM) out of the box) so that the GPU-saturation metrics are available.

`OnStartDeepHealthChecks` (`[InstanceStress, InstanceConnectivity]`) is a production
option that gates nodes on deep health before they become `Schedulable`. Note that it
can add over an hour to node allocation, so weigh it against your startup-latency
requirements.

## Prerequisites

- A **SageMaker HyperPod (EKS)** cluster that can schedule `p5en.48xlarge` nodes
  (8× H200 + 16 EFA NICs each) — see [Cluster](#cluster) above for the Terraform setup.
- The **NVIDIA GPU Operator** and **EFA device plugin** (pods request `nvidia.com/gpu`
  and `vpc.amazonaws.com/efa`).
- The Kubeflow **PyTorchJob** operator — provided by the standard HyperPod Helm chart
  (the training manifests run as `PyTorchJob`, not `JobSet`).
- An **FSx for Lustre** filesystem mounted via a `fsx-claim` PVC (datasets, checkpoints,
  outputs), created by the shared manifests in [`../storage/`](../storage/).
- The **AWS DLC image** built and pushed to your ECR (`../build-push.sh`), plus the
  `hf-token` Secret in your namespace (with the `nvidia/Cosmos-Guardrail1` license
  accepted on that account).
- The [**`a8m/envsubst`**](https://github.com/a8m/envsubst) variant (**not** GNU
  gettext's) — see the top-level [README "Deployment"](../README.md#deployment) for why
  and how to install it.

## Manifests

| File | Purpose |
|------|---------|
| `train-multi-node-dlc.yaml` | Distributed action-policy post-training (Kubeflow `PyTorchJob` + torchrun, EFA / Remote Direct Memory Access (RDMA), real Distributed Checkpoint (DCP) warm-start, job auto-resume). |
| `train-vision-sft.yaml` | Vision supervised fine-tuning (SFT) — Cosmos3-Nano and Super-64B LoRA (parameterized by `SFT_TOML` + `DATASET_PATH`). |
| `generate-vllm-omni-super.yaml` | Synthetic data generation (SDG) — Super video-to-video via the official `vllm/vllm-omni:cosmos3` engine (a separate image from the training stack), with node-health scheduling. |
| `serve-policy.yaml` | Policy-server eval (Deployment + Service), with node-health scheduling. |

Storage is shared by both deployment paths and lives in [`../storage/`](../storage/)
(`storage-fsx-efa-sc.yaml` for the EFA-enabled StorageClass + PVC, and `storage-fsx-dra.yaml`
for the optional S3→FSx-Lustre DRA data plane).

## Render & apply

These manifests render with the [`a8m/envsubst`](https://github.com/a8m/envsubst)
variant (**not** GNU gettext's — see the top-level [README "Deployment"](../README.md#deployment)
for why and how to install it). They write the operator-injected runtime variables —
`$$MASTER_ADDR`, `$$MASTER_PORT`, `$$RANK`, `$$WORLD_SIZE` — with the `$$` escape so a8m
leaves them as a literal `$NAME` for the pod's bash. The HyperPod manifests reuse the
same `env_vars` as the EKS set, plus a few HyperPod-specific variables (`JOB_MAX_RETRY`
and `NUM_WORKERS`); set `NUM_WORKERS = NUM_NODES - 1`, because the PyTorchJob node count
is the Master plus the Workers. Run these commands from the test-case root
(`3.test_cases/pytorch/cosmos3/`). If you have not already, create `env_vars` from the
template and edit it for your cluster (registry, FSx paths, Hugging Face (HF) token):

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
envsubst < hyperpod-eks/train-multi-node-dlc.yaml | kubectl apply -f -
```

**Vision SFT (Nano, or Super LoRA).** This uses the same a8m render, with the per-run
variables set inline:

```bash
NUM_NODES=1 NUM_WORKERS=0 RUN_ID=nano-sft-r1 SFT_TOML=vision_sft_nano.toml \
  envsubst < hyperpod-eks/train-vision-sft.yaml | kubectl apply -f -
```

**Synthetic data generation (SDG).** Generation uses the separate
`vllm/vllm-omni:cosmos3` image rather than the AWS DLC training image, and runs as a
plain Job and Service (with node-health scheduling). Point `IMAGE_URI` at the vLLM-Omni
image, then port-forward and POST a video-to-video request:

```bash
export IMAGE_URI=<acct>.dkr.ecr.<region>.amazonaws.com/vllm-omni:cosmos3
envsubst < hyperpod-eks/generate-vllm-omni-super.yaml | kubectl apply -f -
kubectl port-forward svc/cosmos3-vllm-omni 8000:8000
curl -F input_reference=@clip.mp4 http://localhost:8000/v1/videos/sync
```

**Policy-server eval:**

```bash
envsubst < hyperpod-eks/serve-policy.yaml | kubectl apply -f -
```

Tear down GPU workloads (`kubectl delete -f <manifest>`) when a run or generation
batch is done.

## Observability

On HyperPod, the observability add-on emits DCGM and node metrics to Amazon Managed
Prometheus (AMP) out of the box (enabled at cluster creation — see [Cluster](#cluster)).
Point your DCGM saturation analysis at the AMP query endpoint (SigV4-signed). The metric
names (`DCGM_FI_DEV_*`, GPU utilization, and SM-active) are identical to the plain-EKS
path's DCGM exporter, so any DCGM-based saturation analysis works unchanged against AMP.

> **Model FLOPs Utilization (MFU) capture is off by default** (`job.wandb_mode=disabled`),
> matching the EKS manifests. `cosmos-framework`'s `MFUCallback` emits MFU only through
> Weights & Biases (W&B), but a hosted W&B account is **not** required: set
> `job.wandb_mode=offline` to log MFU to a local `wandb/` datastore with no account,
> server, or network. This sample does not publish an MFU figure; if you capture it,
> cross-check the callback's value against **DCGM SM-active** via AMP.
