# Reinforcement Learning with miles on Amazon SageMaker HyperPod EKS

This test case runs GRPO post-training with [**miles**](https://github.com/radixark/miles) on
[Amazon SageMaker HyperPod](https://aws.amazon.com/sagemaker/hyperpod/) with Amazon EKS
orchestration. It mirrors the sibling [`3.test_cases/pytorch/slime/`](../slime/) test case
end to end -- building a container image, preparing data, deploying a multi-node Ray cluster,
converting model checkpoints, and launching GRPO training on NVIDIA GPUs interconnected with
Elastic Fabric Adapter (EFA) -- but targets **miles**, a fork of SLIME built for CUDA 13 and
NVIDIA Blackwell as first-class hardware.

The hardware-verified runs recorded here were done on a plain Amazon EKS cluster with KubeRay,
EFA, and FSx for Lustre. The same manifests are expected to run unchanged on a SageMaker
HyperPod EKS cluster -- which adds deep health checks and automatic node replacement on top of
the same EKS/KubeRay/EFA stack -- but that was not separately validated.

## Introduction

[**miles**](https://github.com/radixark/miles) is a direct fork of
[**SLIME**](https://github.com/THUDM/slime) (an LLM post-training framework for RL scaling).
It inherits SLIME's core design -- integrating two best-of-breed backends under Ray
orchestration:

- **[SGLang](https://github.com/sgl-project/sglang)** for high-throughput rollout generation
  (inference), providing RadixAttention, continuous batching, and tensor parallelism.
- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** (a radixark fork) for scalable
  distributed training, with support for TP, PP, CP, EP, and ZeRO-style sharding.

**Ray** manages resource orchestration, supporting both **colocated** (training and rollout
share the same GPUs, time-sliced) and **disaggregated** (separate GPU pools connected by
NCCL/EFA weight sync) deployment topologies -- the same two topologies SLIME supports.

Note that "disaggregated" appears here in two unrelated senses, and the files use both.
`COLOCATE=false` splits the **actor and rollout GPU pools**, which is what selects the weight
sync implementation. `env_vars.disaggregated.example` does something else entirely: it moves
**reward scoring** to a CPU pool and leaves the GPU layout alone. The two are independent and
can be combined.

What miles changes relative to upstream SLIME is the platform target: it ships a matched
PyTorch 2.11 / CUDA 13.0.1 stack with prebuilt `flash_attn` / Transformer Engine / `apex`
wheels and a Blackwell (sm_103) Transformer Engine FA2 whitelist patch, and it rewrites the
rollout/training weight-sync path from HTTP endpoints on an SGLang fork to direct Ray actor
methods. See [Why miles](#why-miles-vs-slime) below.

**Amazon SageMaker HyperPod** provides purpose-built infrastructure for distributed model
training with deep health checks, automatic node replacement, and managed Kubernetes (EKS)
integration. Combined with FSx for Lustre shared storage and EFA networking, HyperPod
delivers the resilient, high-performance fabric that large-scale RL workloads demand.

## Why miles vs SLIME

| Aspect | SLIME | miles |
|--------|-------|-------|
| Target CUDA / GPU generation | NGC-image CUDA stack, Hopper-first | CUDA 13.0.1 native, NVIDIA Blackwell (sm_103) first-class |
| Base image | NGC PyTorch (nightly ABI) | `radixark/miles` (matched PyTorch 2.11 + cu130 stable ABI) |
| Rollout <-> training weight sync | HTTP endpoints on an SGLang fork | Ray actor methods (`begin_weight_update` / `pull_weights`) on the rollout engine directly |
| Framework install path (in image) | `/opt/slime` | `/root/miles` (editable install) |
| Megatron fork | NVIDIA/Megatron-LM (SLIME-compatible commit) | radixark/Megatron-LM fork |
| SGLang version pinned | 0.5.12.post1 | 0.5.16.dev |
| `train.py` CLI / GRPO flags | -- | Compatible with SLIME's (same names, same semantics) |
| Relationship | Upstream | Direct fork of SLIME (diverged at commit `fcce96ca0`, 2025-10-05) |

Because miles is a fork rather than an independent reimplementation, the `train.py` CLI used
by this test case's recipes is drop-in compatible with SLIME's -- the [Quick
Start](#quick-start) below tracks the sibling slime test case step for step.

## Architecture

The deployment uses a colocated architecture where training and rollout share the same GPU
pool, connected by EFA networking and FSx for Lustre shared storage. (A disaggregated
topology -- separate GPU pools for training and rollout -- is also supported by miles/Ray but
is not the hardware-verified path documented here; see [Verification Status](#verification-status).)

```
+-----------------------------------------------------------------------+
|                  Amazon SageMaker HyperPod EKS Cluster                |
|                                                                       |
|  +-----------------------------+  +-----------------------------+     |
|  |  Node 1 (GPU worker)         |  |  Node 2 (GPU worker)         |     |
|  |  8x H-series GPU  |  EFA     |  |  8x H-series GPU  |  EFA     |     |
|  +-----------------------------+  +-----------------------------+     |
|           |           |                    |           |               |
|           +--------------EFA (GPU<->GPU RDMA)-----------+               |
|                        |                               |               |
|  +----------------------------------------------------------+         |
|  |          FSx for Lustre, mounted as PVC `fsx-claim`      |         |
|  |  /fsx/models  /fsx/data  /fsx/runs                       |         |
|  +----------------------------------------------------------+         |
|                                                                       |
|  +-----------------------------+                                     |
|  |  Ray head (num-gpus 0)       |  <- co-located on the GPU pool;     |
|  |  on a GPU node (needs CUDA)  |     needs libcuda for Megatron impt |
|  +-----------------------------+                                     |
|                                                                       |
|  Kubernetes Resources:                                                |
|  - KubeRay operator                                                   |
|  - EFA device plugin, NVIDIA device plugin (kube-system)             |
|  - FSx CSI driver (kube-system)                                       |
+-----------------------------------------------------------------------+
```

### miles Internal Loop

```
                    +------------------+
                    |   Data Buffer    |
                    | (prompt queue +  |
                    |  rollout cache)  |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
   +-------------------+        +-------------------+
   |     Rollout        |        |     Training       |
   |  (SGLang engines,  |        |  (Megatron-LM      |
   |   Ray actors)      | <----> |   TP/PP/CP/EP)     |
   |                    | weight  |                    |
   |  - RadixAttention  |  sync  |  - GRPO             |
   |  - Cont. batching  | (Ray    |  - Dynamic batch   |
   |  - TP per engine   |  actor  |  - Gradient ckpt   |
   |                    |  calls) |                    |
   +-------------------+        +-------------------+
```

1. **Data Buffer** manages prompts, dispatches them for rollout, and stores generated samples
   with rewards.
2. **Rollout** runs SGLang engines as Ray actors, generating responses and scoring them via a
   reward function. Unlike SLIME, weight-sync calls (`begin_weight_update` / `pull_weights`)
   go directly to Ray actor methods on the rollout engine rather than HTTP endpoints.
3. **Training** reads batches from the Data Buffer, computes GRPO advantages, and updates the
   policy via Megatron-LM. Updated weights are synced back to the rollout engines.

## Hardware Requirements

miles's own image already builds for Hopper and Blackwell alike (CUDA 13.0.1, with the
sm_103 Transformer Engine FA2 whitelist patch applied on top of the standard sm_90 build),
and nothing in this test case's recipes, env files, or manifests hard-codes a GPU generation
or CUDA/SM version -- the only per-cluster knob is which accelerator NodePool to target
(`GPU_NODE_ROLE` in `env_vars`, substituted into `kubernetes/raycluster.yaml`'s
`nodeSelector`). The table below is the hardware this was actually run on; a B300 row is
included as the expected-compatible target (see Verification Status for what is and is not
confirmed on real hardware).

| Component | H200 (verified) | B300 (expected-compatible, unverified) |
|-----------|------------------|------------------------------------------|
| **Instance type** | p5en.48xlarge | p6-b300.48xlarge |
| **GPUs per node** | 8x NVIDIA H200 (141 GB HBM) | 8x NVIDIA B300 (288 GB HBM) |
| **EFA per node** | 16 physical, **15 allocatable** on EKS | 16 physical, verify allocatable |
| **Storage** | FSx for Lustre, mounted as PVC `fsx-claim` | same |
| **Kubernetes** | EKS, KubeRay operator | same |
| **Ray head placement** | The head co-locates on the GPU pool (`num-gpus 0`); it must be on a CUDA-capable node (miles control actors import Megatron even at `num-gpus 0`). Give the GPU node's root volume >=150 GiB for the ~18 GB image | same |

B300's ~2x larger HBM is expected to relax the memory-driven choices made for H200 in
`env_vars.colocated.example` (e.g. the 30B MoE case needing `--use-distributed-optimizer` and
a colocated 16-GPU layout to avoid OOM at 8 GPU -- see the ALTERNATE block in that file); it
should not require code changes, only re-tuning env values once measured on B300.

### Verification Status

This test case is presented as a GRPO reference, not a research study -- the table below
states plainly what has been run on real hardware and what has not, so you know exactly what
you are inheriting.

Every row below is backed by a recorded run: see [docs/VERIFICATION_LOG.md](./docs/VERIFICATION_LOG.md) for the submission commands, Ray job ids, the flags each
job actually received, and the metric values read back from the trainer's event files.

| Component | Status | Environment / Note |
|-----------|--------|--------------------|
| Qwen3-4B GRPO, colocated, 1 node (8 GPU) | Verified | p5en.48xlarge, H200x8 |
| Qwen3-4B GRPO, colocated, 2 nodes (16 GPU) over EFA, 3 rollout cycles | Verified | 2x p5en.48xlarge; all 3 cycles SUCCEEDED in 676s with `raw_reward` 0.477/0.523/0.492 -- metrics in [docs/VERIFICATION_LOG.md](./docs/VERIFICATION_LOG.md). Fabric separately measured at NCCL all_reduce busbw 190-257 GB/s over EFA (`efa-direct` + GPUDirect RDMA, no TCP fallback), see [docs/EFA_2NODE.md](./docs/EFA_2NODE.md) |
| Qwen3-30B-A3B MoE GRPO, colocated, 2 nodes (16 GPU) -- trains cleanly | Verified | actor spans all 16 GPU with `--use-distributed-optimizer` (required -- confining the actor to 8 GPU OOMs) and the SGLang `triton` MoE runner backend. The shipped block runs the rollout MoE pure expert-parallel (`moe_tp=1`, `EP_SIZE = ROLLOUT_GPUS_PER_ENGINE = 2`): `rollout/raw_reward` 0.578, `rollout/repetition_frac` 0.0, `weight_version` uniform / `mixed_version_ratio` 0.0 -- comparable to the dense 4B run. Running the MoE tensor-parallel AND expert-parallel at once (`moe_tp>1` and `moe_ep>1`) hits a FlashInfer allreduce-fusion bug in this build, so the recipe disables that fusion for such geometries and they train too -- see [Known Issues](#known-issues) item 2 |
| Qwen3-4B GRPO, **disaggregated** (`COLOCATE=false`), 2 nodes | Verified | run on 2x p5en (actor 8 + rollout 8): Ray job SUCCEEDED, `raw_reward` 0.52, `repetition_frac` 0.0, `weight_version` 2 with `mixed_version_ratio` 0.0 -- i.e. weights synced across the node boundary over NCCL/EFA (`UpdateWeightFromDistributed`). A disaggregated run needs GPU nodes for BOTH the actor and rollout pools, so `WORKER_REPLICAS` (derived in `env_vars` as `ceil((ACTOR_NUM_NODES*ACTOR_GPUS_PER_NODE + ROLLOUT_NUM_GPUS)/8)`, here 2) drives the RayCluster worker count rather than `ACTOR_NUM_NODES` alone (see [docs/VERIFICATION_LOG.md](./docs/VERIFICATION_LOG.md)) |
| Qwen3-30B-A3B MoE GRPO, **disaggregated** | UNVERIFIED | the disaggregated actor-8 layout needs B300-class 288GB HBM; only the colocated 16-GPU layout was run, on H200 |
| Any workload on B300 (p6-b300.48xlarge) | UNVERIFIED | expected-compatible (miles's base image already targets CUDA 13 / sm_103; nothing here is GPU-generation-specific) but not run on this hardware -- see Hardware Requirements |
| RayCluster manifest as shipped (head co-located on the GPU pool) | Verified | the shipped `kubernetes/raycluster.yaml` schedules the head onto the GPU pool (`num-gpus 0`, with a `nvidia.com/gpu` toleration) alongside the worker: dense Qwen3-4B GRPO SUCCEEDED with `raw_reward` 0.53 and `repetition_frac` 0.0. The head must be on a CUDA-capable node -- miles control actors import Megatron even at `num-gpus 0`, so a CPU-only head node fails with `libcuda.so.1: cannot open shared object file`. See [docs/VERIFICATION_LOG.md](./docs/VERIFICATION_LOG.md) |
| Disaggregated reward service (`remote_rm` on a CPU pool) | UNVERIFIED | `reward_service/`, `kubernetes/reward-service.yaml` are present and mirror the sibling slime test case, but were not deployed/exercised on miles |
| Checkpoint save-back / long-run checkpointing | Known Issue | `save_model()` fails with a pickle-truncation error in Megatron's distributed checkpoint save; see [Known Issues](#known-issues) |

## Prerequisites

This test case does not create cluster infrastructure; it deploys onto a cluster that already
provides the pieces below. The manifests reference them by label/name, so if any is missing the
failure is a scheduling or mount error, not an obvious message. Confirm each before starting.

1. An Amazon SageMaker HyperPod cluster with EKS orchestration and GPU instance groups
   (p5en.48xlarge/H200 -- verified -- or p6-b300.48xlarge/B300 -- expected-compatible,
   see Hardware Requirements -- with EFA).
2. **Node placement labels.** `kubernetes/raycluster.yaml` schedules the GPU workers on
   `${GPU_NODE_LABEL_KEY}: ${GPU_NODE_ROLE}` and the Ray head on
   `${CPU_NODE_LABEL_KEY}: ${CPU_NODE_ROLE}`. There is no universal `node-role` label; both the
   key and the value are cluster-specific. You can either point the four env vars at a label your
   nodes already carry -- e.g. `GPU_NODE_LABEL_KEY=node.kubernetes.io/instance-type`,
   `GPU_NODE_ROLE=p5en.48xlarge`, or on SageMaker HyperPod the
   `sagemaker.amazonaws.com/instance-group-name` of your GPU group -- or add your own label:
   `kubectl label node <gpu-node> node-role=gpu` (then `GPU_NODE_LABEL_KEY=node-role`,
   `GPU_NODE_ROLE=gpu`). If you have no dedicated CPU pool, set the CPU_ vars equal to the GPU
   ones (the head runs `num-gpus 0`). A wrong key OR value leaves pods `Pending` with
   `FailedScheduling`, not an obvious error.
3. **Cluster add-ons that advertise the scheduled resources**, all of which the manifests
   request and none of which this test case installs:
   - the NVIDIA device plugin (or GPU Operator, or a GPU AMI that bundles it) advertising
     `nvidia.com/gpu`;
   - the AWS EFA Kubernetes device plugin (`aws-efa-k8s-device-plugin`) advertising
     `vpc.amazonaws.com/efa` -- read the allocatable count off a node for `EFA_PER_NODE`
     (`kubectl get node <gpu-node> -o jsonpath='{.status.allocatable.vpc\.amazonaws\.com/efa}'`);
   - the FSx for Lustre CSI driver, bound to the `fsx-claim` PVC below.
4. `kubectl` and `helm` configured to access the cluster.
5. The KubeRay operator installed (see step 0 below).
6. The Ray head co-locates on the GPU pool (it runs `num-gpus 0` and uses no GPU, only CPU/disk).
   It must be on a CUDA-capable node: miles control actors import Megatron/`transformer_engine`
   even at `num-gpus 0`, which loads `libcuda.so.1` at import, so a CPU-only head node fails.
   Ensure the GPU node's root volume has ample ephemeral-storage (>=150 GiB) for the ~18 GB image.
7. An FSx for Lustre `PersistentVolumeClaim` named `fsx-claim`, RWX, mounted at `/fsx`.
8. **EFA security group (multi-node / `COLOCATE=false` only).** The EFA node security group must
   allow all traffic to itself on BOTH ingress AND egress (self-referencing). EFA's OS-bypass
   SRD traffic is not ordinary IP, so a CIDR-only egress rule does not authorize it and NCCL
   over EFA fails with `Unreachable remote` / `Unexpected number of remote rails`. See
   [docs/EFA_2NODE.md](./docs/EFA_2NODE.md).
9. Container registry access (e.g. Amazon ECR) for building/pushing images.
10. A Hugging Face account and access token for model downloads.
11. A Kubernetes Secret named `hf-token` in the target namespace. `kubernetes/raycluster.yaml`
    mounts it without `optional: true`, so the Ray pods will not start without it; if your model
    is public, create it with an empty value (step 1 has the command).

### Cluster assumptions and portability

`kubernetes/raycluster.yaml` was validated on an EKS cluster meeting the Prerequisites above
(a `p5en.48xlarge` / H200 pool with the EFA and FSx add-ons) and carries a few values sized for
that node; on a different cluster, adjust these before deploying or the failure is a silent
`Pending`/`ImagePullBackOff`/OOM rather than a message:

- **EFA is requested unconditionally, including the single-node colocated run.** Without the EFA
  device plugin (or on an instance type without EFA), the worker is `Unschedulable`. For a
  single-node run on a non-EFA cluster, delete the two `vpc.amazonaws.com/efa` lines from the
  worker `resources`; multi-node runs require EFA (verified) and the self-referencing SG above.
- **Worker resources are sized for `p5en.48xlarge` (2 TiB RAM):** cpu 90/96, memory 1800/1900 Gi,
  and a 256 Gi memory-backed `/dev/shm` (which counts against the pod memory limit). On a smaller
  GPU node, lower these below the node's allocatable or the worker will not schedule.
- **Every node pulls the ~18 GB image**, not just the head: give GPU node root volumes >=100 GiB
  free as well (the head needs >=150 GiB, prerequisite 6).
- **Registry auth is assumed to come from the node IAM role (ECR).** For a private or
  cross-account registry, add `imagePullSecrets` to the pod specs.
- **`fsx-claim` must be `ReadWriteMany` and exist in the target `${NAMESPACE}`.** A PVC is
  namespaced, so one created in `default` is invisible to a run in another namespace; an
  `ReadWriteOnce` (e.g. EBS) claim fails to mount across a multi-node run.
- **The disaggregated reward-service overlay** (`kubernetes/reward-service.yaml`, UNVERIFIED)
  selects nodes by the HyperPod label `sagemaker.amazonaws.com/instance-group-name`, which a
  non-HyperPod cluster does not have; edit its `nodeSelector` for your CPU pool if you deploy it.
- **One worker pod consumes a whole 8-GPU node** (`nvidia.com/gpu: 8`, `num-gpus: '8'`). On
  nodes with a different GPU count, change these together with `ACTOR_GPUS_PER_NODE` and the
  actor/rollout GPU split; the manifest is not a fractional-GPU RayCluster.
- **If you point `CPU_NODE_ROLE` at the GPU pool** (no dedicated CPU pool), the head also needs
  a toleration for that pool's taint (commonly `nvidia.com/gpu:NoSchedule`), or it stays Pending
  even though the label matches. Add it to the head pod spec.
- **`/fsx` must already hold the model and data before you launch.** This test case does not
  download or convert during training: `MODEL_LOCAL` (HF checkpoint), `MODEL_DIST` (Megatron
  `torch_dist`, from step 4), `PROMPT_DATA`, and `EVAL_DATA` must all exist. Quick Start steps 3
  and 4 create them.

Preflight (run after `source env_vars`, before deploying) -- turns a silent `Pending`/mount
failure into an early, named error:

```bash
: "${NAMESPACE:?}" "${FULL_IMAGE:?}" "${FSX_CLAIM:?}" "${GPU_NODE_ROLE:?}" "${CPU_NODE_ROLE:?}" "${EFA_PER_NODE:?}" "${WORKER_REPLICAS:?}"
kubectl get nodes -l "${GPU_NODE_LABEL_KEY}=${GPU_NODE_ROLE}" -o name    # GPU pool exists?
kubectl get pvc "${FSX_CLAIM}" -n "${NAMESPACE}"                          # PVC Bound, RWX, this ns?
kubectl get secret hf-token -n "${NAMESPACE}"                            # hf-token present?
kubectl get crd rayclusters.ray.io                                       # KubeRay installed?
# and on /fsx (from a pod that mounts it):
#   ls -ld "$MODEL_LOCAL" "$MODEL_DIST"; test -s "$PROMPT_DATA"; test -s "$EVAL_DATA"
```

## Quick Start

### 0. Install the KubeRay Operator (one-time per cluster)

The Ray cluster is managed by the KubeRay operator. If it is not already present
(`kubectl get crd rayclusters.ray.io`), install it with Helm:

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator \
    --version 1.4.2 --namespace kuberay-operator --create-namespace
```

### 1. Configure Environment Variables

```bash
cd 3.test_cases/pytorch/miles
cp env_vars.colocated.example env_vars
# Edit env_vars with your cluster-specific values
source env_vars
```

Key variables (see `env_vars.colocated.example` for the full annotated file):

```bash
# AWS / ECR (region and account are auto-derived from your AWS credentials)
export AWS_REGION="${AWS_REGION:-$(aws configure get region)}"
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/"
export IMAGE="miles-hyperpod"
export TAG="miles-cuda13-efa-0.1"        # pinned image tag (avoid :latest)
export FULL_IMAGE="${REGISTRY}${IMAGE}:${TAG}"

# Model (Qwen3-4B dense, bf16, colocated -- the verified default)
export MODEL_NAME="Qwen/Qwen3-4B"
export MODEL_LOCAL="/fsx/models/Qwen3-4B"                # SGLang rollout init
export MODEL_DIST="/fsx/models/Qwen3-4B_torch_dist"      # Megatron training (torch_dist)
export COLOCATE="true"
export ACTOR_NUM_NODES=1
export ACTOR_GPUS_PER_NODE=8
export ROLLOUT_NUM_GPUS=8

# Cluster
export NAMESPACE="default"
export FSX_CLAIM="fsx-claim"
export GPU_NODE_ROLE="gpu-p5en"   # node-role label on your GPU pool
export EFA_PER_NODE=15            # the node's FULL allocatable EFA count -- see env_vars
```

`ACTOR_NUM_NODES` drives the RayCluster's worker replica count as well as the actor layout, so
one variable sets both and the manifest cannot disagree with the recipe. Raise it to 2 for the
2-node (16 GPU) configurations.

The pods read `HF_TOKEN` from a Kubernetes Secret (not from `env_vars` / the Ray runtime-env),
so it never lands in the Ray dashboard's job-submission metadata. Create it once per
namespace. This Secret is **required**, not optional: `kubernetes/raycluster.yaml` references
it without `optional: true`, so if it is absent the Ray pods never start and report
`CreateContainerConfigError` rather than anything about HuggingFace. If the model you point at
is public and you have no token, create the Secret with an empty value -- the pods only need
the key to exist:

```bash
kubectl create secret generic hf-token \
  --from-literal=HF_TOKEN=hf_xxx \
  -n "${NAMESPACE}"

# Public model and no token? The pods only need the key to exist:
#   kubectl create secret generic hf-token --from-literal=HF_TOKEN= -n "${NAMESPACE}"
```

`env_vars.colocated.example` also ships a commented-out **ALTERNATE** block for
Qwen3-30B-A3B MoE, colocated on 2 nodes (16 GPU) -- uncomment it (and the matching recipe in
step 7) to run the MoE configuration instead of the 4B dense default. It trains cleanly as
shipped (`rollout/raw_reward` 0.578, `rollout/repetition_frac` 0.0), running the rollout MoE
pure expert-parallel (`moe_tp=1`, `EP_SIZE = ROLLOUT_GPUS_PER_ENGINE`). Geometries that run it
tensor-parallel and expert-parallel at once (`moe_tp>1` and `moe_ep>1`) hit a FlashInfer
allreduce-fusion bug in this build; the recipe disables that fusion for them so they train too
(see [Known Issues](#known-issues) item 2).

### 2. Build and Push the Container Image

The image takes `radixark/miles:<dated-tag>` (which already bundles miles, SGLang,
Megatron-LM, and the matched PyTorch 2.11 / CUDA 13.0.1 stack) as its base and adds **only**
the AWS EFA networking layer (`miles.Dockerfile`) -- GDRCopy, the EFA installer, and NCCL/EFA
runtime defaults. Pin the base by digest (`MILES_BASE_DIGEST` build arg, with `MILES_BASE_TAG` kept alongside for readability) and pin the resulting image
tag to dated/versioned values; never use `:latest`.

```bash
# Authenticate to ECR
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${REGISTRY}

# Create repository (first time only)
aws ecr create-repository --repository-name ${IMAGE} --region ${AWS_REGION} || true

# Build image (build context is this test-case directory)
docker build -t ${FULL_IMAGE} -f miles.Dockerfile .
docker push ${FULL_IMAGE}
```

If the cluster has no node with local Docker access, `kubernetes/buildkit-job.yaml` builds
and pushes the image **in-cluster** with a rootless buildkit Job instead (CPU/IO-only, no GPU
required) -- edit its `nodeSelector` to a node with >=120 GiB free ephemeral-storage, then
create the ConfigMap and Secret it mounts and apply it through `envsubst` (like
`raycluster.yaml` above) so `${AWS_ACCOUNT_ID}`/`${AWS_REGION}` resolve:

```bash
kubectl create configmap miles-build-context \
  --from-file=Dockerfile=miles.Dockerfile -n "${NAMESPACE}"
kubectl create secret docker-registry ecr-miles-push \
  --docker-server="${REGISTRY}" --docker-username=AWS \
  --docker-password="$(aws ecr get-login-password --region "${AWS_REGION}")" \
  -n "${NAMESPACE}"
envsubst < kubernetes/buildkit-job.yaml | kubectl apply -f -
kubectl -n "${NAMESPACE}" logs -f job/miles-efa-build
```

### 3. Download and Prepare the Model

```bash
# Create a data-prep pod
envsubst < kubernetes/data-prep-pod.yaml | kubectl apply -f -
kubectl exec -it data-prep -- bash

# Inside the pod:
pip install huggingface_hub
# HF_TOKEN comes from the hf-token Secret. If you created that Secret with an empty
# value because the model is public, skip this login.
huggingface-cli login --token "${HF_TOKEN}"

# Download model
huggingface-cli download Qwen/Qwen3-4B --local-dir /fsx/models/Qwen3-4B

# Download training + evaluation datasets
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
    --local-dir /fsx/data/dapo-math-17k
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
    --local-dir /fsx/data/aime-2024
```

### 4. Convert Model Weights to Megatron Format

miles's Megatron training backend requires weights in `torch_dist` format. Use the bundled
conversion helper, which sources the model script (`scripts/models/<model>.sh`) from the
miles install at `/root/miles`:

Run this **inside a pod built from the miles image**, not on your workstation: the script reads
the model scripts from `/root/miles`, writes to `/fsx`, and uses `torchrun` when `--num-gpus` is
passed. A **GPU worker** of the RayCluster is such a pod, so deploy step 5 first and come back
here, or run the same command as a one-off Job on the GPU node pool. It must not be the head:
the head runs with `num-gpus 0` and `NVIDIA_VISIBLE_DEVICES=void`, where importing CUDA fails
(see `docs/PORT_NOTES.md`).

```bash
# Must be a GPU worker, not the head -- see above.
W=$(kubectl get pod -n "${NAMESPACE}" -l ray.io/node-type=worker -o name | head -1)
W=${W#pod/}
kubectl cp scripts/convert_checkpoint.sh "${NAMESPACE}/${W}:/tmp/convert_checkpoint.sh"
kubectl exec -n "${NAMESPACE}" "${W}" -- \
  bash /tmp/convert_checkpoint.sh hf2megatron \
    --model-script qwen3-4B.sh \
    --hf-path /fsx/models/Qwen3-4B \
    --save-path /fsx/models/Qwen3-4B_torch_dist
```

For larger models, pass `--num-gpus 8` to parallelize the conversion with `torchrun`:

```bash
bash scripts/convert_checkpoint.sh hf2megatron \
    --model-script qwen3-30B-A3B.sh \
    --hf-path /fsx/models/Qwen3-30B-A3B \
    --save-path /fsx/models/Qwen3-30B-A3B_torch_dist \
    --num-gpus 8
```

### 5. Deploy the Ray Cluster

```bash
# Substitute environment variables into the manifest
source env_vars
envsubst < kubernetes/raycluster.yaml | kubectl apply -f -

# Watch pods come up (1 head + workers)
kubectl get pods -w -l ray.io/is-ray-node=yes

# Port-forward the Ray dashboard (the recipes submit to 127.0.0.1:8265)
kubectl port-forward -n "${NAMESPACE}" svc/miles-ray-head-svc 8265:8265 &
```

The shipped manifest schedules the Ray head onto the GPU pool (`${CPU_NODE_ROLE}` defaults to
`${GPU_NODE_ROLE}`, with a `nvidia.com/gpu` toleration; the head runs `num-gpus 0`) and GPU
workers on `node-role: ${GPU_NODE_ROLE}`, each worker declaring a `gpu_node` custom Ray resource
(see [miles-specific requirements](#miles-specific-requirements-found-on-real-hardware) for
why). The head must be on a CUDA-capable node -- a CPU-only head fails with `libcuda.so.1`
because miles control actors import Megatron even at `num-gpus 0`. Ensure the GPU node's root
volume has headroom for the ~18 GB image (150 GiB or more recommended).

### 6. Configure the Reward

This sample scores rollouts in one of two ways -- pick one in `env_vars`:

- **Built-in rule-based (default, verified):** `RM_TYPE="deepscaler"` (also `dapo`, `math`,
  `f1`, `gpqa`). Runs in-process on the rollout actors; no extra setup. This is the reward
  path used for every hardware-verified run in this test case.
- **Remote reward service on a CPU pool (`RM_TYPE="remote_rm"` + `RM_URL`):** offloads scoring
  to a separate CPU instance group via `reward_service/` and
  `kubernetes/reward-service.yaml`, mirroring the sibling slime test case. This path is
  **UNVERIFIED on miles** -- the manifests are present but have not been deployed/exercised
  here, and miles's `remote_rm` client lacks slime's retry-with-backoff behavior. See
  `env_vars.disaggregated.example` for the overlay.

No file copy is needed for the default path -- `deepscaler` is built into miles.

### 7. Launch GRPO Training

The recipes end in `ray job submit --address http://127.0.0.1:8265`, so run them from a machine
that has the Ray CLI whose version matches the cluster (`pip install "ray==<cluster version>"`)
with the dashboard port-forwarded (step 5). If you cannot install a matching Ray CLI locally --
e.g. no wheel exists for your local Python version -- or want to skip the port-forward, use the
`./run-on-cluster.sh` helper instead, which ships the recipe into the head pod and runs it there
(Ray is already present, `127.0.0.1:8265` is the head's own dashboard, and the version always
matches). `./run-on-cluster.sh --dry-run` prints what it will do; `./run-on-cluster.sh --recipe
run_grpo_qwen3_30b_a3b.sh` runs the MoE recipe. It only launches the recipe -- deploy the
RayCluster (step 5) first.

```bash
# Qwen3-4B, colocated (verified 1-node and 2-node paths):
bash recipe/run_grpo_qwen3_4b.sh

# Qwen3-30B-A3B MoE, colocated on 2 nodes / 16 GPU (trains cleanly: reward 0.578,
# repetition 0.0 -- see Known Issues item 2 for the one rollout-geometry constraint):
# uncomment the ALTERNATE block in env_vars, then launch. The shipped block runs the rollout
# MoE pure expert-parallel (moe_tp=1); for moe_tp>1 & moe_ep>1 the recipe disables the FlashInfer fusion.
bash recipe/run_grpo_qwen3_30b_a3b.sh
```

Monitor training:

```bash
# Ray dashboard (after port-forward)
open http://localhost:8265

# Follow Ray job logs, using the submission id the recipe printed when it submitted
# ("raysubmit_..."). `ray job list` renders a table for humans; parse it at your own risk.
ray job logs <submission-id> --address http://localhost:8265 --follow

# Monitor GPU utilization. This has to be a WORKER: the head runs with num-gpus 0 and
# NVIDIA_VISIBLE_DEVICES=void, so nvidia-smi there reports no devices.
W=$(kubectl get pod -n "${NAMESPACE}" -l ray.io/node-type=worker -o name | head -1)
kubectl exec -n "${NAMESPACE}" "${W#pod/}" -- nvidia-smi
```

Both recipes build the `train.py` argv as a bash array and submit it through
`recipe/launcher/grpo_launch.sh`, which sources the model script (`scripts/models/<model>.sh`)
and expands `MODEL_ARGS` in the same shell -- avoiding a shell-escaping trap where an outer
`ray job submit -- bash -c "..."` string would expand the array before it is defined. The job
is submitted with `--entrypoint-resources '{"gpu_node": 0.001}'` so the Ray driver lands on a
GPU worker rather than the (non-GPU) head; see
[miles-specific requirements](#miles-specific-requirements-found-on-real-hardware) for why.

### 8. Convert Checkpoints Back to HuggingFace Format

**Read this before running the training step, not after.** With the shipped values there is
nothing here to convert. `SAVE_INTERVAL=1000` sits deliberately above `NUM_ROLLOUT=100` so
that no save is ever triggered, because `save_model()` currently fails inside Megatron's
distributed-checkpoint save (Known Issues item 1) and a triggered save would end the run. The
consequence is that a run can finish `SUCCEEDED` after hours of training and leave **no
checkpoint at all** -- the trained weights are gone, and this step is where you would find
that out. The `iter_0060/` path below is therefore an illustration of the command's shape,
not a directory the default configuration produces.

This runs on the same GPU-worker pod as step 4, for the same reason: the script and `/fsx`
only exist there, not on your workstation.

Check first, and expect it to be empty on a default run:

```bash
W=$(kubectl get pod -n "${NAMESPACE}" -l ray.io/node-type=worker -o name | head -1)
W=${W#pod/}
kubectl exec -n "${NAMESPACE}" "${W}" -- ls /fsx/runs/qwen3-4b/ckpt/qwen3-4b-grpo/
```

```bash
kubectl cp scripts/convert_checkpoint.sh "${NAMESPACE}/${W}:/tmp/convert_checkpoint.sh"
kubectl exec -n "${NAMESPACE}" "${W}" -- \
  bash /tmp/convert_checkpoint.sh megatron2hf \
    --input-dir /fsx/runs/qwen3-4b/ckpt/qwen3-4b-grpo/iter_0060/ \
    --output-dir /fsx/models/Qwen3-4B-GRPO-step60 \
    --origin-hf-dir /fsx/models/Qwen3-4B
```

Lowering `SAVE_INTERVAL` to produce a checkpoint is what hits the save bug, so treat this
sample as a measurement and instrumentation harness rather than a route to trained weights
until that issue is resolved -- see [Known Issues](#known-issues).

## miles-specific requirements (found on real hardware)

Three issues surface on the miles base image (`nvidia/cuda`) that do **not** occur on
SLIME's NGC base. All three are fixed in this test case's `miles.Dockerfile` and manifests;
they are called out here because carrying an EFA layer or launch pattern from a different
base image verbatim will silently reintroduce them.

1. **CUDA compat shadowing kills `torch.cuda` (Error 803).** The miles base ships an older
   CUDA forward-compat `libcuda` than the actual node driver. CUDA forward-compat requires
   `compat >= host driver`; when the older bundled compat library is preferred via
   `LD_LIBRARY_PATH`, `torch.cuda.is_available()` dies with `Error 803: unsupported display
   driver / cuda driver combination`, and the SGLang engine fails at `get_device()`.
   `miles.Dockerfile` removes `/usr/local/cuda*/compat` outright and lets the host driver
   resolve instead. (SLIME's NGC base does not hit this because NGC's entrypoint only enables
   compat when `compat >= host driver`.)

2. **`libcuda.so.1` not found inside the SGLang subprocess.** With compat removed, `torch`
   still resolves `libcuda` via `ld.so.cache`, but the SGLang server subprocess (and
   Triton/cuda-python loaders, which scan `LD_LIBRARY_PATH` directly rather than the cache)
   fail with `ImportError: libcuda.so.1: cannot open shared object file`. Fix: append the
   driver-injection directories (`/usr/lib64`, `/usr/lib/x86_64-linux-gnu`) to
   `LD_LIBRARY_PATH` and register them with `ldconfig` -- appended, not prepended, so they are
   only consulted for libraries nothing else resolves.

3. **The Ray job driver dies on the head (no libcuda).** miles's training actor imports
   `mooncake` (P2P weight transfer, libcuda-dependent) at module load. The Ray job driver
   runs wherever the job is submitted to land; on a non-GPU head pod, the import dies with
   the same `libcuda.so.1` error. Fix: the GPU worker group declares a `gpu_node` custom Ray
   resource (`rayStartParams.resources: '{"gpu_node": 1}'`), and the recipe submits with
   `ray job submit --entrypoint-resources '{"gpu_node": 0.001}'`, landing the driver on a GPU
   worker without consuming a full GPU slot from the colocated placement group.

Full evidence and root-cause detail: [docs/PORT_NOTES.md](./docs/PORT_NOTES.md).

## Known Issues

1. **`save_model()` fails with `_pickle.UnpicklingError: pickle data was truncated`.** This
   occurs in Megatron's distributed checkpoint save (`gather_object`) after a training step
   completes; the GRPO loop itself is unaffected, but any run that must checkpoint --
   including the megatron2hf conversion in step 8 above -- is currently blocked. Workaround
   for short smoke runs: set `SAVE_INTERVAL` beyond the total step count so no save is
   triggered.

2. **The 30B MoE rollout degenerates only when SGLang runs the MoE tensor-parallel and
   expert-parallel at the same time (`moe_tp>1` and `moe_ep>1`).** With the shipped rollout
   geometry (pure expert-parallel, `moe_tp=1`) the 30B MoE trains cleanly -- `rollout/raw_reward`
   0.578, `rollout/repetition_frac` 0.0 -- so this is a rollout-configuration constraint, not a
   model or a general "SGLang expert parallelism" problem. The recipe enforces it (below).

   SGLang derives the rollout MoE geometry as `moe_ep = --sglang-expert-parallel-size` (EP_SIZE)
   and `moe_tp = --rollout-num-gpus-per-engine / EP_SIZE`. Serving the converted checkpoint
   directly from SGLang -- no miles, no Megatron, no GRPO -- and sweeping the engine's
   tensor-parallel size (TP) against EP isolates it. `repetition_frac` over 32 prompts, with
   the resulting `moe_tp` = TP/EP annotated:

   | engine TP \\ EP | EP=1 (`moe_ep`=1) | EP=2 | EP=4 | EP=8 |
   |---|---|---|---|---|
   | TP=1 | 0.000 (`moe_tp`=1) | (cannot start) | - | - |
   | TP=4 | 0.000 (`moe_tp`=4) | 0.875 (`moe_tp`=2) | - | - |
   | TP=8 | 0.000 (`moe_tp`=8) | 0.594 (`moe_tp`=4) | 0.844 (`moe_tp`=2) | ~0.0 (`moe_tp`=1) |

   Read by `moe_tp`, the pattern is exact: every clean cell has `moe_tp=1` (the whole EP=8
   column, pure expert-parallel) or `moe_ep=1` (the whole EP=1 column, pure tensor-parallel);
   every degenerate cell has both `moe_tp>1` and `moe_ep>1`. The earlier reading -- "EP>1
   degenerates" -- came from a sweep whose EP>1 cells all happened to have TP>EP, i.e.
   `moe_tp>1`; the pure expert-parallel case (EP=TP, the EP=8 column) was not in it. A stock
   `sglang.Engine` at `tp_size=8` confirms the missing column: `ep_size=8` (`moe_tp=1`)
   generates coherent text (4-gram repetition 0.009), while `ep_size=4` and `ep_size=2`
   (`moe_tp` 2 and 4) collapse to "7. 7. 7...", "1010...", ",,,,". Ruled out along the way: the
   model's own recommended sampling (temperature 0.6 / top_p 0.95 / top_k 20) does not change a
   degenerate cell, the `auto` vs `triton` MoE runner backend does not either, and all 18867
   expert weight keys are present in the checkpoint index with no NaN/Inf, so sampling, backend
   selection and conversion are not involved. `--check-weight-update-equal` passes
   (`weight_version` uniform, `mixed_version_ratio` 0.0), so trainer/rollout weights are not
   diverging.

   Root cause (`0.5.16.dev` in the image): the FlashInfer allreduce+RMSNorm fusion. On SM90/SM100
   it is auto-enabled for Qwen3-MoE (`tp_size>1`, no dp-attention, `moe_a2a=none`) without regard
   to `moe_ep`/`moe_tp`. With the fusion on, both post-experts all-reduces in
   `models/qwen3_moe.py` `forward_normal` are skipped and deferred to the next layer's fused
   `layernorm.forward_with_allreduce_fusion`. That fused reduce
   (`layernorm.py::_forward_with_allreduce_fusion`, `flashinfer_comm_fusion.py`) selects its group
   with `if moe_ep_size>1: use moe_ep_group else: use moe_tp_group` -- assuming the two are mutually
   exclusive. When both are `>1` it reduces over the moe-ep group only and never reduces the moe-tp
   group, so each rank keeps a partial sum over the intermediate dimension and generation collapses
   from layer 0. Pure expert-parallel (`moe_ep=tp`) and pure tensor-parallel (`moe_tp=tp`) are
   unaffected because `_MOE_EP`/`_MOE_TP` alias the full TP group, so either branch reduces over all
   ranks. Setting `enforce_disable_flashinfer_allreduce_fusion` on the same combined config restores
   the correct two-stage reduce and generation is clean (verified). The MoE weight sharding, the
   dispatcher's `local_expert_mapping`, and the moe-ep/moe-tp process groups are all correct; the
   defect is purely in the fused-reduce group selection. See
   [docs/VERIFICATION_LOG.md](./docs/VERIFICATION_LOG.md) "30B MoE root cause".

   The recipe handles it automatically: `run_grpo_qwen3_30b_a3b.sh` computes `moe_tp` from
   `ROLLOUT_GPUS_PER_ENGINE / EP_SIZE`, and when both `moe_tp>1` and `moe_ep>1` it adds
   `--sglang-enforce-disable-flashinfer-allreduce-fusion` so that geometry trains correctly too;
   pure expert-parallel (the shipped default) and pure tensor-parallel leave the fusion on. Losing
   the fusion costs some rollout throughput but not correctness. The underlying SGLang bug is a
   candidate for an upstream fix (generalise the fused-reduce group to the full TP group when
   `moe_tp>1` and `moe_ep>1`).

## File Structure

```
miles/                                    # 3.test_cases/pytorch/miles
├── README.md                            # This documentation
├── .gitignore
├── env_vars.colocated.example           # Base config: Qwen3-4B colocated (+ 30B MoE ALTERNATE block)
├── env_vars.disaggregated.example       # Overlay: reward model on a CPU pool + heavier GRPO
├── run-on-cluster.sh                    # Optional: run a recipe from the head pod (no local ray/port-forward)
├── miles.Dockerfile                     # radixark/miles base + EFA layer
├── requirements.txt                     # Reference Python deps (miles bundles these in the base image)
├── reward_service.Dockerfile            # CPU-only image for the remote reward service
├── reward_service/
│   ├── app.py                           # FastAPI reward server (reward_model / math_verify)
│   └── requirements.txt                 # Pinned CPU deps (no CUDA)
├── kubernetes/
│   ├── buildkit-job.yaml                # In-cluster image build -> registry (CPU-only)
│   ├── raycluster.yaml                  # KubeRay cluster manifest (head co-located on GPU pool, num-gpus 0)
│   ├── reward-service.yaml              # CPU reward service Deployment + Service
│   └── data-prep-pod.yaml               # Utility pod for data preparation
├── recipe/
│   ├── run_grpo_qwen3_4b.sh             # GRPO submit script (Qwen3-4B, colocated)
│   ├── run_grpo_qwen3_30b_a3b.sh        # GRPO submit script (Qwen3-30B-A3B MoE, colocated 16 GPU)
│   └── launcher/
│       └── grpo_launch.sh               # Ray job entrypoint: sources the model script, expands MODEL_ARGS, execs train.py
├── scripts/
│   ├── convert_checkpoint.sh            # HF <-> Megatron conversion helper
│   └── evaluate.sh                      # Evaluation launcher
└── docs/
    ├── EFA_2NODE.md                     # 2-node EFA/NCCL verification record
    ├── PORT_NOTES.md                    # miles-specific porting notes and the 3 hardware pitfalls
    └── VERIFICATION_LOG.md              # runs, job ids, flags, metrics (source of Verification Status)
```

## Training Configuration Deep Dive

### GRPO (Group Relative Policy Optimization)

GRPO is a critic-free RL algorithm that estimates advantages by comparing rewards within a
group of responses generated for the same prompt, eliminating the need for a separate value
model.

| Parameter | Description | Recipe Setting (4B) |
|-----------|-------------|----------------------|
| `--advantage-estimator grpo` | Use GRPO advantage estimation | GRPO |
| `--rollout-batch-size` | Prompts per rollout | 16 |
| `--n-samples-per-prompt` | Responses generated per prompt | 8 |
| `--global-batch-size` | Samples per optimizer step | 128 |
| `--num-steps-per-rollout` | Optimizer steps per rollout cycle | 1 |
| `--eps-clip` / `--eps-clip-high` | PPO-style clipping bounds | 0.2 / 0.28 |
| `--kl-loss-coef` | KL penalty coefficient | 0.0 |
| `--entropy-coef` | Entropy bonus coefficient | 0.0 |

The constraint `rollout_batch_size * n_samples_per_prompt == global_batch_size *
num_steps_per_rollout` (16 * 8 = 128 * 1) must always hold; the recipes validate every
required variable is set before submitting.

### Parallelism Strategy

**Qwen3-4B (colocated, verified on 1 and 2 nodes):**

```
Tensor Parallel (TP) = 1
Pipeline Parallel (PP) = 1
Context Parallel (CP) = 1
Expert Parallel (EP) = 1
Colocated: rollout-num-gpus == actor GPU count (8 on 1 node, 16 on 2 nodes)
```

**Qwen3-30B-A3B MoE (colocated on 2 nodes / 16 GPU, trains cleanly -- reward 0.578,
repetition 0.0; see [Known Issues](#known-issues) item 2 for the rollout-geometry constraint):**

```
Tensor Parallel (TP) = 2
Pipeline Parallel (PP) = 1
Context Parallel (CP) = 1
Expert Parallel (EP) = 2
Expert Tensor Parallel = 1
--use-distributed-optimizer   # required: shards the 30B optimizer state across all
                                # 16 GPU so static memory fits H200 (141 GB); confining
                                # the actor to 8 GPU (one node) OOMs
--colocate                     # rollout time-shares the same 16 GPU
--rollout-num-gpus-per-engine 2
--sglang-moe-runner-backend triton   # MoE online weight update requires the triton
                                       # runner; flashinfer is incompatible with the
                                       # in-place weight update on SGLang 0.5.12+
--sglang-expert-parallel-size 2
```

### Dynamic Batching

```bash
--use-dynamic-batch-size
--max-tokens-per-gpu 8192   # Per-GPU token budget per micro-batch
```

Variable-length responses (math solutions can range from under 100 to 8,000+ tokens) make
dynamic batching materially more efficient than a fixed micro-batch size.

## Reward Function

1. **Built-in rule-based rewards (default, verified).** miles ships several rule-based
   reward types -- `deepscaler`, `dapo`, `math`, `f1`, `gpqa` -- selected via `--rm-type`. The
   math types extract the `\boxed{...}` answer and grade it against the label using
   `math_verify` (LaTeX/sympy equivalence). The bundled DAPO-math recipe defaults to
   `RM_TYPE="deepscaler"` and needs no extra setup.

2. **Remote reward service on a CPU pool (UNVERIFIED on miles).** Set `RM_TYPE="remote_rm"`
   and `RM_URL` to offload scoring to a separate CPU instance group -- useful for a heavier
   reward such as a reward model, code execution, or RAG lookups. The service
   (`reward_service/app.py`) exposes a `reward_model` backend (HuggingFace sequence
   classifier on CPU) and a `math_verify` backend behind `POST /score`, plus `GET /health`.
   This path mirrors the sibling slime test case's manifests but has not been deployed or
   exercised on miles; miles's `remote_rm` client also lacks slime's retry-with-backoff, so
   treat it as a starting point rather than a validated path.

## Software Versions

| Component | Version |
|-----------|---------|
| miles | `radixark/miles` (fork point `fcce96ca0`, 2025-10-05) |
| Base image | `radixark/miles:dev-202607310056` @ `sha256:ca0bb593dd6f4011b444f64d478b72c213e4c70421f4d7f94e593a709562429e` |
| SGLang | 0.5.16.dev |
| Megatron-LM | radixark fork (miles-compatible) |
| Ray | 2.55.1 |
| CUDA | 13.0.1 |
| PyTorch | 2.11 |
| EFA installer | 1.48.0 |
| GDRCopy | v2.5.2 |

Pin the image tag to a dated/versioned value (e.g. `miles-cuda13-efa-0.1`); never use
`:latest`, per the [awsome-distributed-ai CONTRIBUTING guidelines](https://github.com/awslabs/awsome-distributed-ai/blob/main/CONTRIBUTING.md).

## Troubleshooting

**Pod stuck in `Pending` state**
```bash
kubectl describe pod <pod-name>
# Check for resource constraints -- GPU/EFA/memory/ephemeral-storage requests may
# exceed node capacity. The Ray head in particular needs enough ephemeral-storage
# to pull the ~18 GB image; a too-small root volume causes an Evict mid-pull rather
# than a clean Pending.
```

**Ray workers fail to connect to head node**
```bash
kubectl get svc miles-ray-head-svc
kubectl exec <worker-pod> -- nslookup miles-ray-head-svc
# Ensure RAY_memory_monitor_refresh_ms=0 is set (prevents OOM kills during init)
```

**NCCL/EFA initialization errors**
```bash
kubectl exec <pod> -- fi_info -p efa
kubectl exec <pod> -- env | grep NCCL
# Ensure FI_PROVIDER=efa and FI_EFA_USE_DEVICE_RDMA=1 are set
# If NCCL selects efa but every transfer times out (Error 15, "Unreachable remote"),
# check that the EFA security group has self-referencing all-traffic on BOTH
# ingress AND egress -- EFA's OS-bypass SRD traffic is not ordinary IP traffic, so
# a CIDR-only egress rule does not authorize it. See docs/EFA_2NODE.md.
```

**`ImportError: libcuda.so.1: cannot open shared object file`**
- Confirm the Ray job driver landed on a GPU worker (`--entrypoint-resources
  '{"gpu_node": 0.001}'`) and not the head.
- Confirm `LD_LIBRARY_PATH` includes the driver-injection directories
  (`/usr/lib64`, `/usr/lib/x86_64-linux-gnu`) appended at the end.
- See [miles-specific requirements](#miles-specific-requirements-found-on-real-hardware).

**`torch.cuda.is_available()` is `False` / `Error 803`**
- The image's CUDA forward-compat library is older than the node driver. Confirm
  `/usr/local/cuda*/compat` has been removed (as `miles.Dockerfile` does) rather than
  merely dropped from `LD_LIBRARY_PATH`.

**SGLang fails to start (CUDA OOM) in colocated mode**
- In colocated mode, SGLang launches after Megatron occupies GPU memory. Reduce
  `--sglang-mem-fraction-static` (0.8 for 4B; 0.75 was the value that completed for the 30B
  MoE colocated run -- the original 0.85 hardcode left too little room).

**MoE online weight update fails on SGLang 0.5.12+**
- Ensure `--sglang-moe-runner-backend triton` and `--sglang-expert-parallel-size` are both
  set explicitly; the default flashinfer MoE runner is incompatible with the in-place
  weight update used by GRPO training.

**Weight conversion fails**
- Ensure `PYTHONPATH` includes the Megatron-LM directory (`/root/Megatron-LM`).
- Verify model config parameters match (`--rotary-base`, `--vocab-size`, etc.).
- For MoE models, ensure `--expert-model-parallel-size` is set correctly.

**Checkpoint save fails with `_pickle.UnpicklingError: pickle data was truncated`**
- Known upstream issue in Megatron's distributed checkpoint save path; see
  [Known Issues](#known-issues). Raise `SAVE_INTERVAL` beyond the run length to avoid
  triggering a save on short smoke runs.

## References

- [miles GitHub Repository](https://github.com/radixark/miles)
- [SLIME GitHub Repository](https://github.com/THUDM/slime)
- [SLIME Blog: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/)
- [SGLang Project](https://github.com/sgl-project/sglang)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [GRPO Paper (DeepSeek-R1)](https://arxiv.org/abs/2402.03300)
- [Amazon SageMaker HyperPod Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html)
- [Sibling test case: 3.test_cases/pytorch/slime/](../slime/)
- [awsome-distributed-ai](https://github.com/awslabs/awsome-distributed-ai)
- [KubeRay Documentation](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)

## Security

See [CONTRIBUTING](https://github.com/awslabs/awsome-distributed-ai/blob/main/CONTRIBUTING.md) for more information.

## License

This sample code is made available under the MIT-0 license. See the LICENSE file.
