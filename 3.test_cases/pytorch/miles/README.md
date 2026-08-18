# Reinforcement Learning with miles on Amazon SageMaker HyperPod EKS

This test case runs GRPO post-training with [**miles**](https://github.com/radixark/miles) on [Amazon SageMaker HyperPod](https://aws.amazon.com/sagemaker/hyperpod/) with Amazon EKS orchestration, or on a plain Amazon EKS cluster. It mirrors the sibling [`3.test_cases/pytorch/slime/`](../slime/) test case: build a container image, prepare data, deploy a multi-node Ray cluster, convert model checkpoints, and launch GRPO training on NVIDIA GPUs interconnected with Elastic Fabric Adapter networking, EFA. miles is a fork of SLIME built for CUDA 13 and NVIDIA Blackwell as first-class hardware.

## Introduction

[**miles**](https://github.com/radixark/miles) is a fork of [**SLIME**](https://github.com/THUDM/slime), an LLM post-training framework for RL scaling. It keeps SLIME's design of two backends under Ray orchestration:

- **[SGLang](https://github.com/sgl-project/sglang)** for high-throughput rollout generation, providing RadixAttention, continuous batching, and tensor parallelism.
- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)**, a radixark fork, for scalable distributed training with TP, PP, CP, EP, and ZeRO-style sharding.

**Ray** manages resource orchestration and supports two deployment topologies, the same two SLIME supports. In colocated mode the training actors and the rollout engines share one GPU pool. In disaggregated mode, selected with `COLOCATE=false`, the training actors and the rollout engines run on separate GPU pools and synchronize weights over NCCL/EFA. A third, independent choice moves reward scoring to CPU nodes and is configured by `env_vars.disaggregated.example`; it can be combined with either GPU topology.

### Why miles on HyperPod?

This test case adds a miles reference for GRPO post-training and keeps it structurally close to the sibling slime case, so both can be used the same way; it is not a recommendation of one over the other. The table below records how miles differs from slime, with a source link for each row. The slime column is the sibling AWS test case `3.test_cases/pytorch/slime` at commit [`63becdd3`](https://github.com/awslabs/awsome-distributed-ai/tree/63becdd38d0a047a4ec79acad893a6652df1bcbe/3.test_cases/pytorch/slime), on THUDM/slime `v0.2.4`; the miles column is [`radixark/miles`](https://github.com/radixark/miles/tree/fc04f666c08aebd72b241cef586a3939fdd6fa8e) at commit `fc04f66`.

| Aspect | slime — sibling test case, THUDM/slime v0.2.4 | miles — radixark/miles |
|--------|-------|-------|
| Base image | NGC PyTorch container [`nvcr.io/nvidia/pytorch:26.02-py3`](https://github.com/awslabs/awsome-distributed-ai/blob/63becdd38d0a047a4ec79acad893a6652df1bcbe/3.test_cases/pytorch/slime/slime.Dockerfile#L8) | [`radixark/miles`](https://github.com/radixark/miles/blob/fc04f666c08aebd72b241cef586a3939fdd6fa8e/docker/Dockerfile#L14-L23), built on `lmsysorg/sglang:v0.5.16` with [torch 2.11.0 / CUDA 13.0.1](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/docker/Dockerfile#L1) |
| SGLang | [`0.5.12.post1`](https://github.com/awslabs/awsome-distributed-ai/blob/63becdd38d0a047a4ec79acad893a6652df1bcbe/3.test_cases/pytorch/slime/slime.Dockerfile#L20) | the [`sglang-miles`](https://github.com/radixark/miles/blob/fc04f666c08aebd72b241cef586a3939fdd6fa8e/docker/Dockerfile#L14) branch, a development build based on `v0.5.16` |
| Megatron fork | [NVIDIA/Megatron-LM](https://github.com/awslabs/awsome-distributed-ai/blob/63becdd38d0a047a4ec79acad893a6652df1bcbe/3.test_cases/pytorch/slime/slime.Dockerfile#L18) `@3714d81` | [radixark/Megatron-LM](https://github.com/radixark/miles/blob/fc04f666c08aebd72b241cef586a3939fdd6fa8e/docker/Dockerfile#L22-L23) `miles-main` |
| Framework install path | [`/opt/slime`](https://github.com/awslabs/awsome-distributed-ai/blob/63becdd38d0a047a4ec79acad893a6652df1bcbe/3.test_cases/pytorch/slime/slime.Dockerfile#L189) | [`/root/miles`](https://github.com/radixark/miles/blob/fc04f666c08aebd72b241cef586a3939fdd6fa8e/docker/Dockerfile#L238) |
| Weight-sync path | `SGLangEngine(RayActor)` methods called over Ray, each issuing an HTTP request to its local SGLang server | same design; miles adds a two-phase [`begin_weight_update` / `end_weight_update`](https://github.com/radixark/miles/blob/fc04f666c08aebd72b241cef586a3939fdd6fa8e/miles/backends/sglang_utils/sglang_engine.py#L636) session on top |
| Relationship | upstream | fork of SLIME, [shared ancestor commit `fcce96ca0`](https://github.com/radixark/miles/commit/fcce96ca06e47e92f685db91c2a7327fff095906), dated 2025-10-06 |

The `train.py` CLI and GRPO flags are compatible with SLIME's, so the recipes here are drop-in and the [Quick Start](#quick-start) tracks the sibling test case step for step.

## Architecture

```
+---------------------------------------------------------------------+
|             Amazon EKS Cluster, HyperPod-compatible                 |
|                 2x p5en.48xlarge, EKS orchestration                 |
|                                                                     |
|  +----------------------------+  +----------------------------+     |
|  |  Node 1: p5en.48xlarge     |  |  Node 2: p5en.48xlarge     |     |
|  |  8x H200 141GB   16x EFA   |  |  8x H200 141GB   16x EFA   |     |
|  |  ~2TB RAM        192 vCPU  |  |  ~2TB RAM        192 vCPU  |     |
|  +----------------------------+  +----------------------------+     |
|            |                                  |                     |
|            +----- EFA 3200 Gbps/node, GPU<->GPU RDMA -----+         |
|                                                                     |
|  +---------------------------------------------------------+        |
|  |          FSx for Lustre, RWX, mounted at /fsx           |        |
|  |  /fsx/models   /fsx/data   /fsx/runs                    |        |
|  +---------------------------------------------------------+        |
|                                                                     |
|  Kubernetes Resources:                                              |
|  - KubeRay operator, kuberay-operator namespace                     |
|  - EFA device plugin, kube-system                                   |
|  - FSx CSI driver, kube-system                                      |
|  - NVIDIA device plugin, kube-system                                |
+---------------------------------------------------------------------+
```

The diagram shows the 2-node layout. A single node is also supported for the dense 4B case; the multi-node layout adds the second node for MoE and disaggregated runs.

## miles Internal Architecture

```
                    +----------------------+
                    |     Data Buffer      |
                    |  prompt queue and    |
                    |  rollout cache       |
                    +----------+-----------+
                               |
              +----------------+-----------------+
              |                                  |
              v                                  v
   +-----------------------+      +-----------------------+
   |        Rollout        |      |       Training        |
   |  SGLang Ray actors    |      |  Megatron-LM          |
   |  RadixAttention       |      |  TP / PP / CP / EP    |
   |  continuous batching  |      |  GRPO, dynamic batch  |
   +-----------------------+      +-----------------------+
              ^                                  ^
              |           weight sync            |
              +----------------------------------+
```

The Data Buffer manages prompts, dispatches them for rollout, and stores generated samples with rewards. Rollout runs SGLang engines as Ray actors, generating responses and scoring them with a reward function. Training reads batches from the buffer, computes GRPO advantages, updates the policy with Megatron-LM, and syncs the updated weights back to the rollout engines.

## Hardware Requirements

This test case is validated on the following configuration:

| Component | Specification |
|-----------|---------------|
| **Instance type** | p5en.48xlarge |
| **Nodes** | 1 for single-node, or 2 for multi-node and MoE |
| **GPUs per node** | 8x NVIDIA H200 141GB |
| **Total GPUs** | 8 on one node, 16 on two nodes |
| **GPU memory** | 1,128 GB per node aggregate |
| **Host RAM per node** | ~2 TB |
| **EFA per node** | 16 devices, 15 allocatable on EKS |
| **Storage** | FSx for Lustre, RWX, mounted at `/fsx` |
| **Kubernetes** | EKS, KubeRay operator |

Other instance types are expected to work with resource-value retuning. p6-b300.48xlarge, a Blackwell instance, is expected-compatible because miles's base image targets CUDA 13 / sm_103 and nothing here hard-codes a GPU generation, but it has not been verified.

## Supported Model Sizes

The TP and PP columns describe Megatron training-side parallelism, which is distinct from the SGLang rollout MoE geometry `moe_tp` / `moe_ep`. The shipped 30B MoE recipe runs the rollout pure expert-parallel, `moe_tp=1`; see [Known Issues](#known-issues).

| Model | Parameters | Topology | TP | PP | Rollout GPUs | Training GPUs |
|-------|-----------|----------|----|----|-------------|---------------|
| Qwen3-4B | 4B Dense | Colocated | 1 | 1 | 8, shared | 8, shared |
| GLM-Z1-9B | 9B Dense | Colocated | 2 | 1 | 16, shared | 16, shared |
| Qwen3-30B-A3B | 30B MoE | Colocated | 2 | 1 | 16, shared | 16, shared |
| Qwen2.5-72B * | 72B Dense | Disaggregated | 4 | 2 | 8 | 8 |

\* Qwen2.5-72B does not fit this 16-GPU H200 layout; see the Validation table and Known Issues.

## Validation

Each configuration below was launched on 2x p5en.48xlarge and confirmed to complete within the listed wall time. Reward and repetition are the last scalar from the trainer's TensorBoard event files; they indicate the loop closes and generation is healthy, not convergence. Known limitations are the 30B MoE rollout degenerating in one specific parallelism geometry and Qwen2.5-72B not fitting the 16-GPU H200 layout; both are in [Known Issues](#known-issues).

| Config | reward | repetition | wall time |
|--------|--------|------------|-----------|
| Qwen3-4B dense, colocated 1 node | 0.531 | 0.0 | ~13 min |
| Qwen3-4B dense, disaggregated 2 nodes | 0.523 | 0.0 | ~12 min |
| GLM-Z1-9B dense, colocated TP2 | 0.680 | 0.0 | ~13 min |
| Qwen3-30B-A3B MoE, colocated pure EP, `moe_tp=1` | 0.578 | 0.0 | ~21 min |
| Qwen3-30B-A3B MoE, colocated pure TP, `moe_ep=1` | 0.531 | 0.0 | ~25 min |
| Qwen3-30B-A3B MoE, disaggregated pure EP | 0.65 | 0.0 | ~18 min |
| Qwen3-30B-A3B MoE, colocated combined `moe_tp=2` x `moe_ep=2` | 0.555 | 0.0 | ~21 min |
| Qwen3-30B-A3B MoE, combined geometry, FlashInfer fusion on | 0.0 | 0.56 | ~20 min, degenerate output |
| Qwen2.5-72B dense, disaggregated | OOM | -- | did not fit 16x H200 |

## Prerequisites

1. A SageMaker HyperPod EKS cluster, or a plain EKS cluster, with a p5en.48xlarge GPU instance group and EFA. Validated on p5en.48xlarge; other instance types may need resource-value retuning.
2. An FSx for Lustre `PersistentVolumeClaim` mounted at `/fsx`. The claim name is set through `FSX_CLAIM` and defaults to `fsx-claim`, matching the sibling slime test case.
3. Amazon ECR access for building and pushing the image.
4. A Hugging Face account and access token for model downloads.

The KubeRay operator is installed in step 0 below.

## Quick Start

The default path is Qwen3-4B dense, colocated on one node: it builds the image, prepares the model and data, deploys a Ray cluster, and runs a short GRPO loop. To run the 30B MoE case instead, uncomment the `ALTERNATE` block in `env_vars.colocated.example` and launch the MoE recipe in step 7.

### 0. Install the KubeRay Operator, once per cluster

If `kubectl get crd rayclusters.ray.io` returns nothing, install it with Helm:

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
# Edit env_vars for your cluster, then:
source env_vars
```

`env_vars` includes `NAMESPACE`, which must already exist. The Hugging Face token is read from a Kubernetes Secret named `hf-token`, wired into the pods with `secretKeyRef`, so create it once per namespace. This is the same flow as the sibling slime test case:

```bash
kubectl create secret generic hf-token --from-literal=HF_TOKEN=hf_xxx -n "${NAMESPACE}"
# Public model with no token: create it with an empty value so the key exists.
```

### 2. Build and Push the Container Image

The image takes `radixark/miles` as its base and adds only the AWS EFA layer. The base is pinned by `sha256` digest in `miles.Dockerfile`.

```bash
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${REGISTRY}
aws ecr create-repository --repository-name ${IMAGE} --region ${AWS_REGION} || true
docker build -t ${FULL_IMAGE} -f miles.Dockerfile .
docker push ${FULL_IMAGE}
```

If no node has a local Docker daemon, `kubernetes/buildkit-job.yaml` builds and pushes the image in-cluster with a rootless BuildKit Job:

```bash
kubectl create configmap miles-build-context --from-file=Dockerfile=miles.Dockerfile -n "${NAMESPACE}"
kubectl create secret docker-registry ecr-miles-push \
    --docker-server="${REGISTRY}" --docker-username=AWS \
    --docker-password="$(aws ecr get-login-password --region ${AWS_REGION})" -n "${NAMESPACE}"
envsubst < kubernetes/buildkit-job.yaml | kubectl apply -f -
```

### 3. Download and Prepare the Model

As in the sibling slime test case, this uses a data-prep pod and `huggingface-cli` inside it:

```bash
envsubst < kubernetes/data-prep-pod.yaml | kubectl apply -f -
kubectl wait --for=condition=Ready pod/data-prep -n "${NAMESPACE}" --timeout=300s
kubectl exec -it data-prep -n "${NAMESPACE}" -- bash
# Inside the pod:
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-4B --local-dir /fsx/models/Qwen3-4B
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /fsx/data/dapo-math-17k
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 --local-dir /fsx/data/aime-2024
```

### 4. Deploy the Ray Cluster

```bash
source env_vars
envsubst < kubernetes/raycluster.yaml | kubectl apply -f -
kubectl get pods -w -n "${NAMESPACE}" -l ray.io/is-ray-node=yes   # Ctrl-C once head and workers are Running
kubectl port-forward -n "${NAMESPACE}" svc/miles-ray-head-svc 8265:8265 &
```

The manifest schedules GPU workers on the GPU pool and the Ray head on the same pool. The head uses no GPU but must sit on a CUDA-capable node; see [Known Issues](#known-issues).

### 5. Convert Model Weights to Megatron Format

miles's Megatron backend needs weights in `torch_dist` format. Run this on a GPU worker pod, which now exists after step 4. The conversion runs single-process by default; set `CONVERT_NUM_GPUS` or pass `--num-gpus` to parallelize it with `torchrun`:

```bash
W=$(kubectl get pod -n "${NAMESPACE}" -l ray.io/node-type=worker -o name | head -1); W=${W#pod/}
kubectl cp scripts/convert_checkpoint.sh "${NAMESPACE}/${W}:/tmp/convert_checkpoint.sh"
kubectl exec -n "${NAMESPACE}" "${W}" -- bash /tmp/convert_checkpoint.sh hf2megatron \
    --model-script qwen3-4B.sh --hf-path /fsx/models/Qwen3-4B --save-path /fsx/models/Qwen3-4B_torch_dist
```

### 6. Configure the Reward

Pick a reward strategy in `env_vars`. The default is the built-in rule-based reward `RM_TYPE="deepscaler"`, with `dapo`, `math`, `f1`, and `gpqa` also available. It runs in-process on the rollout actors and needs no extra setup. A remote reward service on a CPU pool is available with `RM_TYPE="remote_rm"` and `RM_URL`, mirroring the sibling slime test case; for that path, deploy `kubernetes/reward-service.yaml` first.

### 7. Launch GRPO Training

Run the recipe from a machine with a matching Ray CLI and the dashboard port-forwarded, as in step 4, or use `./run-on-cluster.sh` to launch from inside the head pod with only `kubectl`.

```bash
# Qwen3-4B, colocated:
bash recipe/run_grpo_qwen3_4b.sh
# Qwen3-30B-A3B MoE, colocated on 2 nodes: uncomment the ALTERNATE block in env_vars first.
bash recipe/run_grpo_qwen3_30b_a3b.sh
```

Monitor with the Ray dashboard at `http://localhost:8265`, or follow the job log with `ray job logs <submission-id> --address http://localhost:8265 --follow`.

### 8. Convert Checkpoints Back to HuggingFace Format

A run writes a Megatron `torch_dist` checkpoint to `CHECKPOINT_DIR` at the end of training. To evaluate the trained weights outside miles, convert them back to HuggingFace format on a GPU worker pod:

```bash
W=$(kubectl get pod -n "${NAMESPACE}" -l ray.io/node-type=worker -o name | head -1); W=${W#pod/}
kubectl cp scripts/convert_checkpoint.sh "${NAMESPACE}/${W}:/tmp/convert_checkpoint.sh"
# Replace iter_NNNN with the actual iteration:
#   kubectl exec -n "${NAMESPACE}" "${W}" -- ls /fsx/runs/qwen3-4b/ckpt/qwen3-4b-grpo/
kubectl exec -n "${NAMESPACE}" "${W}" -- bash /tmp/convert_checkpoint.sh megatron2hf \
    --input-dir /fsx/runs/qwen3-4b/ckpt/qwen3-4b-grpo/iter_NNNN/ \
    --output-dir /fsx/models/Qwen3-4B-GRPO --origin-hf-dir /fsx/models/Qwen3-4B
```

## miles-specific requirements

Three items surface on the miles base image that do not occur on the sibling slime NGC base. All three are handled in `miles.Dockerfile` and the manifests; the inline comments at each fix carry the detail.

- The base image's CUDA forward-compat `libcuda` is older than the node driver, so `miles.Dockerfile` removes `/usr/local/cuda*/compat` and uses the host driver.
- The SGLang subprocess resolves `libcuda.so.1` only after the driver-injection directories are appended to `LD_LIBRARY_PATH` and registered with `ldconfig`.
- miles's Ray control actors import Megatron at startup and link `libcuda.so.1`, so they must land on a CUDA-capable node. The manifest keeps every Ray node on the GPU pool and declares a `gpu_node` custom resource so the job driver lands on a GPU worker.

miles adds the sm_103 Transformer Engine FA2 whitelist patch that the sibling slime image reached differently, because the miles base is not the NGC image. The residual patches slime carries for CUDA 13 and Blackwell are tracked in [awsome-distributed-ai issue #1163](https://github.com/awslabs/awsome-distributed-ai/issues/1163); miles applies the equivalent set through its own image build.

## Known Issues

1. **The 30B MoE rollout degenerates when SGLang runs the MoE tensor-parallel and expert-parallel at the same time**, that is when `moe_tp>1` and `moe_ep>1`. Pure expert-parallel with `moe_tp=1`, the shipped default, and pure tensor-parallel with `moe_ep=1` both train cleanly. This is an upstream SGLang bug in the FlashInfer allreduce+RMSNorm fusion, tracked and being fixed upstream in sgl-project/sglang PRs [#32963](https://github.com/sgl-project/sglang/pull/32963), [#32511](https://github.com/sgl-project/sglang/pull/32511), and [#32012](https://github.com/sgl-project/sglang/pull/32012). Disabling the fusion with `enforce_disable_flashinfer_allreduce_fusion` restores clean generation, with 4-gram repetition at 0.006 on par with pure EP, and the recipe applies this automatically for the combined geometry. The workaround stays correct after upstream ships the fix; once you move to a fixed build you can drop the flag to regain the fusion's throughput.

2. **Qwen2.5-72B does not fit the 16-GPU H200 layout.** The disaggregated TP4 PP2 configuration runs out of memory on 2x p5en.48xlarge. It needs a larger cluster or optimizer and activation offload, neither of which has been run here.

3. **The Ray head must run on a CUDA-capable node.** miles's control actors link `libcuda.so.1` at import even with `num-gpus 0`, so the manifest places the head on the GPU pool rather than a CPU node. Requiring a GPU-capable node for a zero-GPU control process is an upstream limitation in miles, not a constraint introduced by this test case.

## Reward Function

miles ships rule-based reward types `deepscaler`, `dapo`, `math`, `f1`, and `gpqa`, selected with `--rm-type`. The math types extract the `\boxed{...}` answer and grade it with `math_verify`. A remote reward service on a CPU pool is available with `RM_TYPE="remote_rm"`, mirroring the sibling slime test case.

## File Structure

```
miles/
├── README.md
├── env_vars.colocated.example       # Qwen3-4B colocated, plus a 30B MoE ALTERNATE block
├── env_vars.disaggregated.example   # overlay: reward model on a CPU pool
├── run-on-cluster.sh                # launch a recipe from inside the head pod
├── miles.Dockerfile                 # radixark/miles base + AWS EFA layer
├── reward_service.Dockerfile
├── reward_service/                  # FastAPI reward app, CPU
├── kubernetes/
│   ├── raycluster.yaml              # KubeRay cluster manifest
│   ├── buildkit-job.yaml            # in-cluster image build
│   ├── data-prep-pod.yaml           # model and data download
│   └── reward-service.yaml          # CPU reward service
├── recipe/                          # GRPO recipes and launcher
└── scripts/                         # convert_checkpoint.sh, evaluate.sh
```

## Software Versions

| Component | Version |
|-----------|---------|
| miles | `radixark/miles` at commit `fc04f66` |
| Base image | `radixark/miles`, pinned by `sha256` digest in `miles.Dockerfile` |
| SGLang | `sglang-miles` branch, based on `v0.5.16` |
| Megatron-LM | radixark fork, `miles-main` |
| Ray | 2.55.1 |
| CUDA | 13.0.1 |
| PyTorch | 2.11.0 |
| EFA installer | 1.48.0 |
| GDRCopy | v2.5.2 |

## Troubleshooting

**Pod stuck in `Pending`**
```bash
kubectl describe pod <pod>
# Check GPU/EFA/memory/ephemeral-storage requests against node capacity.
# The head needs enough ephemeral-storage to pull the ~18 GB image.
```

**Ray workers cannot connect to the head**
```bash
kubectl get svc miles-ray-head-svc -n "${NAMESPACE}"
kubectl exec <worker-pod> -n "${NAMESPACE}" -- nslookup miles-ray-head-svc
```
Separately, if workers are killed mid-run rather than failing to connect, ensure `RAY_memory_monitor_refresh_ms=0` is set to disable the memory monitor.

**NCCL/EFA errors**
```bash
kubectl exec <pod> -- fi_info -p efa
kubectl exec <pod> -- env | grep NCCL
# Ensure FI_PROVIDER=efa. For multi-node, the EFA nodes' security group must
# allow all traffic to itself on both ingress and egress, since EFA SRD is not IP.
```

**`ImportError: libcuda.so.1`** — the Ray job driver or a control actor landed on a node without the driver; see miles-specific requirements.

**`torch.cuda.is_available()` is `False` or `Error 803`**
```bash
kubectl exec <pod> -- ls /usr/local/cuda*/compat   # expect: not found
# The base image's CUDA forward-compat library was older than the node driver;
# miles.Dockerfile removes it.
```

**SGLang out-of-memory in colocated mode** — lower `--sglang-mem-fraction-static` to 0.8 for 4B or 0.75 for the 30B MoE colocated run.

## References

- [miles](https://github.com/radixark/miles)
- [SLIME](https://github.com/THUDM/slime)
- [SGLang](https://github.com/sgl-project/sglang)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [GRPO paper](https://arxiv.org/abs/2402.03300)
- [Amazon SageMaker HyperPod documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html)
- [KubeRay documentation](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
- [Sibling test case: 3.test_cases/pytorch/slime/](../slime/)
- [awsome-distributed-ai](https://github.com/awslabs/awsome-distributed-ai)

## Security

See [CONTRIBUTING](https://github.com/awslabs/awsome-distributed-ai/blob/main/CONTRIBUTING.md).

## License

This sample code is made available under the MIT-0 license. See the LICENSE file.
