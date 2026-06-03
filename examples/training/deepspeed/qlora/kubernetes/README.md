# SageMaker HyperPod (EKS) Deployment

Instructions for deploying QLoRA training on SageMaker HyperPod with EKS orchestrator and DeepSpeed ZeRO-2 or ZeRO-3.

## Prerequisites

1. **SageMaker HyperPod EKS cluster** with the HyperPod Helm chart installed.
   The Helm chart bundles:
   - Kubeflow Training Operator
   - NVIDIA device plugin
   - Health monitoring agents (node health checks, deep health checks for GPU/NCCL)

   See [architectures/sagemaker-hyperpod-eks/](../../../../../architectures/sagemaker-hyperpod-eks/) for cluster setup.

2. **FSx for Lustre filesystem** provisioned in the same VPC as the cluster.
   Note the File System ID, DNS name, and Mount name from the AWS console.

3. **FSx CSI driver** installed (included in the HyperPod Helm chart).

4. **kubectl** configured to access the cluster

5. **ECR repository** for the training image

## Build and Push Docker Image

```bash
./0.build-image.sh
```

Or manually:

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1

aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

docker build -t ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/qwen3-qlora-training:latest .
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/qwen3-qlora-training:latest
```

## Deploy Training Job

Set the required environment variables and run the deploy script:

```bash
export IMAGE=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/qwen3-qlora-training:latest
export NUM_GPUS=4
export INSTANCE_TYPE=ml.g5.12xlarge
export FSX_FILESYSTEM_ID=fs-0123456789abcdef0
export FSX_DNS_NAME=fs-0123456789abcdef0.fsx.us-east-1.amazonaws.com
export FSX_MOUNT_NAME=abcdef01
```

### ZeRO-2, single node (Default)

```bash
./1.deploy-training.sh
```

### ZeRO-2, multi-node (2 nodes = 8 GPUs)

```bash
NUM_NODES=2 ./1.deploy-training.sh
```

### ZeRO-3

```bash
DEEPSPEED_STRATEGY=zero3 ./1.deploy-training.sh
```

The `NUM_NODES` variable controls how many nodes to use (default: 1). The deploy script
computes `NUM_WORKERS = NUM_NODES - 1` and creates 1 Master pod + N Worker pods.
Both ZeRO-2 and ZeRO-3 jobs can run side-by-side — they share the FSx PVC
(`fsx-claim-qlora`) but write to separate output directories.

## Storage

The `storage.yaml` manifest creates:

- **fsx-sc** StorageClass (FSx for Lustre CSI driver)
- **fsx-pv-qlora** PersistentVolume (static, binds to pre-existing FSx filesystem)
- **fsx-claim-qlora** PVC (ReadWriteMany) — shared by both ZeRO-2 and ZeRO-3 jobs

Storage is applied automatically when you run `1.deploy-training.sh`. FSx storage
survives node replacements, enabling HyperPod auto-resume to restart jobs on healthy
nodes without losing checkpoints.

## HyperPod Features

The PyTorchJob manifests include HyperPod-specific annotations and scheduling:

- **Auto-resume**: `sagemaker.amazonaws.com/enable-job-auto-resume: "true"` restarts jobs on healthy nodes after hardware failure
- **Health-aware scheduling**: pods only schedule on nodes with `node-health-status: Schedulable` and prefer nodes that pass deep health checks
- **Instance type selection**: `nodeSelector` targets `ml.` prefixed instance types via `${INSTANCE_TYPE}`

## MIG (Multi-Instance GPU) Support

NVIDIA MIG partitions a single physical GPU (A100, H100) into multiple isolated GPU instances. This is useful for QLoRA because:

- **Parallel experiments**: Run multiple QLoRA jobs with different hyperparameters on a single GPU
- **Resource efficiency**: QLoRA with a 4-bit quantized 8B model uses ~15 GB VRAM — an A100-80GB can run 2 experiments in parallel via MIG
- **Isolation**: Each MIG instance has hard memory isolation — one experiment cannot OOM another

### Prerequisites for MIG on EKS

1. **MIG-capable GPUs**: A100 or H100 instances (e.g., `ml.p4d.24xlarge`, `ml.p5.48xlarge`)
2. **MIG mode enabled** on the GPU nodes (done by cluster administrator)
3. **NVIDIA device plugin** configured with the correct MIG strategy:
   - **Single strategy** (default): All MIG instances on a node use the same profile. Each appears as `nvidia.com/gpu`. No manifest changes needed — use `NUM_GPUS` to request MIG instances.
   - **Mixed strategy**: Different MIG profiles can coexist on a node. Each profile gets its own resource name (e.g., `nvidia.com/mig-3g.20gb`). Set `MIG_PROFILE` in the deploy script.

### Deploying with MIG

```bash
# Single strategy (device plugin exposes MIG instances as nvidia.com/gpu):
# Just set NUM_GPUS to the number of MIG instances to use
export INSTANCE_TYPE=ml.p4d.24xlarge
NUM_GPUS=2 ./1.deploy-training.sh

# Mixed strategy (requires explicit MIG resource name):
MIG_PROFILE=3g.20gb NUM_GPUS=2 ./1.deploy-training.sh
```

When `MIG_PROFILE` is set, the deploy script automatically:
- Sets the GPU resource to `nvidia.com/mig-<profile>` (e.g., `nvidia.com/mig-3g.20gb`)
- Reduces pod CPU/memory requests to fit within a partition (40Gi RAM, 8 vCPU by default)
- Reduces shared memory (`/dev/shm`) to 16Gi

Override the defaults if needed:

```bash
MIG_PROFILE=4g.40gb NUM_GPUS=2 \
  POD_MEMORY_REQUEST=64Gi POD_MEMORY_LIMIT=80Gi \
  POD_CPU_REQUEST=16 POD_CPU_LIMIT=20 \
  ./1.deploy-training.sh
```

### Recommended MIG Profiles for QLoRA

| GPU | Profile | Memory | QLoRA Fit | Notes |
|-----|---------|--------|-----------|-------|
| A100 40GB | `3g.20gb` | 20 GB | Yes | 2 instances per GPU, recommended |
| A100 80GB | `4g.40gb` | 40 GB | Yes | 2 instances per GPU, comfortable headroom |
| A100 80GB | `3g.40gb` | 40 GB | Yes | Alternative partition |
| H100 80GB | `4g.40gb` | 40 GB | Yes | 2 instances per GPU |

### Enabling MIG on Nodes

MIG mode must be enabled by a cluster administrator. See the [Slurm MIG guide](../slurm/README.md#mig-configuration) for step-by-step instructions — the `nvidia-smi` commands are the same on EKS worker nodes. On HyperPod EKS, you can configure MIG in the [lifecycle scripts](../../../../../architectures/sagemaker-hyperpod-eks/) or via a DaemonSet that runs on GPU nodes.

## Monitor Training

### ZeRO-2

```bash
kubectl get pods -n ml-training -w
kubectl logs -f qwen3-qlora-training-zero2-master-0 -n ml-training
kubectl exec -it qwen3-qlora-training-zero2-master-0 -n ml-training -- nvidia-smi
```

### ZeRO-3

```bash
kubectl get pods -n ml-training -w
kubectl logs -f qwen3-qlora-training-zero3-master-0 -n ml-training
kubectl exec -it qwen3-qlora-training-zero3-master-0 -n ml-training -- nvidia-smi
```

## Cleanup

```bash
./2.cleanup.sh
```

Or manually:

```bash
kubectl delete pytorchjob qwen3-qlora-training-zero2 -n ml-training
kubectl delete pytorchjob qwen3-qlora-training-zero3 -n ml-training
```
