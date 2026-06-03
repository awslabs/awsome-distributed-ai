# NVRx Resiliency Testing for Distributed Training

This test case benchmarks [NVIDIA Resiliency Extension (NVRx)](https://github.com/NVIDIA/nvidia-resiliency-ext) features for PyTorch distributed training on AWS. NVRx provides fault tolerance and resiliency primitives for large-scale training, integrating as a simple `pip install` layer on top of standard PyTorch -- no custom CUDA builds or kernel modifications required.

## NVRx Features Tested

| Feature | Training Script | Description | Tested On |
|---------|----------------|-------------|-----------|
| Async Checkpointing | `src/train_async_ckpt.py` | Non-blocking checkpoint saves that overlap with training | LLaMA-3.1-8B on p4de, p5 |
| In-Process Restart | `src/train_inprocess.py` | Recover from faults without restarting the container | GPT-2 on g5; LLaMA-3.1-8B on p4de, p5 |
| ft_launcher (In-Job Restart) | `src/train_ft_launcher.py` | Automatic fault detection and worker respawn | GPT-2 on g5; LLaMA-3.1-8B on p5 |
| ft_launcher + In-Process | `src/train_ft_launcher.py --inprocess` | Combined: in-process for fast faults, ft_launcher for hard faults | GPT-2 on g5 |
| Local Checkpointing | `src/train_local_ckpt.py` | Node-local checkpoint storage for faster writes | GPT-2 on g5 |
| Baseline (K8s restart) | `src/train_inprocess.py --disable_nvrx_wrapper` | No NVRx -- relies on K8s container restart for comparison | GPT-2 on g5; LLaMA-3.1-8B on p4de, p5 |

> **Note:** Local checkpointing with NVRx `LocalCheckpointManager` has been tested with GPT-2 on g5 instances only. FSDP sharded state dicts (ShardedTensor) are not yet compatible with NVRx `BasicTensorAwareStateDict` on larger models.

## Parallelism Support

All training scripts support both **FSDP** and **DDP** via the `--parallel_strategy` argument:
- `--parallel_strategy=fsdp` (default) -- Full Sharded Data Parallel, recommended for large models (LLaMA-3.1-8B)
- `--parallel_strategy=ddp` -- Distributed Data Parallel, suitable for smaller models (GPT-2)

## Model Support

| Model | Recommended Instances | Parallelism | Notes |
|-------|----------------------|-------------|-------|
| **GPT-2 (124M)** | g5.8xlarge (A10G) | DDP or FSDP | Fast iteration for testing NVRx features. Small model, single GPU per node. |
| **LLaMA-3.1-8B** | p4de.24xlarge (A100), p5.48xlarge (H100) | FSDP | Realistic distributed training workload. FSDP across 16+ GPUs with EFA. |

Set the model via `MODEL_NAME` in `env_vars` (e.g., `MODEL_NAME=gpt2` or `MODEL_NAME=meta-llama/Llama-3.1-8B`).

## Fault Injection

The test case includes a configurable fault injection framework (`src/failure_simulator.py`) that simulates real-world training failures:

- **Exception faults** -- `RuntimeError` caught by NVRx in-process Wrapper for sub-second recovery
- **Hang faults** -- Simulates NCCL deadlocks, detected by ft_launcher heartbeat timeout

Two injection modes are supported:
- **Deterministic** (recommended for benchmarks): `--fault_count=5 --fault_seed=42` injects exactly N faults at pre-determined steps for reproducible, apples-to-apples comparisons across recovery mechanisms
- **Stochastic** (for exploratory testing): `--fault_probability=0.005` injects faults randomly per step per rank

## Prerequisites

To run these tests, you need a training cluster with GPU nodes and shared storage. Instructions for creating a cluster can be found in [architectures](../../../architectures), the [aws-do-eks](https://bit.ly/do-eks) project, or [EKS Blueprints](https://github.com/aws-ia/terraform-aws-eks-blueprints).

## 1. Build Container Image

From the `nvrx/` directory, build a container image with PyTorch 2.9, NVRx 0.4.1, and the training scripts:

```bash
cd examples/training/nvrx

export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com

docker build -f Dockerfile -t ${REGISTRY}/nvrx-fsdp-training:latest .
```

## 2. Push Container Image to Amazon ECR

```bash
# Create registry if needed
REGISTRY_COUNT=$(aws ecr describe-repositories | grep \"nvrx-fsdp-training\" | wc -l)
if [ "$REGISTRY_COUNT" == "0" ]; then
    aws ecr create-repository --repository-name nvrx-fsdp-training
fi

# Login and push
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY
docker image push ${REGISTRY}/nvrx-fsdp-training:latest
```

Alternatively, use the provided `build_and_push.sh` script which handles ECR login, repository creation, and pushing in one step:

```bash
cp env_vars.template env_vars
# Edit env_vars with your settings
source env_vars
./build_and_push.sh
```

Set `BUILD_HOST` in `env_vars` to build remotely via SSH (e.g., on an EC2 instance with Docker). Leave it empty for local builds.

## 3. Prepare Dataset (Optional)

Training scripts support two dataset modes:

**Option A: Stream from HuggingFace Hub (default)** -- No setup required. Works well for experiments without frequent restarts.

**Option B: Pre-download to local storage (recommended for fault recovery experiments)** -- Eliminates HuggingFace API rate limiting (429 errors) during rapid restarts:

```bash
# Run inside the training container on your cluster
python prepare_dataset.py --output_path /checkpoints/c4_subset --num_samples 100000
```

This downloads 100K samples (~227 MB) in about 20 seconds. Then add `--dataset_path=/checkpoints/c4_subset` to the training script arguments. See the platform-specific README for detailed instructions on running this on your cluster.

## Platform-Specific Instructions

- **Amazon EKS**: See [kubernetes/README.md](kubernetes/README.md)
- **Slurm**: Coming soon

## Source Files

| File | Purpose |
|------|---------|
| `src/train_inprocess.py` | In-process restart with NVRx Wrapper + baseline mode (`--disable_nvrx_wrapper`) |
| `src/train_ft_launcher.py` | ft_launcher in-job restart + combined mode (`--inprocess`) |
| `src/train_async_ckpt.py` | Async vs sync checkpointing comparison (`--use_async_checkpoint`) |
| `src/train_local_ckpt.py` | Local checkpointing comparison (GPT-2 on g5 only) |
| `src/distributed_utils.py` | Shared utilities: model creation, FSDP/DDP wrapping, DCP checkpoint save/load |
| `src/failure_simulator.py` | Configurable fault injection framework (deterministic + stochastic) |
| `src/fsdp_config.py` | FSDP configuration for GPT-2 and LLaMA models |
| `src/metrics_collector.py` | Training metrics collection and JSON persistence |
| `prepare_dataset.py` | One-time dataset download utility |

## References

- [NVIDIA Resiliency Extension (NVRx) GitHub](https://github.com/NVIDIA/nvidia-resiliency-ext)
- [NVRx Documentation](https://nvidia.github.io/nvidia-resiliency-ext/)
- [NVRx on PyPI](https://pypi.org/project/nvidia-resiliency-ext/)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Amazon EKS User Guide](https://docs.aws.amazon.com/eks/latest/userguide/)
- [AWS EFA Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
