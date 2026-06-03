# QLoRA Fine-tuning on SageMaker HyperPod (Slurm)

This guide covers deploying the Qwen3-8B QLoRA training job on a **SageMaker HyperPod** cluster managed by Slurm, including setup for **NVIDIA MIG** (Multi-Instance GPU) partitioning.

For the EKS/Kubernetes deployment path, see the [top-level README](../README.md).

## Prerequisites

1. **SageMaker HyperPod cluster** with GPU worker nodes provisioned and running.
   See [`architectures/sagemaker-hyperpod-slurm/`](https://github.com/awslabs/awsome-distributed-ai/tree/main/architectures/sagemaker-hyperpod-slurm) for cluster setup.

2. **Shared filesystem** — HyperPod clusters use Amazon FSx for Lustre mounted at `/fsx`.

3. **GPU instance types** (recommended):

   | Instance | GPUs | GPU Type | VRAM per GPU | EFA | Notes |
   |----------|------|----------|-------------|-----|-------|
   | g5.12xlarge | 4 | A10G | 24 GB | No | Budget option, sufficient for QLoRA |
   | p4d.24xlarge | 8 | A100 40GB | 40 GB | Yes | MIG supported (up to 7 instances per GPU) |
   | p5.48xlarge | 8 | H100 80GB | 80 GB | Yes | MIG supported, fastest training |

4. **SSH access** to the cluster head node (or use SSM Session Manager).

## Environment Setup

### Option A: Python Virtual Environment on /fsx (Recommended)

```bash
# SSH to the head node (or use SSM Session Manager)
ssh ubuntu@<head-node-ip>

# Clone the repo to shared storage
cd /fsx
git clone https://github.com/awslabs/awsome-distributed-ai.git
cd awsome-distributed-ai/examples/training/deepspeed/qlora

# Install the venv package (not pre-installed on HyperPod AMI)
sudo apt-get update && sudo apt-get install -y python3.10-venv

# Create a virtual environment on shared storage (visible to all nodes)
python3 -m venv /fsx/venvs/qwen3-qlora
source /fsx/venvs/qwen3-qlora/bin/activate

# Install PyTorch (cu126 is a safe default for CUDA 12.8 hosts)
pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu126

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note**: We use `torch+cu126` as a conservative default because it is
> forward-compatible with CUDA 12.8 drivers and avoids potential CUBLAS errors
> caused by conflicting CUDA toolkit versions on the host. `torch+cu128` has
> been tested successfully on HyperPod Slurm (torch 2.9.1, bitsandbytes
> 0.49.2, 4x A10G) — use it if your environment has a single, consistent
> CUDA installation. See [docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md)
> for diagnosis steps if you encounter CUBLAS errors.

### Option B: Enroot/Pyxis Container

If your cluster has [Enroot](https://github.com/NVIDIA/enroot) and [Pyxis](https://github.com/NVIDIA/pyxis) installed:

```bash
# Build the Docker image
docker build -t qwen3-qlora:latest .

# Import into Enroot squashfs format
enroot import dockerd://qwen3-qlora:latest
mv qwen3-qlora+latest.sqsh /fsx/containers/qwen3-qlora.sqsh

# Then edit slurm/qwen3_8b-qlora-zero2.sbatch and uncomment:
#   CONTAINER_IMAGE=/fsx/containers/qwen3-qlora.sqsh
```

## Submit Training

### ZeRO-2 (Default)

```bash
cd /fsx/awsome-distributed-ai/examples/training/deepspeed/qlora/slurm

# Activate venv (skip if using container mode)
source /fsx/venvs/qwen3-qlora/bin/activate

# Submit the job
sbatch qwen3_8b-qlora-zero2.sbatch
```

### ZeRO-3

ZeRO-3 shards optimizer states, gradients, **and model parameters** across GPUs for lower per-GPU memory (~12-14 GB vs ~15-17 GB with ZeRO-2).

```bash
sbatch qwen3_8b-qlora-zero3.sbatch
```

Output is written to `/fsx/qwen3-qlora/outputs-zero3`. Both jobs can run side-by-side on separate nodes.

### Customization Examples

**Override GPU count** (e.g., 8 GPUs on p4d):

```bash
# Edit GPUS_PER_NODE in the sbatch script, or override at submit time:
GPUS_PER_NODE=8 sbatch qwen3_8b-qlora-zero2.sbatch
```

**Multi-node training** (2 nodes):

```bash
sbatch --nodes=2 qwen3_8b-qlora-zero2.sbatch
```

**Container mode**:

```bash
# Uncomment CONTAINER_IMAGE in the sbatch script, then:
sbatch qwen3_8b-qlora-zero2.sbatch
```

## MIG Configuration

### Overview

[NVIDIA Multi-Instance GPU (MIG)](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) partitions a single physical GPU into multiple isolated GPU instances, each with its own compute, memory, and memory bandwidth. This is useful for QLoRA because:

- **Parallel experiments**: Run multiple QLoRA fine-tuning jobs with different hyperparameters simultaneously on a single GPU.
- **Resource efficiency**: QLoRA with a 4-bit quantized 8B model uses ~15 GB VRAM — a single A100-80GB can run multiple experiments in parallel via MIG.
- **Isolation**: Each MIG instance has hard memory isolation — one experiment cannot OOM another.

MIG instances appear as regular CUDA devices to PyTorch. The training code (`src/train.py`) requires **no changes** — it uses `LOCAL_RANK` and `WORLD_SIZE` which work identically with MIG.

### Supported MIG Profiles

| GPU | Profile | Compute | Memory | Recommended for QLoRA |
|-----|---------|---------|--------|-----------------------|
| A100 40GB | `1g.5gb` | 1/7 SMs | 5 GB | Too small |
| A100 40GB | `2g.10gb` | 2/7 SMs | 10 GB | Marginal (reduce seq_len) |
| A100 40GB | **`3g.20gb`** | 3/7 SMs | 20 GB | **Recommended** (2 instances per GPU) |
| A100 80GB | `3g.40gb` | 3/7 SMs | 40 GB | Comfortable |
| A100 80GB | **`4g.40gb`** | 4/7 SMs | 40 GB | **Recommended** (2 instances per GPU) |
| H100 80GB | `3g.40gb` | 3/7 SMs | 40 GB | Comfortable |
| H100 80GB | **`4g.40gb`** | 4/7 SMs | 40 GB | **Recommended** |

### Step-by-Step MIG Setup

MIG configuration must be done by a cluster administrator (requires root access). On HyperPod, this is typically done on the worker nodes.

```bash
# 1. Drain the Slurm node to prevent new jobs from scheduling
sudo scontrol update NodeName=<node-name> State=DRAIN Reason="MIG setup"

# 2. Stop all GPU processes on the node
sudo nvidia-smi -i 0 --gpu-reset    # repeat for each GPU index

# 3. Enable MIG mode on all GPUs
sudo nvidia-smi -i 0 -mig 1         # repeat for each GPU index
# Note: some systems require a reboot after enabling MIG mode

# 4. Create MIG instances — example: 2x 3g.20gb on A100-40GB
#    List available profiles first:
nvidia-smi mig -lgip

#    Create GPU instances (GI) then compute instances (CI):
sudo nvidia-smi mig -i 0 -cgi 9,9 -C    # profile ID 9 = 3g.20gb on A100-40GB
# Repeat for each physical GPU (-i 1, -i 2, etc.)
# Profile IDs vary by GPU model — always check with `nvidia-smi mig -lgip`

# 5. Verify MIG instances are visible
nvidia-smi -L
# Should show entries like:
#   GPU 0: NVIDIA A100 (UUID: GPU-xxxx)
#     MIG 3g.20gb Device 0: (UUID: MIG-xxxx)
#     MIG 3g.20gb Device 1: (UUID: MIG-xxxx)

# 6. Resume the Slurm node
sudo scontrol update NodeName=<node-name> State=RESUME
```

### Slurm MIG Configuration (Administrators)

For Slurm to correctly schedule MIG instances, the cluster `gres.conf` must list MIG devices. Consult the [Slurm MIG documentation](https://slurm.schedmd.com/gres.html#MIG_Management) and your HyperPod lifecycle scripts. With `--exclusive` mode (used by our sbatch script), Slurm allocates the full node and all MIG instances are visible without extra gres configuration.

### Running QLoRA on MIG

The sbatch script auto-detects MIG instances. When MIG is enabled, it counts the visible `MIG` devices from `nvidia-smi -L` and overrides `GPUS_PER_NODE` accordingly.

```bash
# Simply submit — MIG is detected automatically
sbatch qwen3_8b-qlora-zero2.sbatch

# The script logs: "MIG detected: N MIG instance(s) found, overriding GPUS_PER_NODE"
```

If the auto-detection is incorrect for your setup, override manually:

```bash
GPUS_PER_NODE=4 sbatch qwen3_8b-qlora-zero2.sbatch
```

### Parallel Experiments on MIG

To run multiple independent QLoRA experiments on separate MIG instances (e.g., hyperparameter search), assign specific MIG UUIDs to each job:

```bash
# List MIG UUIDs
nvidia-smi -L | grep MIG

# Run experiment 1 on MIG instance 0
CUDA_VISIBLE_DEVICES=MIG-<uuid-0> python src/train.py \
    --config_path configs/training_config_zero2.yaml \
    --output_dir /fsx/experiments/lr_1e-4

# Run experiment 2 on MIG instance 1 (in a separate terminal/job)
CUDA_VISIBLE_DEVICES=MIG-<uuid-1> python src/train.py \
    --config_path configs/training_config_zero2.yaml \
    --output_dir /fsx/experiments/lr_3e-4
```

## HyperPod Features

### Auto-Resume

When the sbatch script detects `/opt/sagemaker_cluster` (present on all HyperPod nodes), it passes `--auto-resume=1` to `srun`. This enables:

- **Automatic job restart** if a node is replaced due to hardware failure.
- **Checkpoint-based recovery** — the training script already resumes from the latest checkpoint in `OUTPUT_DIR` via `--resume_from_checkpoint auto`.

These two features work together: HyperPod replaces the failed node and restarts the Slurm job, and the training script picks up from the last saved checkpoint.

### Shared Filesystem

HyperPod mounts FSx for Lustre at `/fsx`, which is shared across all nodes. The sbatch script sets:

- `HF_HOME=/fsx/hf_cache` — model downloads are cached once and shared across nodes.
- `OUTPUT_DIR=/fsx/qwen3-qlora/outputs` — checkpoints are on shared storage, accessible from any node after failover.

## Monitoring

```bash
# Check job status
squeue -u $USER

# Watch training logs in real time
tail -f logs/qwen3_8b-qlora-zero2_<job-id>.out

# Monitor GPU utilization
watch -n 2 nvidia-smi

# Monitor MIG instance utilization
nvidia-smi mig -lgi

# TensorBoard (from head node)
tensorboard --logdir /fsx/qwen3-qlora/outputs/logs --bind_all
```

## Troubleshooting

### NCCL errors in containers

If you see `NCCL WARN` errors when using Enroot containers, ensure `/fsx` is mounted correctly and the NCCL socket interface is configured:

```bash
# In the sbatch script, add to CONTAINER_ARGS:
--container-mounts "/fsx:/fsx,/var/run:/var/run"
```

For EFA-enabled instances (p4d, p5), uncomment the EFA environment variables in the sbatch script.

### MIG not detected

If the sbatch script reports `GPUS_PER_NODE=4` (or whatever default) when MIG is enabled:

```bash
# Verify MIG is active
nvidia-smi --query-gpu=mig.mode.current --format=csv
# Should show "Enabled" for each GPU

# Verify MIG instances exist
nvidia-smi -L | grep MIG
# Should list MIG devices

# If empty, MIG mode is enabled but no instances were created.
# Create instances (see Step-by-Step MIG Setup above).
```

### OOM on MIG instances

MIG instances have less memory than the full GPU. If training crashes with OOM:

1. Reduce `max_seq_length` in `configs/training_config_zero2.yaml` (try 1024 or 768).
2. Ensure `per_device_train_batch_size` is 1.
3. Use a larger MIG profile (e.g., `4g.40gb` instead of `3g.20gb`).
4. Switch to DeepSpeed ZeRO-3 (`configs/deepspeed_zero3.json`) for additional memory savings.

### Job stuck in PENDING

```bash
# Check the reason
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

# Common reasons:
#   Resources — not enough free GPUs; wait or reduce --nodes
#   ReqNodeNotAvail — node is in DRAIN state (check MIG setup)
```

## Cleanup

```bash
# Cancel a running job
scancel <job-id>

# Cancel all your jobs
scancel -u $USER

# Clean up outputs (careful!)
# rm -rf /fsx/qwen3-qlora/outputs
```
