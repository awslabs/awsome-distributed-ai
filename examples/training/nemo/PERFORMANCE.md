# Performance

This document describes the process of performance measurements of NeMo 2.x framework on AWS infrastructure.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Expected Outputs](#expected-outputs)
3. [Multi-Node Distributed Training](#multi-node-distributed-training)
4. [Environment Setup](#environment-setup)
5. [Pre-training Performance](#pre-training-performance)
6. [Fine-Tuning Performance](#fine-tuning-performance)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Useful Links and Documentation
* [NVIDIA NeMo Performance Summary](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-summary.html)
* [NVIDIA NeMo Performance Scripts](https://github.com/NVIDIA/NeMo/tree/main/scripts/performance/llm)
* [NVIDIA NeMo Compatibility Matrix](https://docs.nvidia.com/nemo-framework/user-guide/latest/softwarecomponentversions.html)
* [AWS EFA Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html)
* [EFA Cheatsheet](../../../architectures/efa-cheatsheet.md)

### NeMo Version Compatibility

| NeMo Version | Docker Image Tag | Notes |
|--------------|------------------|-------|
| 2.5.0 | nvcr.io/nvidia/nemo:25.07 | Recommended |
| 2.4.0 | - | Avoid - see [issue #14392](https://github.com/NVIDIA/NeMo/issues/14392) |

Always verify compatibility with the [NVIDIA NeMo Compatibility Matrix](https://docs.nvidia.com/nemo-framework/user-guide/latest/softwarecomponentversions.html).

## Environment Setup

### Create Conda Environment

```bash
conda create -yn nemo python=3.12
conda activate nemo
```

### Install NeMo

Make sure that NeMo version is compatible with the docker image according to the [compatibility matrix](https://docs.nvidia.com/nemo-framework/user-guide/latest/softwarecomponentversions.html).

```bash
git clone git@github.com:NVIDIA/NeMo.git
cd NeMo
git checkout v2.5.0rc0  # 2.4.0 was broken https://github.com/NVIDIA/NeMo/issues/14392
pip install -e '.[all]' # -e makes recommended model configs loadable
```

### Optional: Configure Results Directory

```bash
export NEMORUN_HOME=/fsxl/.../nemo_run
```

Default location is `~/.nemo_run/experiments/`.

### Build Docker Image

The Dockerfile extends the NVIDIA NeMo container with AWS EFA (Elastic Fabric Adapter) support for high-performance networking. See the [Dockerfile](../Dockerfile) in this directory for the complete configuration.

Key components installed:
- **EFA installer (v1.47.0)** - provides libfabric and Open MPI
- **GDRCOPY v2.5.1** - for GPU Direct RDMA  
- **AWS-OFI-NCCL plugin** - bundled with EFA installer at `/opt/amazon/ofi-nccl`

For detailed EFA installation instructions, see the [AWS EFA documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html).

```bash
docker build --progress=plain -t aws-nemo:latest -f Dockerfile .
enroot import -o ~/aws-nemo.sqsh dockerd://aws-nemo:latest
```

### Environment Variables

Create an `env_vars.json` file with optimized settings for EFA:

```json
{
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
    "NVTE_ASYNC_AMAX_REDUCTION": "1",
    "NVTE_FUSED_ATTN": "0",
    "FI_EFA_USE_HUGE_PAGE": "0",
    "NCCL_DEBUG": "INFO",
    "FI_PROVIDER": "efa"
}
```

See the [EFA Cheatsheet](../../../architectures/efa-cheatsheet.md) for detailed explanations of each variable.

### Run Performance Test

To enable EFA, export these environment variables before running tests:

```bash
export FI_PROVIDER=efa
export NCCL_DEBUG=INFO
```

## Expected Outputs

Performance tests will generate the following metrics:
- **Throughput**: samples/sec or tokens/sec
- **FLOPs utilization**: Percentage of theoretical peak FLOPs achieved
- **Memory usage**: GPU memory consumption per device
- **Step time**: Average time per training step in seconds

Results are stored in the directory specified by `NEMORUN_HOME` (default: `~/.nemo_run/experiments/`).

Example output:
```
Training epoch 1, iteration 10/999 | lr: 4.498e-06 | global_batch_size: 512 | 
global_step: 29 | reduced_train_loss: 11.03 | train_step_timing in s: 46.84 | 
consumed_samples: 15360
```

## Multi-Node Distributed Training

The performance scripts support multi-node training via Slurm. The `--num_gpus` parameter specifies total GPUs across all nodes.

**Examples:**
| Total GPUs | GPUs per Node | Number of Nodes |
|------------|---------------|-----------------|
| 8 | 8 | 1 |
| 64 | 8 | 8 |
| 128 | 8 | 16 |
| 256 | 8 | 32 |

Ensure your Slurm partition has sufficient nodes available. See the [Slurm README](../slurm/README.md) for detailed setup instructions.

## Pre-training Performance

> **Note:** All configurations below use mock data for testing. No dataset preprocessing is required.

### NVIDIA H100(also applicable to H200)

`NeMo/scripts/performance/recommended_model_configs/model_configs_h100.csv`

| Model     | #-GPUs | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | GA |
|-----------|--------|-----|-----|-----------------|----|----|----|----|----|----|
| LLAMA3-8B | 8      | 128 | 1   | 8192            | 1  | 1  | 2  | 1  | 1  | 32 |

```bash
python -m scripts.performance.llm.pretrain_llama3_8b \
    --account $(whoami) --partition p5en -i ./aws-nemo.sqsh \
    --gpu h100 --num_gpus 8 -gb 128 -mb 1 -tp 1 -pp 1 -cp 2 -vp 1 -ep 1
```

| Model      | #-GPUs | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | GA |
|------------|--------|-----|-----|-----------------|----|----|----|----|----|----|
| LLAMA3-70B | 64     | 128 | 1   | 8192            | 4  | 8  | 1  | 5  | 1  | 64 |

```bash
python -m scripts.performance.llm.pretrain_llama3_70b \
    --account $(whoami) --partition p5en -i ./aws-nemo.sqsh \
    --gpu h100 --num_gpus 64 -gb 128 -mb 1 -tp 4 -pp 8 -cp 1 -vp 5 -ep 1
```

| Model         | #-GPUs | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | GA |
|---------------|--------|-----|-----|-----------------|----|----|----|----|----|----|
| LLAMA3.1-405B | 128    | 64  | 1   | 8192            | 8  | 8  | 2  | 8  | 1  | 64 |

```bash
python -m scripts.performance.llm.pretrain_llama31_405b \
    --account $(whoami) --partition p5en -i ./aws-nemo.sqsh \
    --gpu h100 --num_gpus 128 -gb 64 -mb 1 -tp 8 -pp 8 -cp 2 -vp 8 -ep 1
```

### NVIDIA B200

`NeMo/scripts/performance/recommended_model_configs/model_configs_b200.csv`

| Model     | #-GPUs | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | GA |
|-----------|--------|-----|-----|-----------------|----|----|----|----|----|----|
| LLAMA3-8B | 8      | 128 | 2   | 8192            | 1  | 1  | 1  | 1  | 1  | 8  |

```bash
python -m scripts.performance.llm.pretrain_llama3_8b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    --gpu b200 -c fp8 --num_gpus 8 -gb 128 -mb 1 -tp 1 -pp 1 -cp 1 -vp 1 -ep 1
```

mxfp8:
```bash
python -m scripts.performance.llm.pretrain_llama3_8b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    --gpu b200 -c fp8 --num_gpus 8 -gb 128 -mb 1 -tp 1 -pp 1 -cp 1 -vp 1 -ep 1 -fr mxfp8
```

| Model      | #-GPUs | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | GA |
|------------|--------|-----|-----|-----------------|----|----|----|----|----|----|
| LLAMA3-70B | 64     | 128 | 1   | 8192            | 1  | 1  | 1  | 1  | 1  | 2  |

```bash
python -m scripts.performance.llm.pretrain_llama3_70b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    --gpu b200 -c fp8 --num_gpus 64 # -gb 128 -mb 1 -tp 1 -pp 1 -cp 1 -vp 1 -ep 1 -fsdp 1
```

| Model      | #-GPUs | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | GA |
|------------|--------|-----|-----|-----------------|----|----|----|----|----|----|
| LLAMA3-70B | 64     | 128 | 1   | 8192            | 2  | 4  | 2  | 5  | 1  | 32 |

```bash
python -m scripts.performance.llm.pretrain_llama3_70b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    --gpu b200 -c fp8 --num_gpus 64 -gb 128 -mb 1 -tp 2 -pp 4 -cp 2 -vp 5 -ep 1
```

> **Note:** The configuration above with `-tp 2 -pp 4 -cp 2 -vp 5` uses tensor parallelism (TP) across 2 GPUs, 
> pipeline parallelism (PP) across 4 stages, context parallelism (CP) across 2 GPUs, and 5 virtual pipeline stages.
> This configuration is optimized for B200 GPUs with FP8 precision.

| Model         | #-GPUs | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | GA |
|---------------|--------|-----|-----|-----------------|----|----|----|----|----|----|
| LLAMA3.1-405B | 128    | 64  | 1   | 8192            | 4  | 8  | 2  | 8  | 1  | 32 |

```bash
python -m scripts.performance.llm.pretrain_llama31_405b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    --gpu b200 -c fp8 --num_gpus 128 -gb 64 -mb 1 -tp 4 -pp 8 -cp 2 -vp 8 -ep 1
```

| Model            | #-GPUs | GBS  | MBS | Sequence Length | TP | PP | CP | VP | EP | GA |
|------------------|--------|------|-----|-----------------|----|----|----|----|----|----|
| LLAMA4-Scout-LLM | 64     | 1024 | 1   | 8192            | 1  | 2  | 1  | 24 | 8  | 32 |

```bash
python -m scripts.performance.llm.pretrain_llama4_e16 \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    --gpu b200 -c fp8 --num_gpus 64 -gb 1024 -mb 1 -tp 1 -pp 2 -cp 1 -vp 24 -ep 8
```

| Model               | #-GPUs | GBS  | MBS | Sequence Length | TP | PP | CP | VP | EP | GA |
|---------------------|--------|------|-----|-----------------|----|----|----|----|----|----|
| LLAMA4-Maverick-LLM | 128    | 1024 | 1   | 8192            | 1  | 2  | 1  | 12 | 64 | 16 |

```bash
python -m scripts.performance.llm.pretrain_llama4_e128 \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    --gpu b200 -c fp8 --num_gpus 128 -gb 1024 -mb 1 -tp 1 -pp 2 -cp 1 -vp 12 -ep 64
```


### Mixtral Models

| Model        | #-GPUs | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | GA |
|--------------|--------|-----|-----|-----------------|----|----|----|----|----|----|
| Mixtral-8x7B | 64     | 256 | 2   | 4096            | 1  | 1  | 1  | 1  | 8  | 2  |

```bash
python -m scripts.performance.llm.pretrain_mixtral_8x7b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    --gpu b200 -c fp8 --num_gpus 64 -gb 256 -mb 2 -tp 1 -pp 1 -cp 1 -vp 1 -ep 8
```

| Model         | #-GPUs | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | GA |
|---------------|--------|-----|-----|-----------------|----|----|----|----|----|----|
| Mixtral-8x22B | 256    | 64  | 1   | 65536           | 2  | 4  | 8  | 14 | 8  | 16 |

```bash
python -m scripts.performance.llm.pretrain_mixtral_8x22b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    --gpu b200 -c fp8 --num_gpus 256 -gb 64 -mb 1 -tp 2 -pp 4 -cp 8 -vp 14 -ep 8
```

### Nemotron Models

| Model           | #-GPUs | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | GA |
|-----------------|--------|-----|-----|-----------------|----|----|----|----|----|----|
| Nemotron5-H-56B | 64     | 192 | 2   | 8192            | 4  | 1  | 1  | 1  | 1  | 6  |

```bash
python -m scripts.performance.llm.pretrain_nemotronh_56b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    --gpu b200 -c fp8 --num_gpus 64 -gb 192 -mb 2 -tp 4 -pp 1 -cp 1 -vp 1 -ep 1
```

### DeepSeek Models

| Model      | #-GPUs | GBS  | MBS | Sequence Length | TP | PP | CP | VP | EP | GA  |
|------------|--------|------|-----|-----------------|----|----|----|----|----|-----|
| DeepSeekV3 | 256    | 2048 | 1   | 4096            | 2  | 16 | 1  | 1  | 8  | 256 |

```bash
python -m scripts.performance.llm.pretrain_deepseek_v3 \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    --gpu b200 -c fp8 --num_gpus 256 -gb 2048 -mb 1 -tp 2 -pp 16 -cp 1 -vp 1 -ep 8
```

## Fine-Tuning Performance

### Prerequisites

1. **Set NEMO_HOME**: Specify `NEMO_HOME` explicitly to prevent `fiddle._src.experimental.serialization.UnserializableValueError`.
   See [NeMo-Run FAQ](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/faqs.html#q-unserializablevalueerror-when-using-run-partial-or-run-config).

```bash
export NEMO_HOME=...
```

~~2. Append `FLOPsMeasurementCallback` for task == "none" in `set_exp_logging_configs` in `helpers.py` to get FLOPs measurements.~~

3. To enable `mxfp8` recipe, add `recipe.trainer.strategy.ddp.fp8_param_gather = True` in `finetune_llama3_8b.py`/`finetune_llama3_70b.py`:

```python
if args.fp8_recipe == "mxfp8":
    recipe.trainer.strategy.ddp.fp8_param_gather = True
```

### NVIDIA B200

#### LLAMA3-8B SFT

**Configuration:**

| Model     | Task | #-GPUs | GBS | MBS | Packed Sequence Length | TP | PP | VP | GA |
|-----------|------|--------|-----|-----|------------------------|----|----|----|----|
| LLAMA3-8B | SFT  | 8      | 8   | 1   | 16384                  | 1  | 1  | 1  | 1  |

**fp8:**
```bash
python -m scripts.performance.llm.finetune_llama3_8b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    -hf $HF_TOKEN \
    --gpu b200 -c fp8 -f sft --num_gpus 8 -gb 8 -mb 1 -tp 1 -pp 1 -vp 1
```

**mxfp8:**
```bash
python -m scripts.performance.llm.finetune_llama3_8b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    -hf $HF_TOKEN \
    --gpu b200 -c fp8 -f sft --num_gpus 8 -gb 8 -mb 1 -tp 1 -pp 1 -vp 1 -fr mxfp8
```

> **Note:** Replace `$HF_TOKEN` with your Hugging Face access token for gated models.

#### LLAMA3-70B SFT

| Model      | Task | #-GPUs | GBS | MBS | Packed Sequence Length | TP | PP | VP | GA |
|------------|------|--------|-----|-----|------------------------|----|----|----|----|
| LLAMA3-70B | SFT  | 32     | 32  | 1   | 4096                   | 2  | 4  | 5  | 8  |

fp8:
```bash
python -m scripts.performance.llm.finetune_llama3_70b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    -hf $HF_TOKEN \
    --gpu b200 -c fp8 -f sft --num_gpus 32 -gb 32 -mb 1 -tp 2 -pp 4 -vp 5
```
mxfp8:
```bash
python -m scripts.performance.llm.finetune_llama3_70b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    -hf $HF_TOKEN \
    --gpu b200 -c fp8 -f sft --num_gpus 32 -gb 32 -mb 1 -tp 2 -pp 4 -vp 5 -fr mxfp8
```

#### LLAMA3-70B LoRA

| Model      | Task  | #-GPUs | GBS | MBS | Packed Sequence Length | TP | PP | VP | GA |
|------------|-------|--------|-----|-----|------------------------|----|----|----|----|
| LLAMA3-70B | LoRA  | 8      | 32  | 1   | 4096                   | 1  | 4  | 20 | 16 |

fp8:
```bash
python -m scripts.performance.llm.finetune_llama3_70b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    -hf $HF_TOKEN \
    --gpu b200 -c fp8 -f lora --num_gpus 8 -gb 32 -mb 1 -tp 1 -pp 4 -vp 20
```
mxfp8:
```bash
python -m scripts.performance.llm.finetune_llama3_70b \
    --account $(whoami) --partition p6 -i ./aws-nemo.sqsh \
    -hf $HF_TOKEN \
    --gpu b200 -c fp8 -f lora --num_gpus 8 -gb 32 -mb 1 -tp 1 -pp 4 -vp 20 -fr mxfp8
```

## Troubleshooting

### EFA Not Detected
If NCCL falls back to TCP/IP instead of EFA:
1. Verify EFA is installed in the container: `fi_info -p efa`
2. Check environment variables are set correctly (especially `FI_PROVIDER=efa`)
3. Ensure security group allows all traffic from within the same security group
4. Verify EFA devices are available: `ls -la /dev/infiniband/`

### Out of Memory (OOM) Errors
- Reduce micro-batch size (`-mb`)
- Increase tensor parallelism (`-tp`)
- Enable activation checkpointing
- Use gradient accumulation (`-ga`)

### Slow Performance
- Verify EFA is active: `NCCL_DEBUG=INFO` should show `efa` provider in logs
- Check NVLink status: `nvidia-smi nvlink -s`
- Ensure GPUs are not throttling: `nvidia-smi dmon`

### Checkpoint Issues
- Ensure `NEMO_HOME` is set to a shared filesystem accessible from all nodes
- Verify sufficient disk space for checkpoints
