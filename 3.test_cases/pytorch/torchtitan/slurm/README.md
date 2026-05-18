## Setup Instructions

### 0. Prerequisites

Before running this training, you'll need to create a Slurm cluster with an FSx for Lustre file system. Instructions can be found in [1.architectures](../../../1.architectures). FP8 data types are natively supported on NVIDIA H100 and subsequent generations, so it is recommended to run this on at least 1 x p5/p5e/p5en.48xlarge (H100) or p6-b200/p6-b300 (Blackwell) instance. The [Performance Numbers](#performance-numbers) section was originally captured on 4 x p5.48xlarge.

The setup script targets CUDA 13 (`cu130`) wheels so that `torch.compile`-ed FP8 kernels run with native `sm_103` binaries on P6-B300; older drivers/CUDA toolkits will fall back to PTX-JIT for B300.

### 1. Create torchtitan venv

On your cluster head node, run the `0.create_venv.sh` script:

```bash
bash 0.create_venv.sh
```

This script:
- Creates a Python 3.11 stdlib virtual environment named `pt_torchtitan`
- Installs **pinned** versions of `torch` (2.9.1+cu130) and `torchao` (0.17.0+cu130) from `https://download.pytorch.org/whl/cu130`
- Clones torchtitan at the **`v0.2.2` release tag** (not `main`) and installs it editable

Override the defaults by exporting `PYTHON_BIN`, `TORCH_VERSION`, `TORCHAO_VERSION`, `TORCHTITAN_REF`, or `PYTORCH_INDEX_URL` before invoking the script.


### 2. Download the Tokenizer

First, create a Hugging Face account to retrieve a [token](https://huggingface.co/settings/tokens.). Log in to your account and create an access token from Hugging Face Tokens. Then apply for Llama3.1 weight access from [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) page.

Use the following command to download the Meta Llama 3 tokenizer:

```bash
cd torchtitan
python scripts/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original" --hf_token=YOUR_HF_TOKEN_HERE
```

The tokenizer will be downloaded to `torchtitan/assets/tokenizer/original`. Ensure that you update the tokenizer path in the training config TOML file, for example:

```
tokenizer_path = "./torchtitan/assets/tokenizer/original/tokenizer.model"
```

### 3. Launch Distributed Training

The provided SLURM batch script configures and launches distributed training:

```bash
sbatch 1.llama_3_8b_torchtitan.sh
```

This script:
- Sets the path to the torchtitan training script: `./torchtitan/torchtitan/train.py`
- Uses the **vendored** default Llama 3 8B configuration: `./configs/llama3_8b.toml` (a copy of torchtitan v0.2.2's preset, included so the test case behavior doesn't drift with upstream `main`)
- Launches distributed training on your cluster

To run with FP8 + `torch.compile` instead, point `CONFIG_FILE` at the optimized variant before sbatch:

```bash
CONFIG_FILE="$(pwd)/configs/llama3_8b_fp8_compile.toml" sbatch 1.llama_3_8b_torchtitan.sh
```

The training will log metrics including loss, throughput, memory utilization, and MFU (Model FLOPS Utilization) to help monitor training efficiency.

## Performance Numbers

Running the llama3_8b.toml default configuration in torchtitan/models/llama/train_configs on 4 x p5.48xlarge instances (each instance contains 8 x H100 GPUs)

```bash
1: 2025-03-04 00:44:44,441 - root - INFO - [36mstep: 1990  [32mloss:  3.4370  [33mmemory: 68.57GiB(86.69%)  [34mtps: 6,785  [35mmfu: 39.73%[39m
0: 2025-03-04 00:44:44,441 - root - INFO - [36mstep: 1990  [32mloss:  3.4370  [33mmemory: 68.57GiB(86.69%)  [34mtps: 6,785  [35mmfu: 39.73%[39m
0: 2025-03-04 00:44:44,441 - root - INFO - [36mstep: 1990  [32mloss:  3.4370  [33mmemory: 68.57GiB(86.69%)  [34mtps: 6,785  [35mmfu: 39.73%[39m
3: 2025-03-04 00:44:44,441 - root - INFO - [36mstep: 1990  [32mloss:  3.4370  [33mmemory: 68.57GiB(86.69%)  [34mtps: 6,785  [35mmfu: 39.73%[39m
2: 2025-03-04 00:44:44,441 - root - INFO - [36mstep: 1990  [32mloss:  3.4370  [33mmemory: 68.57GiB(86.69%)  [34mtps: 6,785  [35mmfu: 39.73%[39m
2: 2025-03-04 00:44:44,441 - root - INFO - [36mstep: 1990  [32mloss:  3.4370  [33mmemory: 68.57GiB(86.69%)  [34mtps: 6,785  [35mmfu: 39.73%[39m
2: 2025-03-04 00:44:44,441 - root - INFO - [36mstep: 1990  [32mloss:  3.4370  [33mmemory: 68.57GiB(86.69%)  [34mtps: 6,785  [35mmfu: 39.73%[39m
3: 2025-03-04 00:44:44,441 - root - INFO - [36mstep: 1990  [32mloss:  3.4370  [33mmemory: 68.57GiB(86.69%)  [34mtps: 6,785  [35mmfu: 39.73%[39m
2: 2025-03-04 00:44:44,441 - root - INFO - [36mstep: 1990  [32mloss:  3.4370  [33mmemory: 68.57GiB(86.69%)  [34mtps: 6,785  [35mmfu: 39.73%[39m
```


## Performance Optimizations

`configs/llama3_8b_fp8_compile.toml` is a copy of `configs/llama3_8b.toml` with `torch.compile` and FP8 (rowwise dynamic) enabled. The toggles in torchtitan v0.2.2 are:

```toml
[model]
converters = ["float8"]

[compile]
enable = true
components = ["model", "loss"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
precompute_float8_dynamic_scale_for_fsdp = true
filter_fqns = ["output"]
```

Running with this optimized config on 4 x p5.48xlarge (32 H100s) improved throughput by **15.92%** and MFU from **39.73% → 46.06%** compared to the default configuration:

```bash
2: 2025-03-04 00:31:19,918 - root - INFO - [36mstep: 1990  [32mloss:  3.4255  [33mmemory: 63.48GiB(80.25%)  [34mtps: 7,865  [35mmfu: 46.06%[39m
0: 2025-03-04 00:31:19,918 - root - INFO - [36mstep: 1990  [32mloss:  3.4255  [33mmemory: 63.48GiB(80.25%)  [34mtps: 7,865  [35mmfu: 46.06%[39m
1: 2025-03-04 00:31:19,918 - root - INFO - [36mstep: 1990  [32mloss:  3.4255  [33mmemory: 63.48GiB(80.25%)  [34mtps: 7,865  [35mmfu: 46.06%[39m
1: 2025-03-04 00:31:19,918 - root - INFO - [36mstep: 1990  [32mloss:  3.4255  [33mmemory: 63.48GiB(80.25%)  [34mtps: 7,865  [35mmfu: 46.06%[39m
3: 2025-03-04 00:31:19,918 - root - INFO - [36mstep: 1990  [32mloss:  3.4255  [33mmemory: 63.48GiB(80.25%)  [34mtps: 7,865  [35mmfu: 46.06%[39m
2: 2025-03-04 00:31:19,918 - root - INFO - [36mstep: 1990  [32mloss:  3.4255  [33mmemory: 63.48GiB(80.25%)  [34mtps: 7,865  [35mmfu: 46.06%[39m
2: 2025-03-04 00:31:19,918 - root - INFO - [36mstep: 1990  [32mloss:  3.4255  [33mmemory: 63.48GiB(80.25%)  [34mtps: 7,865  [35mmfu: 46.06%[39m
0: 2025-03-04 00:31:19,918 - root - INFO - [36mstep: 1990  [32mloss:  3.4255  [33mmemory: 63.48GiB(80.25%)  [34mtps: 7,865  [35mmfu: 46.06%[39m
0: 2025-03-04 00:31:19,918 - root - INFO - [36mstep: 1990  [32mloss:  3.4255  [33mmemory: 63.48GiB(80.25%)  [34mtps: 7,865  [35mmfu: 46.06%[39m
```

