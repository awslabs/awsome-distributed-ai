# Running PyTorch FSDP with Slurm

The following content provides guidance on how to run PyTorch FSDP on a Slurm cluster using the common crawl dataset.

## Retrieve the guidance on your cluster

On your cluster head node,
1. Navigate to your shared FSx for Lustre file system.
* If you followed the tutorial linked above, it will be location at `/fsx`.
2. Clone this repo.

```bash
cd /fsx
git clone https://github.com/awslabs/awsome-distributed-training/
cd awsome-distributed-training/examples/pytorch/FSDP/slurm
```

3. You can launch the training through:
   - Option 1: Creating a Python Virtual Environment to install the necessary packages.
   - Option 2: Create a container image to install the packages and run.

### Option 1: Creating a Python Virtual Environment to install the necessary packages.
Run the `create_venv.sh` script:

```bash
. ./create_venv.sh
```
* By creating this environment on the shared FSx for Lustre volume, all compute nodes in our cluster will have access to it.

### Option 2: Create a container image to install the packages and run.

You will first build the container image with the command below:


```bash
docker build -f ../Dockerfile -t fsdp:pytorch2.7.1 ..
```

You will then convert the container image to a squash file via Enroot:

```bash
enroot import -o pytorch-fsdp.sqsh  dockerd://fsdp:pytorch2.7.1
```

## Data

For this example, we'll be using the [allenai/c4](https://huggingface.co/datasets/allenai/c4) dataset. Instead of downloading the whole thing, the `create_streaming_dataloaders` function will stream the dataset from [HuggingFace](https://huggingface.co/datasets), so there's no data prep required for running this training.

**For this dataset, we will need a Hugging Face access token**. First, create a [Hugging Face account](https://huggingface.co/welcome). Then [generate your access token with read permissions](https://huggingface.co/docs/hub/en/security-tokens). Set your HuggingFace Token as an environment variable in your Python Virtual Environment by running:

``` bash
export HF_TOKEN=<YOUR HF ACCESS TOKEN>
```

If you'd like to instead use your own dataset, you can do so by [formatting it as a HuggingFace dataset](https://huggingface.co/docs/datasets/create_dataset), and passing its location to the `--dataset` argument.

## Launch Training

In this solution, you will find FSDP training examples for Llama 2(7B, 13B, 70B), Llama 3.1(8B, 70B), Llama 3.2(1B, 3B),  Mistral 8x7b and Mistral Mathstral.
You can adjust the number of training nodes by modifying `#SBATCH --nodes=4` to match the size of your cluster.

If you are using a container image, you need to uncomment the line below in the sbatch script to use the squash file

```bash
#export CONTAINER_IMAGE=$(pwd)/pytorch-fsdp.sqsh
```

If you are using non-EFA enabled instances, such as G4dn, or single GPU g5 nodes, comment out all EFA environment variables on lines 24-25.

Also, under `User Variables` make sure to adjust `GPUS_PER_NODE` to match the number of GPUs on your instance type (8 for P4d(e)/P5/P6-B200), 4 for G5.12xlarge, 1 for G5.xlarge).

You can also adjust the training parameters in `TRAINING_ARGS` (for example, to increase batch size). Additional parameters can be found in `src/model_utils/arguments.py`. Note that we use the same directory for both `--checkpoint_dir` and `--resume_from_checkpoint`. If there are multiple checkpoints, `--resume_from_checkpoint` will automatically select the most recent one. This way if our training is interupted for any reason, it will automatically pick up the most recent checkpoint.

### Llama 3.1 8B training

To launch your training for Llama 3.1 8B, run

```bash
sbatch llama3_1_8b-training.sbatch
```

You'll find a new file in the FSDP directory of the form `llama3_1_8b-FSDP_[JOB ID].out`. This will be continuously updated with your training logs. Don't be worried if you see a long stream of NCCL logs (we prefer to use `NCCL_DEBUG=INFO` for verbose logging). After about a minute, you should see your model training, with an output similar to below for Llama3.1 8B:

```text
+ TORCHRUN_ARGS=('--nproc_per_node=8' '--nnodes=4' '--rdzv_id=288' '--rdzv_backend=c10d' '--rdzv_endpoint=p5-dy-gpu-1')
+ declare -a TORCHRUN_ARGS
+ export TORCHRUN=torchrun
+ TORCHRUN=torchrun
+ export TRAIN_SCRIPT=../src/train.py
+ TRAIN_SCRIPT=../src/train.py
+ TRAINING_ARGS=('--max_context_width=8192' '--num_key_value_heads=8' '--intermediate_size=14336' '--hidden_width=4096' '--num_layers=32' '--num_heads=32' '--model_type=llama_v3' '--tokenizer=hf-internal-testing/llama-tokenizer' '--checkpoint_freq=50' '--validation_freq=25' '--max_steps=100' '--checkpoint_dir=./checkpoints' '--dataset=allenai/c4' '--dataset_config_name=en' '--resume_from_checkpoint=./checkpoints' '--train_batch_size=1' '--val_batch_size=1' '--sharding_strategy=full' '--offload_activations=1')
...
0: 2025-06-17 16:23:09 I [train.py:156] Creating Model
0: 2025-06-17 16:24:21 I [train.py:172] Created model with total parameters: 7392727040 (7.39 B)
...
1: p5-dy-gpu-3:47930:47930 [0] NCCL INFO NCCL version 2.26.2+cuda12.2
1: p5-dy-gpu-3:47936:47936 [6] NCCL INFO cudaDriverVersion 12080
2: p5-dy-gpu-4:930643:930850 [7] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.13.2-aws
2: p5-dy-gpu-4:930643:930850 [7] NCCL INFO NET/OFI Using Libfabric version 1.22
...
0: 2025-06-17 16:24:52 I [train.py:103] Batch 0 Loss: 11.61653, Speed: 4.88 samples/sec, lr: 0.000100
0: 2025-06-17 16:24:55 I [train.py:103] Batch 1 Loss: 11.64398, Speed: 10.90 samples/sec, lr: 0.000100
0: 2025-06-17 16:24:58 I [train.py:103] Batch 2 Loss: 10.58705, Speed: 11.09 samples/sec, lr: 0.000100
0: 2025-06-17 16:25:01 I [train.py:103] Batch 3 Loss: 15.01381, Speed: 10.61 samples/sec, lr: 0.000100
0: 2025-06-17 16:25:04 I [train.py:103] Batch 4 Loss: 11.78982, Speed: 10.18 samples/sec, lr: 0.000099
0: 2025-06-17 16:25:08 I [train.py:103] Batch 5 Loss: 14.34635, Speed: 8.98 samples/sec, lr: 0.000099
```

###  Mistral 8x7B

To run Mistral 8x7B model, you will need first to review the terms of usage on [HuggingFace](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1).
Then you will need to create a [user access token](https://huggingface.co/docs/hub/en/security-tokens) to access the gated Mathstral 7B model.
Once created you will need to define it in your environment:

```bash
export HF_TOKEN=<YOUR TOKEN>
```

You are now ready to launch your training for Mistral 8x7B with the following command:

```bash
sbatch mistral_8x7b-training.sbatch
```

You'll find a new file in the FSDP directory of the form `mistral_8x7b-FSDP_[JOB ID].out`. This will be continuously updated with your training logs. Don't be worried if you see a long stream of NCCL logs (we prefer to use `NCCL_DEBUG=INFO` for verbose logging). After about a minute, you should see your model training, with an output similar to below for Mistral:

```text
...
+ export TORCHRUN=torchrun
+ TORCHRUN=torchrun
+ export TRAIN_SCRIPT=./train.py
+ TRAIN_SCRIPT=./train.py
+ TRAINING_ARGS=('--train_batch_size=4' '--val_batch_size=4' '--max_steps=5000' '--seed=42' '--bf16=1' '--grad_clip=1.0' '--weight_decay=0.2' '--beta1=0.9' '--beta2=0.95' '--activation_checkpointing=1' '--intermediate_size=14336' '--num_key_value_heads=8' '--logging_freq=1' '--max_context_width=32768' '--vocab_size=32000' '--hidden_width=4096' '--num_layers=32' '--num_heads=32' '--resid_pdrop=0.1' '--embd_pdrop=0.1' '--attn_pdrop=0.1' '--summary_first_pdrop=0.1' '--initializer_range=0.02' '--model_type=mixtral' '--rotary_pct=0.25' '--rotary_emb_base=10000' '--lr=0.0001' '--lr_decay_style=cosine' '--min_lr=1e-5' '--warmup=0.0032' '--plateau=0.0' '--dataset=allenai/c4' '--tokenizer=mistralai/Mixtral-8x7B-v0.1' '--epochs=3' '--dataset_config_name=en' '--limit_all_gathers=1' '--sharding_strategy=full' ' #' 'https://pytorch.org/docs/stable/fsdp.html' '--offload_activations=1')
+ declare -a TRAINING_ARGS
...
0: 2025-04-11 16:49:59 I [train.py:156] Creating Model
0: 2025-04-11 16:57:23 I [train.py:172] Created model with total parameters: 46702792704 (46.70 B)
0: 2025-04-11 16:57:56 I [train.py:216] Wrapped model with FSDP
0: 2025-04-11 16:57:56 I [train.py:233] Created optimizer
...
1: p5-dy-gpu-2:62571:62571 [1] NCCL INFO NCCL version 2.26.2+cuda12.2
1: p5-dy-gpu-2:62574:62574 [4] NCCL INFO cudaDriverVersion 12040
2: p5-dy-gpu-3:60823:61204 [2] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.14.0
2: p5-dy-gpu-3:60823:61204 [2] NCCL INFO NET/OFI Using Libfabric version 1.22
...
0: 2025-04-11 16:58:41 I [train.py:103] Batch 0 Loss: 11.21702, Speed: 6.19 samples/sec, lr: 0.000006
0: 2025-04-11 16:58:49 I [train.py:103] Batch 1 Loss: 11.20650, Speed: 14.51 samples/sec, lr: 0.000013
0: 2025-04-11 16:58:58 I [train.py:103] Batch 2 Loss: 11.12571, Speed: 15.06 samples/sec, lr: 0.000019
0: 2025-04-11 16:59:07 I [train.py:103] Batch 3 Loss: 10.97558, Speed: 14.70 samples/sec, lr: 0.000025
0: 2025-04-11 16:59:15 I [train.py:103] Batch 4 Loss: 10.82548, Speed: 14.48 samples/sec, lr: 0.000031
0: 2025-04-11 16:59:24 I [train.py:103] Batch 5 Loss: 10.31511, Speed: 14.50 samples/sec, lr: 0.000038
```

###  Mistral Mathstral 7B

To run Mistral Mathstral 7B model, you will need first to review the terms of usage on [HuggingFace](https://huggingface.co/mistralai/Mistral-7B-v0.1).
Then you will need to create a [user access token](https://huggingface.co/docs/hub/en/security-tokens) to access the gated Mathstral 7B model.
Once created you will need to define it in your environment:

```bash
export HF_TOKEN=<YOUR TOKEN>
```

You are now ready to launch your training for Mathstral 7B with the following command:

```bash
sbatch mathstral_7b-training.sbatch
```

For Mathstral, your output should look similar to the one below:

```text
...
+ TORCHRUN_ARGS=('--nproc_per_node=8' '--nnodes=4' '--rdzv_id=2515' '--rdzv_backend=c10d' '--rdzv_endpoint=p5-dy-gpu-1')
+ declare -a TORCHRUN_ARGS
+ export TORCHRUN=torchrun
+ TORCHRUN=torchrun
+ export TRAIN_SCRIPT=./train.py
+ TRAIN_SCRIPT=./train.py
+ TRAINING_ARGS=('--train_batch_size=1' '--val_batch_size=1' '--max_steps=5000' '--seed=42' '--grad_clip=1.0' '--weight_decay=0.2' '--beta1=0.9' '--beta2=0.95' '--activation_checkpointing=1' '--intermediate_size=14336' '--num_key_value_heads=8' '--logging_freq=1' '--max_context_width=32768' '--vocab_size=32768' '--hidden_width=4096' '--num_layers=32' '--num_heads=32' '--resid_pdrop=0.1' '--embd_pdrop=0.1' '--attn_pdrop=0.1' '--summary_first_pdrop=0.1' '--initializer_range=0.02' '--model_type=mistral' '--rotary_pct=0.25' '--rotary_emb_base=10000' '--lr=0.0001' '--lr_decay_style=cosine' '--min_lr=1e-5' '--warmup=0.0032' '--plateau=0.0' '--dataset=allenai/c4' '--tokenizer=mistralai/mathstral-7B-v0.1' '--epochs=3' '--checkpoint_dir=./checkpoints/mathstral-7B' '--resume_from_checkpoint=./checkpoints/mathstral-7B' '--checkpoint_freq=50' '--validation_freq=500' '--dataset_config_name=en' '--limit_all_gathers=1' '--sharding_strategy=full' ' #' 'https://pytorch.org/docs/stable/fsdp.html' '--offload_activations=1')
...
1: p5-dy-gpu-2:62571:62571 [1] NCCL INFO NCCL version 2.26.2+cuda12.2
1: p5-dy-gpu-2:62574:62574 [4] NCCL INFO cudaDriverVersion 12040
2: p5-dy-gpu-3:60823:61204 [2] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.14.0
2: p5-dy-gpu-3:60823:61204 [2] NCCL INFO NET/OFI Using Libfabric version 1.22
...
0: 2025-04-07 22:04:23 I [train.py:156] Creating Model
0: 2025-04-07 22:05:30 I [train.py:172] Created model with total parameters: 7248023552 (7.25 B)
0: 2025-04-07 22:05:40 I [train.py:216] Wrapped model with FSDP
0: 2025-04-07 22:05:40 I [train.py:233] Created optimizer
...
0: 2025-04-07 22:06:15 I [train.py:103] Batch 0 Loss: 11.21489, Speed: 2.61 samples/sec, lr: 0.000006
0: 2025-04-07 22:06:17 I [train.py:103] Batch 1 Loss: 11.20829, Speed: 15.37 samples/sec, lr: 0.000013
0: 2025-04-07 22:06:19 I [train.py:103] Batch 2 Loss: 11.15640, Speed: 14.87 samples/sec, lr: 0.000019
0: 2025-04-07 22:06:21 I [train.py:103] Batch 3 Loss: 10.90571, Speed: 15.45 samples/sec, lr: 0.000025
0: 2025-04-07 22:06:24 I [train.py:103] Batch 4 Loss: 10.60309, Speed: 15.35 samples/sec, lr: 0.000031
0: 2025-04-07 22:06:25 I [train.py:103] Batch 5 Loss: 10.02562, Speed: 16.59 samples/sec, lr: 0.000038
```

## References
Llama 2 and  Llama 3.x models parameters are based on the values in the [Llama 2 paper](https://arxiv.org/abs/2307.09288) and [Llama 3 paper](https://arxiv.org/abs/2407.21783) 


| Parameter            | Llama 2 7B | Llama 2 13B | Llama 2 70B | Llama 3.1 8B | Llama 3.1 70B | Llama 3.2 1B | Llama 3.2 3B |
|----------------------|------------|-------------|-------------|--------------|---------------|--------------|--------------|
| intermediate_size    | 11008      | 13824       | 28672       | 14336        | 28672         | 8192         | 11008        |
| num_key_value_heads  | 32         | 40          | 8           | 8            | 8             | 8            | 8            |
| hidden_width         | 4096       | 5120        | 8192        | 4096         | 8192          | 2048         | 3072         |
| num_layers           | 32         | 40          | 80          | 32           | 80            | 16           | 28           |
| num_heads            | 32         | 40          | 64          | 32           | 64            | 32           | 24           |
| max_context_length   | 4096       | 4096        | 4096        | 8192         | 8192          | 8192         | 8192         |


If you need to cancel or modify your job, see the Slurm commands available in the [Slurm documentation](https://slurm.schedmd.com/quickstart.html).
