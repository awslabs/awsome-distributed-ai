## PEFT Fine Tuning of Llama 3 on Slurm Cluster (trn1/trn2)

This example showcases how to fine tune Llama 3 models using AWS Trainium instances and [Hugging Face Optimum Neuron](https://huggingface.co/docs/optimum-neuron). Optimum Neuron is the interface between the Transformers library and AWS Accelerators including AWS Trainium and AWS Inferentia. It provides tools for model loading, training, and inference on single- and multi-accelerator settings.

**Supported instances:** trn1.32xlarge, trn1n.32xlarge, trn2.48xlarge, trn2.3xlarge. The training script auto-detects the instance type and sets tensor parallelism accordingly.

This training script is adapted from the [official upstream example](https://github.com/huggingface/optimum-neuron/blob/main/examples/training/llama/finetune_llama.py) and uses:

- **`NeuronModelForCausalLM`** for tensor-parallel model loading
- **`NeuronSFTTrainer`** with **LoRA** (PEFT) for parameter-efficient fine tuning
- **Flash Attention 2** for memory-efficient attention
- **Chat template formatting** with sequence packing
- The [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset

### Software Versions

| Package | Version |
|---------|---------|
| optimum-neuron | 0.4.5 |
| trl | 0.24.0 |
| peft | 0.17.0 |
| transformers | ~4.57 |
| torch | 2.8.0 |
| neuronx-distributed | 0.17.x |
| neuronx-cc | 2.23.x |
| Python | 3.10 |

## Prerequisites

Before running this training, you'll need a SageMaker HyperPod cluster with at least 1 Trainium instance. Instructions can be found in the [Cluster Setup](https://catalog.workshops.aws/sagemaker-hyperpod/en-US/01-cluster) section.

You will also need to complete the following prerequisites:

* Submit a service quota increase request for Trainium instances (ml.trn1.32xlarge, ml.trn1n.32xlarge, or ml.trn2.48xlarge) in your AWS Region.
* Locally, install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) (minimum version 2.14.3).
* Locally, install the [AWS Systems Manager Session Manager Plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) to SSH into your cluster.
* Since Llama 3 is a gated model, register on [Hugging Face](https://huggingface.co) and obtain an [access token](https://huggingface.co/docs/hub/en/security-tokens).

### Training

## Step 1: Download Training Scripts

Begin by downloading the training scripts from the awsome-distributed-ai repo:

```bash
cd ~/
git clone https://github.com/awslabs/awsome-distributed-ai

cd ~/awsome-distributed-ai/examples/training/optimum-neuron/llama3/slurm/fine-tuning
```

## Step 2: Setup Python Environment

Setup a virtual Python environment and install training dependencies. Make sure this repo is stored on the shared FSx volume of your cluster so all nodes have access to it.

> **Note:** Optimum Neuron 0.4.5 requires Python 3.10. The environment setup script installs Python 3.10 via the deadsnakes PPA and creates a clean virtual environment with the Neuron SDK and optimum-neuron.

```bash
sbatch submit_jobs/0.create_env.sh
```

View the logs:

```bash
tail -f /fsx/ubuntu/peft_ft/logs/0_create_env.out
```

Before proceeding to the next step, check if the current job has finished:

```bash
squeue
```

## Step 3: Download the Model

Download the model to your FSx file volume. First modify the `submit_jobs/1.download_model.sh` script to include your Hugging Face access token:

```bash
export HF_TOKEN="<Your Hugging Face Token>"
```

Then run:

```bash
sbatch submit_jobs/1.download_model.sh
```

## Step 4: Compile the Model

Before training on Trainium with Neuron, pre-compile your model with the [neuron_parallel_compile CLI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/training/pytorch-neuron-parallel-compile.html). This traces through the training code and compiles the computation graphs ahead of time, reducing compilation time during actual training.

```bash
sbatch submit_jobs/2.compile_model.sh
```

The compilation process generates NEFF (Neuron Executable File Format) files that are cached and reused during fine tuning.

## Step 5: Fine Tuning

With the model compiled, begin fine tuning. The training script auto-detects the instance type and configures parallelism:

| Instance | NeuronCores | TP Degree | DP Workers |
|----------|-------------|-----------|------------|
| trn1.32xlarge / trn1n.32xlarge | 32 | 8 | 4 |
| trn2.48xlarge | 64 | 4 | 16 |
| trn2.3xlarge | 4 | 4 | 1 |

Common training settings:
- **BFloat16** precision
- **LoRA** targeting all linear projections (q, k, v, o, gate, up, down)
- **Sequence packing** for efficient batching
- **Max sequence length 2048** (required minimum for flash attention)

```bash
sbatch submit_jobs/3.finetune.sh
```

The training configuration can be modified in `finetune-llama3-8B.sh`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_id` | `/fsx/ubuntu/peft_ft/model_artifacts/llama3-8B` | Path to model |
| `--dataset` | `databricks/databricks-dolly-15k` | Training dataset |
| `--max_seq_length` | 2048 | Max sequence length (multiple of 2048 for flash attention) |
| `--tensor_parallel_size` | auto-detected | Tensor parallelism degree (8 for trn1, 4 for trn2) |
| `--per_device_train_batch_size` | 1 | Batch size per DP worker |
| `--gradient_accumulation_steps` | 3 | Gradient accumulation steps |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha |
| `--learning_rate` | 2e-05 | Learning rate |

## Step 6: Model Weight Consolidation

After training, the checkpoint directory contains model parallel shards from each neuron device. Use the consolidation script to combine the shards into a single `model.safetensors` file:

```bash
sbatch submit_jobs/4.model_consolidation.sh
```

> **Note:** Update the `--input_dir` in `submit_jobs/4.model_consolidation.sh` to match your actual checkpoint name (e.g., `checkpoint-1251`).

## Step 7: Merge LoRA Weights

After consolidating the model shards, merge the LoRA adapter weights back into the base Llama 3 model:

```bash
sbatch submit_jobs/5.merge_lora_weights.sh
```

Your final fine tuned model weights will be saved to the `--final_model_path` directory specified in the script.

## Step 8: Validate Your Trained Model

See how the fine tuned model's generations differ from the base model:

```bash
sbatch submit_jobs/6.inference.sh
```

This generates a prediction for the question "Who are you?", comparing the base model response to the fine tuned model response with a system prompt to respond like a pirate.

And that's it! You've successfully fine tuned a Llama 3 model on Amazon SageMaker HyperPod using PEFT with Neuron.
