# Running NVIDIA NeMo 2.0 with Nemo-Run on Amazon EKS and SageMaker HyperPod EKS

## Overview

[NVIDIA NeMo](https://developer.nvidia.com/nemo-framework) is a cloud-native framework for building, training, and fine-tuning generative AI models at scale, from billions to trillions of parameters. This repository provides a comprehensive guide for running NVIDIA NeMo 2.0 on Amazon EKS (Elastic Kubernetes Service) or SageMaker HyperPod EKS for large language model training.

## Table of Contents

1. [Overview](#overview)
2. [Key Features of NeMo 2.0](#key-features-of-nemo-20)
3. [Architecture](#architecture)
4. [Prerequisites](#prerequisites)
   - [Additional Setup for Standard EKS (Non-HyperPod)](#additional-setup-for-standard-eks-non-hyperpod)
   - [Required Tools](#required-tools)
   - [Storage](#storage)
5. [Testing Configuration and GPU Requirements](#testing-configuration-and-gpu-requirements)
6. [Building the AWS-Optimized NeMo Container for EFA Enabled Instances](#building-the-aws-optimized-nemo-container-for-efa-enabled-instances)
   - [Build the Docker Image](#build-the-docker-image)
   - [Push to Amazon ECR](#push-to-amazon-ecr)
7. [Setting up Development Environment](#setting-up-development-environment)
   - [System Prerequisites](#system-prerequisites)
   - [Install Software Dependencies](#install-software-dependencies)
8. [Getting the Kubernetes Cluster Ready](#getting-the-kubernetes-cluster-ready)
   - [Configure SkyPilot for Kubernetes](#configure-skypilot-for-kubernetes)
   - [Initialize Git Repository](#initialize-git-repository)
   - [Verifying PVC Setup](#verifying-pvc-setup)
9. [Data Preprocessing for Custom Datasets (Pretraining Only)](#data-preprocessing-for-custom-datasets-pretraining-only)
   - [Overview](#overview-1)
   - [Automated Data Processing Script](#automated-data-processing-script)
   - [Running the Data Processing](#running-the-data-processing)
   - [Processed Data Structure](#processed-data-structure)
10. [Launching NeMo Training Jobs](#launching-nemo-training-jobs)
   - [Overview](#overview-2)
   - [Training Parameters](#training-parameters)
   - [Pretraining Jobs](#pretraining-jobs)
   - [Finetuning Jobs](#finetuning-jobs)
   - [Customizing Datasets](#customizing-datasets)
   - [Important Notes](#important-notes)
11. [Monitoring and Debugging](#monitoring-and-debugging)
    - [NeMo-Run Monitoring](#nemo-run-monitoring)
    - [Check Experiment Status](#check-experiment-status)
    - [View Training Logs](#view-training-logs)
    - [View Job Queue](#view-job-queue)
    - [View Job Logs](#view-job-logs)
12. [References](#references)

### Key Features of NeMo 2.0

- **Python-based Configuration System**: Enhanced flexibility and streamlined customization
- **Comprehensive Development Toolkit**: End-to-end solutions for data preparation, training, and deployment
- **Advanced Infrastructure Optimization**: Multi-GPU and multi-node support with distributed training capabilities
- **Enterprise-grade Scalability**: Parallelism techniques, memory optimization, and deployment pipelines
- **Improved IDE Integration**: Better developer experience with Python-native configurations

## Architecture

This implementation leverages Kubernetes on AWS infrastructure to orchestrate distributed NeMo training workloads:

- **Amazon EKS/SageMaker HyperPod**: Container orchestration platform
- **FSx for Lustre**: High-performance file system for training data and checkpoints
- **AWS-optimized Container**: Custom Docker image with EFA support for EFA enabled instances
- **NeMo-Run**: Python-based workflow management for NeMo training jobs
- **SkyPilot**: Job orchestration backend for Kubernetes
- **Automated Data Processing**: Custom scripts for dataset preparation and preprocessing
- **Hugging Face Integration**: Direct dataset loading and preprocessing from Hugging Face Hub

## Prerequisites

Before you begin, ensure you have the following:

- **Kubernetes Cluster**: An EKS cluster or SageMaker HyperPod EKS cluster
  - For SageMaker HyperPod EKS: Follow [this workshop](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US/00-setup) or [this repository](https://github.com/awslabs/awsome-distributed-training/tree/main/1.architectures/7.sagemaker-hyperpod-eks)
  - For standard EKS: Follow [these instructions](https://github.com/awslabs/awsome-distributed-training/tree/main/1.architectures/4.amazon-eks)
    
    **Additional Setup for Standard EKS (Non-HyperPod):**
    
    If you're using a standard EKS cluster (not SageMaker HyperPod), you'll need to install the following device plugins:
    
    1. **The NVIDIA GPU Operator** (if not already installed):
    ```bash
    # Add NVIDIA Helm repo
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
    helm repo update
    # Install the GPU Operator in a namespace (e.g. gpu-operator)
    helm install gpu-operator nvidia/gpu-operator \
      --namespace gpu-operator --create-namespace \
      --set driver.enabled=false \
      --set toolkit.enabled=false \
      --wait
    ```
    
    2. **AWS EFA Kubernetes Device Plugin** (for instances with EFA support):
    ```bash
    helm repo add eks https://aws.github.io/eks-charts
    helm install efa eks/aws-efa-k8s-device-plugin -n kube-system
    ```
    
    After installation, verify the GPU Operator and EFA plugin are running:
    ```bash
    # Check GPU Operator pods
    kubectl get pods -n gpu-operator
    
    # Check EFA device plugin
    kubectl get pods -n kube-system | grep aws-efa-k8s-device-plugin
    ```
    
    For P4 instances, you should see `vpc.amazonaws.com/efa: 4` in the node's allocatable resources. For P5.48xlarge instances, you should see `vpc.amazonaws.com/efa: 32` if you check the node details `kubectl describe node <node name>`
    
    > **Note**: If EFA is enabled in the node group, ensure your security group allows all outgoing traffic originating from the same security group for EFA to work properly.

- **Required Tools**:
  - [Docker](https://docs.docker.com/engine/install/) for container management
  - [kubectl](https://kubernetes.io/docs/tasks/tools/) configured for your EKS cluster

- **Storage**:
  - An FSx for Lustre filesystem mounted in the EKS cluster ([setup instructions](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US/01-cluster/06-fsx-for-lustre))
  - Make note of the PVC (Persistent Volume Claim) name for your FSx volume and decide on a mount path to use in the container

## Testing Configuration and GPU Requirements

> **Important**: All examples in this repository were tested on a minimum node configuration of **1 g6e.24xlarge instance type (4 L40S GPUs)**. 
>
> **Memory Considerations**: Using lower memory GPU instances like A10G (g5 instance types) might result in **CUDA out of memory errors** for the finetuning examples.

## 1. Building the AWS-Optimized NeMo Container for EFA Enabled Instances

**If you're not using an EFA enabled instance type, you can skip this step**. Here the base NeMo image (`nvcr.io/nvidia/nemo:26.02`) is enhanced with AWS-specific optimizations for EFA support.

### Build the Docker Image

```bash
# Build the Docker image using the build script
./build.sh
```

### Push to Amazon ECR

```bash
# Push the image to ECR using the push script
./push.sh
```

The script will automatically:
- Create the ECR repository if it doesn't exist
- Login to ECR
- Tag the image with the ECR URI
- Push the image to ECR

You can customize the AWS region by setting the `AWS_REGION` environment variable:
```bash
AWS_REGION=us-west-2 ./push.sh
```

## 2. Setting up Development Environment

### System Prerequisites

**Note:** The following instructions were tested and developed on an Ubuntu-based workstation. If you're using a different operating system, please adapt the installation commands to your OS package manager and environment.

Before proceeding, ensure you have the following installed:

- **Python 3.10**
- [wget](https://www.gnu.org/software/wget/)
- [socat](https://linux.die.net/man/1/socat) and [netcat](https://netcat.sourceforge.net/) (required by SkyPilot)

**Installation example (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install wget socat netcat -y
```

### Install Software Dependencies

**Note:** The following is an example using a basic Python virtual environment. Activate your Python 3.10 environment however you have it configured (conda, pyenv, virtualenv, etc.) before executing `bash venv.sh`

```bash
# Verify Python 3.10 installation
python --version

# Example: Create and activate virtual environment
python -m venv nemo-env
source nemo-env/bin/activate

# Install dependencies (including NeMo-Run and SkyPilot)
bash venv.sh
```

## 3. Getting the Kubernetes Cluster Ready 

### Configure SkyPilot for Kubernetes

**Note:** Before proceeding, ensure that you can communicate with your Kubernetes cluster using `kubectl`. Test your connection with `kubectl get nodes` or `kubectl cluster-info` to verify cluster access.

NeMo-Run uses [SkyPilot](https://docs.skypilot.co/en/latest/getting-started/installation.html#) to orchestrate jobs on kubernetes. SkyPilot has been installed with other dependencies. Validate that SkyPilot can connect to the EKS cluster and ready to submit jobs.

#### First get the list fo nodes in your cluster

```bash
kubectl get nodes
```

#### Label your GPU nodes for SkyPilot using the native GPU labeler:

SkyPilot provides a built-in GPU labeler that automatically detects and labels all GPU nodes in your cluster with the correct accelerator types.

```bash
# Use SkyPilot's native GPU labeler to automatically detect and label all GPU nodes
python -m sky.utils.kubernetes.gpu_labeler
```

This command will automatically:
- Detect all GPU nodes in your cluster
- Identify the GPU types (L40S, H100, A100, etc.)
- Apply the correct `skypilot.co/accelerator` labels with lowercase names
- Handle multiple node types if your cluster has mixed GPU types

#### ensure skypilot is connected to your cluster

```bash
sky check
```

For more information on node labeling, see the [SkyPilot Kubernetes documentation](https://docs.skypilot.co/en/latest/reference/kubernetes/kubernetes-setup.html#setting-up-gpu-labels).

### Initialize Git Repository

Because we are using GitArchivePackager, our directory must be a Git repository. Add and commit any relevant file that should be copied over to your job execution environment, below I am commiting the entire directory

```bash
# Initialize Git repository if not already done
git init

# Add and commit your training script
git add .
git commit -m "Add training scripts and configuration"
```

### Verifying PVC Setup

Before running your training jobs, ensure your PVC is properly configured:

```bash
# Check available PVCs in your cluster
kubectl get pvc

# Verify the PVC details (replace 'fsx-claim' with your actual PVC name)
kubectl describe pvc fsx-claim
```

## 4. Data Preprocessing for Custom Datasets (Pretraining Only)

### Overview

The repository includes automated data preprocessing capabilities that allow you to prepare custom datasets from Hugging Face for NeMo **pretraining**. This preprocessing step is **only required for pretraining workflows using custom datasets** and involves:

> **Important**: This section is **only required** if you plan to run pretraining with custom datasets using the main `pretrain_custom_dataset.py` script. This preprocessing is **NOT needed** for:
> - Pretraining with mock data (`pretrain_mock_dataset.py`)
> - Finetuning workflows (`finetune_default_dataset.py` and `finetune_custom_dataset.py`) - these handle data processing automatically

1. **Dataset Download**: Automatically downloads datasets from Hugging Face Hub
2. **Tokenizer Setup**: Downloads and configures GPT-2 tokenizer files
3. **Data Formatting**: Converts dataset to JSONL format compatible with NeMo
4. **Automated Processing Pod**: Deploys a Kubernetes pod for data processing

### Automated Data Processing Script

The `data-processing/data-processing.sh` script provides a complete automation solution for dataset preparation:

```bash
cd data-processing/
```

### Running the Data Processing

**Note**: Only run this if you plan to use the pretraining workflow with custom datasets (`pretrain_custom_dataset.py`).

1. **Deploy the Processing Pod**:
   ```bash
   # Example: Process the WikiText dataset (default)
   ./data-processing.sh deploy
   
   # Or explicitly specify the wikitext dataset
   ./data-processing.sh deploy --dataset-name wikitext --dataset-config wikitext-103-v1
   ```

2. **Access the Pod and Run Initial Processing**:
   ```bash
   # Access the pod shell
   ./data-processing.sh exec

   # Check pod status
   ./data-processing.sh status
   
   # Inside the pod, navigate to your mount path and run the script
   cd /mnt/nemo
   python /scripts/load_dataset.py
   ```

3. **Preprocess Data for Megatron Training**:
   ```bash
   # Still inside the pod, run the Megatron preprocessing script
   cd /tmp
   python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
      --input=/mnt/nemo/data/train_dataset.jsonl \
      --json-keys=text \
      --tokenizer-library=megatron \
      --tokenizer-type=GPT2BPETokenizer \
      --vocab=/mnt/nemo/data/tokenizer/gpt2-vocab.json \
      --merge-file=/mnt/nemo/data/tokenizer/gpt2-merges.txt \
      --dataset-impl=mmap \
      --output-prefix=/mnt/nemo/data/processed_data/train_dataset \
      --append-eod \
      --workers=4
   ```

      #### Processed Data Structure

      After processing, your data will be organized as follows (Using /mnt/nemo mount path as an example):
      ```
      /mnt/nemo/data/
      ├── tokenizer/
      │   ├── gpt2-vocab.json
      │   └── gpt2-merges.txt
      ├── processed_data/
      │   └── train_dataset/
      └── train_dataset.jsonl
      ```

4. **Exit the pod after data processing**:
   ```bash
   exit
   ```

5. **Cleanup Pod After Processing**:
   ```bash
   ./data-processing.sh delete
   cd ..
   ```

#### Available Commands

| Command | Description |
|---------|-------------|
| `deploy` | Deploy the data processing pod with specified dataset configuration |
| `exec` | Execute interactive shell in the running pod |
| `status` | Check the status of the pod and ConfigMap |
| `logs` | View pod logs |
| `delete` | Delete the pod and ConfigMap |
| `help` | Show detailed help information |

#### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--pvc-name` | Name of the PVC to mount | `fsx-claim` |
| `--mount-path` | Mount path in the container | `/mnt/nemo` |
| `--dataset-name` | Hugging Face dataset name | `wikitext` |
| `--dataset-config` | Dataset configuration (ignore for datasets without configs) | `wikitext-103-v1` |

## 5. Launching NeMo Training Jobs

> **Note**: The AWS-optimized container with EFA support can only be used for EFA enabled instances. For non-EFA usage, the default NeMo container (`nvcr.io/nvidia/nemo:26.02`) will work fine and you can omit the `--container_image` parameter.

### Overview

The repository provides multiple training scenarios to meet different needs:

**Pretraining Options:**
- `pretrain_custom_dataset.py` - **Pretraining with custom preprocessed datasets** (requires [Section 4 preprocessing](#data-preprocessing-for-custom-datasets-pretraining-only))
- `pretrain_mock_dataset.py` - Pretraining with mock data for testing (no preprocessing required)

**Finetuning Options:**
- `finetune_default_dataset.py` - Finetuning using NeMo's default datasets/recipes (no preprocessing required)
- `finetune_custom_dataset.py` - Finetuning with custom datasets from Hugging Face (automatic data processing included)

> **Data Processing Requirements**:
> - **Preprocessing Required**: Only `pretrain_custom_dataset.py` (pretraining with custom datasets)
> - **No Preprocessing Needed**: All other scripts handle data processing automatically or use mock data

### Training Parameters

#### Common Parameters (All Scripts):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max_steps` | Maximum number of training steps | 200 |
| `--nodes` | Number of compute nodes | 1 |
| `--gpus` | GPU type (e.g., L40S, H100, A10G) | L40S |
| `--gpu-devices` | Number of GPUs per node | 4 |
| `--efa-devices` | Number of EFA devices per node | None |
| `--container_image` | Container image for training (required for using EFA) | nvcr.io/nvidia/nemo:26.02 |
| `--env_vars_file` | JSON file with environment variables | env_vars.json |
| `--pvc_name` | Name of the Persistent Volume Claim to use | fsx-claim |
| `--pvc_mount_path` | Path where the PVC should be mounted in the container | /mnt/nemo |

> **Note on EFA Devices**: The `--efa-devices` parameter is only needed when using instances that have EFA (Elastic Fabric Adapter) support for high-performance networking. Specify the number of EFA devices an instance supports. For example, for P4 instances, use `--efa-devices 4`, for P5.48xlarge instances, use `--efa-devices 32`.

#### Additional Parameters for Pretraining (pretrain_custom_dataset.py):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--seq_length` | Sequence length for the dataset | 1024 |
| `--micro_batch_size` | Micro batch size for the dataset | 2 |
| `--global_batch_size` | Global batch size (auto-calculated if not specified) | None |

#### Additional Parameters for Finetuning Scripts:

| Parameter | Description | Default | Scripts |
|-----------|-------------|---------|---------|
| `--hf_token` | Hugging Face token for accessing gated models | None | Both finetuning scripts |
| `--disable_lora` | Disable LoRA finetuning (LoRA is enabled by default) | False | Both finetuning scripts |
| `--seq_length` | Sequence length for the dataset | 2048 | finetune_custom_dataset.py |
| `--micro_batch_size` | Micro batch size for the dataset | 1 | finetune_custom_dataset.py |
| `--global_batch_size` | Global batch size | 8 | finetune_custom_dataset.py |

### Pretraining Jobs

#### Option 1: Pretraining with Custom Datasets (pretrain_custom_dataset.py)

**Prerequisites**: Complete the [data preprocessing step](#data-preprocessing-for-custom-datasets-pretraining-only) above.

This is the main pretraining script that uses preprocessed custom datasets. It expects the data to be processed and available in the standard directory structure.

**For EFA enabled instances (using AWS-optimized container):**
For example, using a p5.48xlarge instance with EFA
```bash
python pretrain_custom_dataset.py \
    --max_steps 200 \
    --nodes 1 \
    --gpus H100 \
    --gpu-devices 8 \
    --efa-devices 32 \
    --container_image $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG \
    --env_vars_file env_vars.json \
    --pvc_name fsx-claim \
    --pvc_mount_path /mnt/nemo \
    --seq_length 1024 \
    --micro_batch_size 2 \
    --global_batch_size 512
```

**For all other instance types without EFA (using default NeMo container):**
For example, using a g6e.24xlarge instance without EFA
```bash
python pretrain_custom_dataset.py \
    --max_steps 200 \
    --nodes 1 \
    --gpus L40S \
    --gpu-devices 4 \
    --env_vars_file env_vars.json \
    --pvc_name fsx-claim \
    --pvc_mount_path /mnt/nemo \
    --seq_length 1024 \
    --micro_batch_size 2 \
    --global_batch_size 512
```

#### Option 2: Pretraining with Mock Data (pretrain_mock_dataset.py)

**Prerequisites**: None - this script uses mock data and does not require the data preprocessing step.

For quick testing or when you want to train with mock data without custom dataset preprocessing.

```bash
python pretrain_mock_dataset.py \
    --max_steps 200 \
    --nodes 1 \
    --gpus L40S \
    --gpu-devices 4 \
    --container_image nvcr.io/nvidia/nemo:26.02 \
    --env_vars_file env_vars.json \
    --pvc_name fsx-claim \
    --pvc_mount_path /mnt/nemo
```

> **Note**: If you're using EFA-compatible instances with the EFA optimized image, add the `--efa-devices` parameter to the command above.

### Finetuning Jobs

#### Option 3: Finetuning with Default Dataset (finetune_default_dataset.py)

**Prerequisites**: None - this script handles all data processing automatically.

This script performs finetuning using NeMo's built-in datasets and recipes. It supports both full finetuning and LoRA (Low-Rank Adaptation) finetuning, with LoRA enabled by default for efficiency.

**Features:**
- **Model Support**: Uses Gemma-2-2B by default (Llama-3-8B options available via code modification)
- **LoRA Support**: Enabled by default, can be disabled with `--disable_lora`
- **Checkpoint Conversion**: Automatically downloads and converts model checkpoints from Hugging Face
- **Gated Model Access**: Supports Hugging Face tokens for accessing gated models

**Basic Usage:**
```bash
python finetune_default_dataset.py \
    --max_steps 200 \
    --nodes 1 \
    --gpus L40S \
    --gpu-devices 4 \
    --container_image nvcr.io/nvidia/nemo:26.02 \
    --env_vars_file env_vars.json \
    --pvc_name fsx-claim \
    --pvc_mount_path /mnt/nemo
```

**With Hugging Face Token (for gated models):**
```bash
python finetune_default_dataset.py \
    --max_steps 200 \
    --nodes 1 \
    --gpus L40S \
    --gpu-devices 4 \
    --container_image nvcr.io/nvidia/nemo:26.02 \
    --env_vars_file env_vars.json \
    --pvc_name fsx-claim \
    --pvc_mount_path /mnt/nemo \
    --hf_token your_huggingface_token_here
```

**Full Finetuning (disable LoRA):**
```bash
python finetune_default_dataset.py \
    --max_steps 200 \
    --nodes 1 \
    --gpus L40S \
    --gpu-devices 4 \
    --container_image nvcr.io/nvidia/nemo:26.02 \
    --env_vars_file env_vars.json \
    --pvc_name fsx-claim \
    --pvc_mount_path /mnt/nemo \
    --disable_lora
```

> **Note**: If you're using EFA-compatible instances with the EFA optimized image, add the `--efa-devices` parameter to the command above.

#### Option 4: Finetuning with Custom Dataset (finetune_custom_dataset.py)

**Prerequisites**: None - this script includes automatic data processing for custom datasets.

This script performs finetuning on custom datasets from Hugging Face. It includes a comprehensive data processing pipeline that handles dataset download, preprocessing, and splitting automatically.

**Features:**
- **Custom Dataset Support**: Uses Databricks Dolly-15k dataset by default (easily configurable)
- **Automatic Data Processing**: Downloads, preprocesses, and splits data automatically
- **Flexible Configuration**: Supports custom sequence lengths and batch sizes
- **LoRA Support**: Same LoRA capabilities as the default dataset script
- **Data Validation**: Handles various dataset formats and structures

**Default Configuration**: Uses `databricks/databricks-dolly-15k` dataset. To change the dataset, modify the `DATASET_NAME` variable at the top of the script.

**Basic Usage:**
```bash
python finetune_custom_dataset.py \
    --max_steps 200 \
    --nodes 1 \
    --gpus L40S \
    --gpu-devices 4 \
    --container_image nvcr.io/nvidia/nemo:26.02 \
    --env_vars_file env_vars.json \
    --pvc_name fsx-claim \
    --pvc_mount_path /mnt/nemo \
    --seq_length 2048 \
    --micro_batch_size 1 \
    --global_batch_size 8
```

**Advanced Configuration:**
```bash
python finetune_custom_dataset.py \
    --max_steps 500 \
    --nodes 2 \
    --gpus L40S \
    --gpu-devices 4 \
    --container_image nvcr.io/nvidia/nemo:26.02 \
    --env_vars_file env_vars.json \
    --pvc_name fsx-claim \
    --pvc_mount_path /mnt/nemo \
    --seq_length 4096 \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --hf_token your_huggingface_token_here
```

> **Note**: If you're using EFA-compatible instances with the EFA optimized image, add the `--efa-devices` parameter to the command above.

#### Customizing Datasets

To use a different dataset in `finetune_custom_dataset.py`:

1. **Change the Dataset**: Modify the `DATASET_NAME` variable at the top of the script:
   ```python
   DATASET_NAME = "your-dataset/dataset-name"
   ```

2. **Adapt Data Processing**: If your dataset has a different structure, modify the `_preprocess_and_split_data` method in the `CustomDataModule` class to match your dataset's field names and structure.

### Important Notes

- **LoRA vs Full Finetuning**: LoRA is more memory efficient and faster, making it ideal for most finetuning scenarios. Use `--disable_lora` only if you need full parameter finetuning.
- **Model Selection**: Both finetuning scripts use Gemma-2-2B by default. You can modify the code to use other models like Llama-3-8B by uncommenting the relevant lines.
- **Multi-node Training**: When using multiple nodes (`--nodes > 1`), the scripts automatically configure the appropriate launcher (torchrun).
- **Checkpoint Management**: All scripts include automatic checkpoint conversion and management, downloading models from Hugging Face as needed.

## 6. Monitoring and Debugging

### NeMo-Run Monitoring

NeMo-Run provides comprehensive monitoring capabilities for tracking your training jobs:

#### Check Experiment Status

```bash
# View status of a job
$ nemo experiment status <experiment id e.g. aws-nemo2-pretrain-20250522-194450_1747943090>
```

Example output for a running job:

```
───────────────────────────────────────── Entering Experiment aws-nemo2-pretrain-20250522-194450 with id: aws-nemo2-pretrain-20250522-194450_1747943090 ──────────────────────────────────────────

Experiment Status for aws-nemo2-pretrain-20250522-194450_1747943090

Task 0: pretraining
- Status: RUNNING
- Executor: SkypilotExecutor
- Job id: aws-nemo2-pretrain-20250522-194450_1747943090___aws-nemo2-pretrain-20250522-194450_1747943090___pretraining___1
- Local Directory: /home/sagemaker-user/.nemo_run/experiments/aws-nemo2-pretrain-20250522-194450/aws-nemo2-pretrain-20250522-194450_1747943090/pretraining
```

#### View Training Logs

```bash
# Get logs from a running job
nemo experiment logs <experiment id e.g. aws-nemo2-pretrain-20250522-194450_1747943090> 0
```

Example training log output:

```
───────────────────────────────────────── Entering Experiment aws-nemo2-pretrain-20250522-194450 with id: aws-nemo2-pretrain-20250522-194450_1747943090 ──────────────────────────────────────────
[20:33:17] Fetching logs for pretraining                                                                                                                          
├── Waiting for task resources on 1 node.
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)

(pretraining, pid=4112) num_nodes=1
(pretraining, pid=4112) head_node_ip=10.1.99.25
...
(pretraining, pid=4112) [NeMo I 2025-05-22 20:08:53 model_checkpoint:497] Scheduled async checkpoint save for /mnt/nemo/experiments/aws-nemo2-pretrain-20250522-194450/checkpoints/model_name=0--val_loss=0.00-step=18-consumed_samples=9728.0-last.ckpt
(pretraining, pid=4112) Training epoch 1, iteration 0/999 | lr: 2.999e-06 | global_batch_size: 512 | global_step: 19 | reduced_train_loss: 11.03 | train_step_timing in s: 46.86 | consumed_samples: 10240
(pretraining, pid=4112) [NeMo I 2025-05-22 20:09:41 model_checkpoint:522] Async checkpoint save for step 19 finalized successfully.
(pretraining, pid=4112) Training epoch 1, iteration 10/999 | lr: 4.498e-06 | global_batch_size: 512 | global_step: 29 | reduced_train_loss: 11.03 | train_step_timing in s: 46.84 | consumed_samples: 15360
```

You can also use SkyPilot to monitor your jobs:

#### View Job Queue
```bash
# View running jobs
sky queue
```

Example output:
```
Fetching and parsing job queue...

Job queue of cluster aws-nemo2-pretrain-20250522-194450_1747943090
ID  NAME         SUBMITTED    STARTED      DURATION  RESOURCES   STATUS   LOG                                        
1   pretraining  35 mins ago  35 mins ago  35m 14s   1x[L40S:4]  RUNNING  ~/sky_logs/sky-2025-05-22-19-44-54-587711  
```

#### View Job Logs
```bash
# Get logs from a running job
sky logs <job-id>
```

Example training log output:
```
Tailing logs of the last job on cluster 'aws-nemo2-pretrain-20250522-194450_1747943090'...
Job ID not provided. Streaming the logs of the latest job.
├── Waiting for task resources on 1 node.
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)

(pretraining, pid=4112) num_nodes=1
(pretraining, pid=4112) head_node_ip=10.1.99.25
...
(pretraining, pid=4112) Training epoch 2, iteration 7/999 | lr: 6.897e-06 | global_batch_size: 512 | global_step: 45 | reduced_train_loss: 11.02 | train_step_timing in s: 47.12 | consumed_samples: 23552
(pretraining, pid=4112) Training epoch 2, iteration 8/999 | lr: 7.046e-06 | global_batch_size: 512 | global_step: 46 | reduced_train_loss: 11.02 | train_step_timing in s: 47.15 | consumed_samples: 24064
(pretraining, pid=4112) [NeMo I 2025-05-22 20:31:49 model_checkpoint:497] Scheduled async checkpoint save for /mnt/nemo/experiments/aws-nemo2-pretrain-20250522-194450/checkpoints/model_name=0--val_loss=0.00-step=47-consumed_samples=24576.0-last.ckpt
(pretraining, pid=4112) Training epoch 2, iteration 10/999 | lr: 7.346e-06 | global_batch_size: 512 | global_step: 48 | reduced_train_loss: 11.02 | train_step_timing in s: 46.81 | consumed_samples: 25088
(pretraining, pid=4112) [NeMo I 2025-05-22 20:32:36 model_checkpoint:522] Async checkpoint save for step 48 finalized successfully.
```

## 7. References

- [NVIDIA NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/index.html)
- [NVIDIA NeMo GitHub Repository](https://github.com/NVIDIA/NeMo)
- [NeMo Resiliency Examples](https://github.com/NVIDIA/NeMo/tree/main/examples/llm/resiliency)
- [AWS SageMaker HyperPod Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html)
- [AWS SageMaker HyperPod Workshop](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US/00-setup)
- [SkyPilot Documentation](https://docs.skypilot.co/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)