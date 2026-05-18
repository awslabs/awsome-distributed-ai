# Knowledge Distillation on SageMaker HyperPod

This walkthrough demonstrates how to set up and run large language model (LLM) knowledge distillation workloads on Amazon SageMaker HyperPod using PyTorch and Huggingface's TRL (adapted from Arcee's DistillKit). Our setup leverages a distributed training architecture to efficiently transfer knowledge from a large teacher model to a smaller student model.

## Repository Structure
```
3.test_cases/pytorch/distillation/
├── Dockerfile                 # Container definition for running distillation workloads
├── kubernetes/
│   └── distill.yaml          # Kubernetes configuration for distributed training
└── src/
    ├── distil_logits_cli.py  # Main distillation script with CLI interface
    ├── requirements.txt      # Python package dependencies
    └── setup.sh              # Environment setup script for PyTorch and dependencies
```

## Getting Started

**Compatible Instance Types**: The architectures and test cases in this repository are designed to work with GPU instances including P4d, P5, and P5en instance types for optimal distributed training performance.

### Setting Up the Environment

First, prepare your container image with all necessary dependencies:

```bash
# Clone the repository
cd ~
git clone https://github.com/awslabs/awsome-distributed-training/
cd awsome-distributed-training/3.test_cases/pytorch/distillation

# Set up environment variables
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/hpc-cloud
export REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com
```

### Building the Docker Image

Build your Docker image containing PyTorch, Transformers, and other required libraries:

```bash
# For standard Linux environments
docker build -t ${REGISTRY}/distill:pytorch2.7.1 .

# For Mac developers targeting Linux/AMD64
docker buildx build --platform linux/amd64 -t ${REGISTRY}/distill:pytorch2.7.1 .
```

### Pushing to Amazon ECR

Push the image to Amazon ECR for use with your EKS cluster:

```bash
# Create repository if needed
REGISTRY_COUNT=$(aws ecr describe-repositories --query 'repositories[?repositoryName==`distill`]' --output text | wc -l)
if [ "$REGISTRY_COUNT" -eq 0 ]; then
    aws ecr create-repository --repository-name distill
fi

# Login to ECR
echo "Logging in to $REGISTRY..."
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY

# Push image
docker image push ${REGISTRY}/distill:pytorch2.7.1
```

### Accessing Hugging Face Datasets

Our distillation process uses datasets from Hugging Face. We can either download them to FSx for Lustre or load them dynamically using the Hugging Face Datasets library. For this walkthrough, we'll use the on-the-fly approach with either the "mlabonne/FineTome-100k" or "webinstruct-sub-200k" dataset.

#### Creating a Hugging Face Token

To access these datasets:
1. Create a Hugging Face account if you don't have one
2. Generate an [access token](https://huggingface.co/docs/hub/main/en/security-tokens) with read permissions 
3. Keep this token for use in our Kubernetes manifest

### Setting up Hugging Face Hub Location

To store your distilled model artifacts, you'll need to configure a Hugging Face Hub location:

1. **Create a [model repository on Hugging Face Hub](https://github.com/huggingface/hub-docs/blob/main/docs/hub/models-adding-libraries.md):**
   - Go to https://huggingface.co/new
   - Create a new model repository (e.g., "your-username/distilled-llama-7b")
   - Make note of the full repository name

2. **Set the hub location environment variable:**
   ```bash
   export HUB_LOCATION="your-username/your-model-name"
   ```

### Setup Persistent Volume Claim(PVC) for fsx 

Depending on the save location, We need to setup an PVC for FSx to save the model artifacts. Please follow the link [here](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US/01-cluster/06-fsx-for-lustre) to setup FSx CSI Driver and PVC. You can skip this step if you have a model repository setup on Huggingface.


### Preparing the Kubernetes Manifest

First, install [envsubst](https://github.com/a8m/envsubst) if you don't have it already. Then generate your Kubernetes manifest from the template:

```bash
# Change to Kubernetes directory
cd kubernetes/

# Set environment variables
export IMAGE_URI=${REGISTRY}/distill:pytorch2.7.1
export INSTANCE_TYPE=ml.p5en.48xlarge
export NUM_NODES=2
export GPU_PER_NODE=8
export EFA_PER_NODE=16
export TOTAL_PROCESSES=$((GPU_PER_NODE * NUM_NODES))
export FI_PROVIDER=efa
export HF_TOKEN=<Your HuggingFace Token> # irrespective of the model artifacts save location, you need HF token (for model download)
export SAVE_LOCATION=both  # Options: hub, fsx, both
export HUB_LOCATION=<the hub location to drop the trained artifacts>
#add as needed
export TEACHER_MODEL="arcee-ai/Arcee-Spark"
export STUDENT_MODEL="Qwen/Qwen2-1.5B"
export NUM_SAMPLES=100
export NUM_EPOCHS=3
export OUTPUT_DIR=/fsx
```
If you are not using both or FSx lustre for model save location, comment out lines 34-36 and 104-105 on distill.yaml-template file.

envsubst is part of the [GNU gettext utilities](https://www.gnu.org/software/gettext/manual/html_node/envsubst-Invocation.html) and is used for environment variable substitution in text files.
You can use Q developer to get the setup instructions


```bash
# Generate manifest
cat distill.yaml-template | envsubst > distill.yaml
```

### Deploying the Training Job

Deploy your distillation job to the EKS cluster:

```bash
kubectl apply -f ./distill.yaml
```

You should see confirmation:
```
pytorchjob.kubeflow.org/distill created
```

### Monitoring the Training Job

Monitor the progress of your job:

```bash
# Check job status
kubectl get pytorchjob

# Check pod status
kubectl get pods
```

Note: It may take 3-4 minutes for pods to transition from ContainerCreating to Running when first launched.

To find the master pod and check training logs:

```bash
# Find the master node
kubectl logs distill-worker-0 | grep master_addr=

# View logs from the master node (adjust node number as needed)
kubectl logs -f distill-worker-0
```

## Understanding the Distillation Framework

Our workload uses a powerful model distillation framework adapted from DistillKit that:

1. Loads a large teacher model and a smaller student model
2. Trains the student model to mimic the teacher's predictions
3. Uses a combination of KL-divergence and cross-entropy loss
4. Handles vocabulary differences and applies temperature scaling
5. Saves the distilled model for deployment

The Accelerate launcher in our Kubernetes manifest configures the distributed training environment, handling process coordination and mixed precision training across multiple nodes.

## Customizing Your Distillation Job

When running the LLM distillation job on EKS, you have numerous parameters and configurations available to optimize the process. Let's explore the key customization options within the Kubernetes manifest.

### Accelerate Launch Command Customization

The accelerate launch command in our YAML manifest coordinates the distributed training environment. Here are the key parameters you can modify:

```bash
accelerate launch --num_processes=16 --num_machines=2 --machine_rank=0 \
  --main_process_ip=distill-worker-0 --main_process_port=23456 \
  --mixed_precision=bf16 --rdzv_backend=c10d --use_deepspeed distil_logits_cli.py
```

- `--num_processes`: Total number of processes across all nodes. Typically set to (GPUs per node × number of nodes)
- `--num_machines`: Total number of nodes participating in training
- `--machine_rank`: Rank of the current machine (0 for the master node)
- `--main_process_ip`: Hostname of the master node, typically set to the first worker pod name
- `--main_process_port`: Communication port for distributed coordination
- `--mixed_precision`: Precision mode for training:
  - `bf16`: BFloat16 precision (best for newer GPUs like A100, H100)
  - `fp16`: Half precision (good for older GPUs)
  - `no`: Full precision (slower but may be needed for some models)
- `--rdzv_backend`: Rendezvous backend for process coordination
- `--use_deepspeed`: Enable DeepSpeed integration for memory optimization

### Python Script Arguments

You can pass various arguments to the distil_logits_cli.py script by appending them to the accelerate command.

#### Model Selection
```bash
--teacher_model "arcee-ai/Arcee-Spark" \
--student_model "Qwen/Qwen2-1.5B"
```
Change these to specify different teacher and student models from Hugging Face.

#### Dataset Configuration
```bash
--dataset_name "mlabonne/FineTome-100k" \
--dataset_split "train" \
--num_samples 10000 \
--max_length 4096
```
- `dataset_name`: Choose different datasets for distillation (e.g., "webinstruct-sub-200k")
- `num_samples`: Limit the number of training examples (useful for testing)
- `max_length`: Maximum sequence length for tokenization

#### Training Hyperparameters
```bash
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--learning_rate 2e-5 \
--weight_decay 0.05 \
--warmup_ratio 0.1 \
--lr_scheduler_type "cosine"
```
These parameters control the training process and can be adjusted based on your hardware and model sizes.

#### Distillation Parameters
```bash
--temperature 2.0 \
--alpha 0.5
```
- `temperature`: Controls the "softness" of the teacher model's output distribution. Higher values produce a softer distribution
- `alpha`: Balances between distillation loss and cross-entropy loss. Higher values favor matching the teacher's outputs

#### Advanced Options
```bash
--use_flash_attention \
--layers_to_unfreeze "spectrum.yaml" \
--hub_location "YourUsername/distilled-model-v1"
```
- `use_flash_attention`: Enable Flash Attention 2 for faster training
- `layers_to_unfreeze`: Path to a YAML file specifying which layers to train
- `hub_location`: Where to publish the final model on Hugging Face Hub


## Configuration Options Reference

### Project Settings
- `--project_name`: Project name for logging (default: "distil-logits")

### Dataset Configuration
- `--dataset_name`: HuggingFace dataset name (default: "mlabonne/FineTome-100k")
- `--dataset_split`: Dataset split to use (default: "train")
- `--num_samples`: Limit number of samples (default: None - use all)
- `--seed`: Random seed for reproducibility (default: 42)

### Model Configuration
- `--teacher_model`: Teacher model name (default: "arcee-ai/Arcee-Spark")
- `--student_model`: Student model name (default: "Qwen/Qwen2-1.5B")
- `--use_flash_attention`: Enable Flash Attention 2 for memory efficiency

### Tokenizer Settings
- `--max_length`: Maximum sequence length (default: 4096)
- `--chat_template`: Custom chat template for tokenization

### Save Location Configuration
- `--save_location`: Where to save the model - "hub", "fsx", or "both" (default: "both")
- `--output_dir`: FSX output directory for checkpoints (default: "/fsx/model")
- `--hub_location`: HuggingFace Hub location for model upload (default: "Satyach/distilled-model-v1")

### Training Parameters
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per device (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 8)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay for regularization (default: 0.05)
- `--warmup_ratio`: Learning rate warmup ratio (default: 0.1)
- `--lr_scheduler_type`: LR scheduler type (default: "cosine")
- `--save_steps`: Save checkpoint every X steps (default: 1000)
- `--logging_steps`: Log metrics every X steps (default: 1)
- `--resume_from_checkpoint`: Path to resume training from checkpoint

### Precision Settings
- `--fp16`: Use FP16 mixed precision
- `--bf16`: Use BF16 mixed precision (default: True)

### Distillation Parameters
- `--temperature`: Temperature for knowledge distillation (default: 2.0)
- `--alpha`: Balance between distillation and task loss (default: 0.5)

### Advanced Configuration
- `--layers_to_unfreeze`: Path to YAML file specifying which layers to unfreeze (Spectrum)
- `--config_file`: Path to JSON/YAML config file (overridden by CLI args)

## Example Configurations


```yaml
command: 
  - /bin/bash
  - -c
  - |
    # Installation commands...
    accelerate launch --num_processes=16 --num_machines=2 --machine_rank=0 \
      --main_process_ip=distill-worker-0 --main_process_port=23456 \
      --mixed_precision=bf16 --rdzv_backend=c10d --use_deepspeed \
      distil_logits_cli.py \
      --save_location both \
      --teacher_model "arcee-ai/Arcee-Spark" \
      --student_model "Qwen/Qwen2-1.5B" \
      --dataset_name "mlabonne/FineTome-100k" \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 16 \
      --output_dir /fsx \
      --use_flash_attention
```
