# Running Megatron-LM on Kubernetes

This directory contains Kubernetes-specific instructions and templates for setting up and running MegatronLM on an EKS cluster.

## 1. Preparation

Ensure you have the following prerequisites:

- A functional EKS cluster on AWS with EFA device plugin and NVIDIA device plugin deployed.
- Docker installed for building the container image.
- An FSx for Lustre filesystem mounted via a persistent volume claim on `/fsx` in EKS pods. An example of setting up FSx on EKS is available [here](https://docs.aws.amazon.com/eks/latest/userguide/fsx-csi-create.html).
- To run distributed training jobs as described in this guide, you must also have the [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/) installed and configured on your EKS cluster. Please follow the [official Kubeflow Training Operator installation guide](https://www.kubeflow.org/docs/components/training/overview/) to set it up before proceeding with the training steps.



### 2. Building the Container

1. Copy the megatron-lm.Dockerfile to your local machine.

2. Build the containerimage:

```bash
docker build -t aws-megatron-lm -f aws-megatron-lm.Dockerfile .
```

3. Tag and push the image to your container registry:

```bash
export AWS_REGION=us-east-1  # Set to the AWS region where your EKS cluster and ECR repository are located
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com
export ECR_REPOSITORY_NAME=aws-megatron-lm
export REPO_URI=${REGISTRY}/${ECR_REPOSITORY_NAME}:latest

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names ${ECR_REPOSITORY_NAME} --region ${AWS_REGION} 2>/dev/null || \
aws ecr create-repository --repository-name ${ECR_REPOSITORY_NAME} --region ${AWS_REGION}

# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${REGISTRY}

docker tag ${ECR_REPOSITORY_NAME}:latest ${REGISTRY}/${ECR_REPOSITORY_NAME}:latest
docker push ${REGISTRY}/${ECR_REPOSITORY_NAME}:latest
```

Now you are all set for distributed training with Megatron-LM on EKS! Proceed to the subdirectories for detailed instructions for different model training.