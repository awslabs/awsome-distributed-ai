# RLVR Recipe

This repository provides a complete setup for running reinforcement learning from verifiable rewards (RLVR) on EKS clusters using Ray and verl. RLVR trains language models using verifiable rewards from math and coding tasks, where correctness can be automatically verified. The project uses verl, an efficient RL training framework from ByteDance, to run algorithms like GRPO (Group Relative Policy Optimization) and DAPO (Direct Advantage Policy Optimization) on distributed GPU clusters.

## What is verl?

[verl (Volcano Engine Reinforcement Learning)](https://github.com/volcengine/verl) is a flexible, production-ready RL training library for large language models. It provides seamless integration with popular frameworks like FSDP, Megatron-LM, vLLM, and Ray, enabling efficient distributed training with state-of-the-art throughput. This repo includes the full verl codebase with custom run scripts optimized for HyperPod.

## What is RLVR?

[Reinforcement Learning from Verifiable Rewards (RLVR)](https://arxiv.org/abs/2506.14245) is a training approach where models learn from tasks with objectively verifiable outcomes, such as math problems or code execution. Unlike human preference-based RL, RLVR uses ground-truth correctness as the reward signal, making it particularly effective for reasoning tasks.

## Getting started

### Prerequisites

**Cluster**:
From here on out, we will assume you have an EKS cluster with GPU nodes (e.g., p5en.48xlarge). This example can be run on an EKS or HyperPod EKS cluster. 

This example was tested on 4 p5en.48xlarge nodes (8xH200 GPUs each). If you are using different node types, modify the cluster environment variables in `env_vars`. Feel free to change the model type/size, and training parameters to accomodate smaller or larger node types. 

**Storage**:
- This examples uses a FSx for Lustre file system that mounts to the pods via a pvc called `fsx-claim`. We store the dataset, as well as model checkpoints here. Feel free to substitute this claim with your own. 

**Versions**:
The example was tested on versions:
- EKS: 1.33
- KubeRay: 1.4.2
- VERL: v0.6.1

### Clone this repo
```bash
git clone https://github.com/awslabs/awsome-distributed-ai.git 
cd awsome-distributed-ai/examples/training/verl/kubernetes/rlvr
```

### Install verl repository
This repository contains the verl framework and scripts needed for RLVR training. We install it to get access to the distributed RL training algorithms (GRPO, DAPO, and more) and the integration code that connects verl with EKS/Ray clusters for scalable language model fine-tuning on math and coding tasks.

```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout v0.6.1
cd ..
```

### Create RayCluster

Install KubeRay operator to manage Ray clusters on Kubernetes:
```bash
./setup/install-kuberay.sh
```

Configure your cluster settings (AWS region, cluster name, GPU counts, model paths):
```bash
# Copy the example file and customize it with your values
cp setup/env_vars.example setup/env_vars
vim setup/env_vars
```

> **Important**: The `env_vars` file contains sensitive information like your HuggingFace token, AWS account details, and cluster IDs. This file is gitignored to prevent accidentally committing credentials. Always use `env_vars.example` as your template.

Load the environment variables into your shell session:
```bash
source setup/env_vars
```

Build a Docker image with verl, EFA networking support, and push to ECR:
```bash
./setup/build-push.sh
```

Generate kustomization.yaml from your environment variables and deploy the Ray cluster:
```bash
./setup/generate-kustomization.sh
kubectl apply -k setup/
```

Alternatively, you can combine both steps:
```bash
./setup/generate-kustomization.sh && kubectl apply -k setup/
```

> **Note**: Considerations before applying raycluster.yaml
> - Ensure you have a file system before applying the RayCluster. This raycluster.yaml is assuming you have a pvc in place called `fsx-claim`. Feel free to modify the configuration depending on your file system setup
> - This Raycluster is assuming you have 4 p5en.48xlarge instance types. Modify your setup/env_vars and NodeSelector in the yaml to adjust for your cluster. 


Download the GSM8K math dataset and prepare it for GRPO training:
```bash
./setup/load_data_grpo.sh
```

Forward the Ray dashboard to localhost for monitoring training progress:
```bash
./ray-expose.sh
```

Submit a GRPO training job to the Ray cluster. This trains a language model on math reasoning using group relative policy optimization:
```bash
./recipe/run_grpo_configurable.sh
```

The `verl/` directory contains the official verl framework, and `recipe/` includes custom run scripts (`run_grpo_configurable.sh`, `run_dapo_configurable.sh`) that integrate with your environment variables for easy configuration.

### Observability

For EKS:
Please see this documentation to set up Prometheus and Grafana dashboards for Ray clusters: [Using Prometheus & Grafana](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/prometheus-grafana.html)

For HyperPod EKS:
Check out the `observability/` directory to integrate Ray's native metrics dashboards with HyperPod's Amazon Managed Prometheus and Grafana