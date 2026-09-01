# Running Small LLM with 3D Parallelism on Picotron 

This guide explains how to run distributed training using 3D parallelism with Picotron, using a 1.7B parameter LLaMA-based model as an example.

## What is 3D Parallelism?

3D parallelism combines three types of model parallelism to efficiently train large language models:

1. **Data Parallelism (DP)**: Splits training batches across multiple GPUs, with each GPU processing a subset of the data. This allows parallel processing of different batches to speed up training.

2. **Tensor Parallelism (TP)**: Splits individual tensors/layers across GPUs to distribute the model's parameters. This reduces memory requirements per device by partitioning large matrices.

3. **Pipeline Parallelism (PP)**: Splits the model vertically into stages that run on different GPUs in a pipelined fashion. This enables training of very deep models by distributing layers across devices.

This combination allows training of very large models that wouldn't fit on a single GPU while maintaining computational efficiency through balanced workload distribution and optimized communication patterns.

## Creating Training Configurations

The `create_config.py` script helps generate configuration files for different training scenarios. The script accepts various parameters to customize the training setup, including parallelism dimensions, model architecture, and training hyperparameters.

## Running Distributed Training

The training configurations are organized into two subdirectories:

- `ec2/`: Contains configurations for running on Amazon EC2 instances
- `slurm/`: Contains configurations for running on Slurm clusters
