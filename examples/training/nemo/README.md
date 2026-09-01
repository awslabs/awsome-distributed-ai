# NVIDIA NeMo 2.0 Distributed Training

This test case contains examples and configurations for running distributed training with NVIDIA NeMo 2.0.

## Overview

[NVIDIA NeMo](https://developer.nvidia.com/nemo-framework) is a cloud-native framework for training and deploying generative AI models, optimized for architectures ranging from billions to trillions of parameters. NeMo 2.0 introduces a Python-based configuration system, providing enhanced flexibility, better IDE integration, and streamlined customization for large language model training.


- **Comprehensive development tools** for data preparation, model training, and deployment.
- **Advanced customization** for fine-tuning models to specific use cases.
- **Optimized infrastructure** with multi-GPU and multi-node support.
- **Enterprise-grade features** such as parallelism techniques, memory optimization, and deployment pipelines.

NeMo 2.0 introduces a Python-based configuration system, providing enhanced flexibility, better IDE integration, and streamlined customization.

## Slurm-based Deployment

The [slurm](./slurm/) directory provides implementation examples for running NeMo 2.0 using Slurm as the workload manager. This approach leverages AWS's purpose-built infrastructure for large-scale AI training. See the [README in the slurm directory](./slurm/README.md) for detailed setup and usage instructions.

## Kubernetes based Deployment

The [kubernetes](./kubernetes/) directory provides implementation examples for running NeMo 2.0 using Kubernetes as the orhestrator. This approach leverages AWS EKS for large-scale AI training. See the [README in the kubernetes directory](./kubernetes/README.md) for detailed setup and usage instructions.
