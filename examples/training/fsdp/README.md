# Get Started Training Llama 2, Mixtral 8x7B, and Mistral Mathstral with PyTorch FSDP in 5 Minutes

This content provides a quickstart with multinode PyTorch [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) training on Slurm and Kubernetes.
It is designed to be simple with no data preparation or tokenizer to download, and uses Python virtual environment.

## Prerequisites

To run FSDP training, you will need to create a training cluster based on Slurm or Kubermetes with an [Amazon FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html)
You can find instruction how to create a Amazon SageMaker Hyperpod cluster with [Slurm](https://catalog.workshops.aws/sagemaker-hyperpod/en-US), [Kubernetes](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US) or with in [Amazon EKS](../../architectures).

## FSDP Training

This fold provides examples on how to train with PyTorch FSDP with Slurm or Kubernetes.
You will find instructions for [Slurm](slurm) or [Kubernetes](kubernetes) in the subdirectories.