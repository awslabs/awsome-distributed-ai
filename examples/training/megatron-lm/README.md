# MegatronLM Test Case

[MegatronLM](https://github.com/NVIDIA/Megatron-LM) is a framework from Nvidia designed for training large language models (LLMs). We recommend reading the following papers to understand the various tuning options available:

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [Reducing Activation Recomputatio in Large Transformer Models](https://arxiv.org/pdf/2205.05198)

To run a test case, follow these steps:

1. Prepare your environment.
2. Build a container, download, and preprocess the data.
3. Train the model.

We provide guidance for both Slurm and Kubernetes users. For detailed instructions, refer to the [slurm](./slurm) or [kubernetes](./kubernetes) subdirectories.