# TRL (Transformer Reinforcement Learning) Test Cases

This directory contains test cases for distributed training with [Hugging Face TRL](https://huggingface.co/docs/trl), a library for post-training LLMs using reinforcement learning techniques such as GRPO, PPO, DPO, and SFT.

## Base Docker Image

All test cases share a common base Docker image defined in [`Dockerfile`](Dockerfile). It includes Python 3.12, PyTorch 2.6.0, TRL with vLLM backend, Flash Attention, FlashInfer, and common training dependencies.

Build the shared base image:

```bash
cd examples/training/trl
docker build -t trl-base:latest .
```

## Test Cases

| Test Case | Description | Platform | Model |
|-----------|-------------|----------|-------|
| [grpo-math-reasoning](grpo-math-reasoning/) | Multi-node GRPO training for math reasoning | Slurm | Qwen2.5-72B + NuminaMath |
| [gpt-oss-lora-grpo](gpt-oss-lora-grpo/) | SFT LoRA + GRPO for multilingual reasoning | HyperPod EKS | GPT-OSS 20B |

## Prerequisites

- GPU cluster with EFA networking (see [`architectures/`](../../../architectures/) for cluster setup)
- Shared filesystem (e.g., Amazon FSx for Lustre) accessible from all nodes
- [Enroot](https://github.com/NVIDIA/enroot) and [Pyxis](https://github.com/NVIDIA/pyxis) for Slurm container execution
- Hugging Face model access tokens configured via `HF_HOME`
