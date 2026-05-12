# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Extends the shared trl/Dockerfile which includes PyTorch, TRL, Flash Attention,
# DeepSpeed, and common training dependencies.
# Build the base image first: docker build -t trl-base:latest ../../

FROM trl-base:latest

# GPT-OSS specific: HyperPod elastic agent for EKS distributed training
# Fix torchvision compatibility with PyTorch 2.6.0
# Remove vLLM (installed by trl[vllm] in base) - incompatible ABI and not needed for this recipe
RUN pip install hyperpod-elastic-agent torchvision==0.21.0 langdetect trackio deep-translator && pip uninstall -y vllm
# Copy source code
WORKDIR /app
COPY src/ /app/

# Note: Container runs as root - required for NCCL/EFA device access in distributed training
ENV PYTHONPATH="${PYTHONPATH}:/app"
