# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Training and evaluation image for the agentic-coding-sft use case.
# Built on the AWS PyTorch Deep Learning Container (DLC). The AWS_REGION build arg selects the
# regional DLC mirror; override it to match the region of your ECR / cluster.
ARG AWS_REGION=us-east-1
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-ec2

WORKDIR /workspace
COPY src/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt
COPY src /workspace

ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/tmp/hf \
    PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1