# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
FROM public.ecr.aws/hpc-cloud/nccl-tests:latest

# Install Miniconda (Python 3.12)
RUN mkdir -p /opt/miniconda3 \
    && curl -L https://repo.anaconda.com/miniconda/Miniconda3-py312_25.3.1-1-Linux-x86_64.sh -o /tmp/Miniconda3.sh \
    && bash /tmp/Miniconda3.sh -b -f -p /opt/miniconda3 \
    && rm /tmp/Miniconda3.sh \
    && /opt/miniconda3/bin/conda init bash
ENV PATH="/opt/miniconda3/bin:${PATH}"

# Install Rust (required by TRL's 'outlines' dependency)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Core TRL dependencies (shared across all test cases)
RUN pip install -v \
    torch==2.6.0 \
    transformers>=4.48.0 \
    datasets>=2.17.0 \
    accelerate>=0.27.0 \
    peft>=0.9.0 \
    deepspeed>=0.14.0 \
    wandb

# TRL with vLLM backend
RUN pip install trl[vllm]

# FlashInfer for optimized inference
RUN pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/

# Flash Attention (must install separately with --no-build-isolation)
RUN pip install packaging ninja
RUN pip install flash-attn>=2.5.0 --no-build-isolation

# Additional dependencies for all test cases
RUN pip install \
    math_verify \
    bitsandbytes>=0.43.0 \
    scipy \
    sentencepiece \
    protobuf \
    langdetect \
    tiktoken \
    evaluate \
    trackio
