# Use AWS PyTorch DLC with EFA support as base
# PyTorch 2.9.0 with CUDA 13.0
# AWS_REGION controls which regional ECR to pull the DLC from.
# Override at build time: docker build --build-arg AWS_REGION=us-east-2 ...
ARG AWS_REGION=us-west-2
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.9.0-gpu-py312-ec2

# Set working directory
WORKDIR /app

# Install system dependencies (git for datasets, no CUDA toolkit needed for PyPI install)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Note: torch is already in the base DLC image, but requirements.txt pins
# the version for reproducibility across environments
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install NVRx from PyPI (includes async_ckpt, local_ckpt, inprocess modules)
# The PyPI wheel comes pre-compiled with CUPTI extension -- no CUDA headers needed
RUN pip install --no-cache-dir nvidia-resiliency-ext==0.4.1

# flash_attn (from base DLC image) was compiled against CUDA 12 and requires
# libcudart.so.12, but the base image ships CUDA 13.0. Install CUDA 12 runtime
# for binary compatibility.
RUN pip install --no-cache-dir nvidia-cuda-runtime-cu12 && \
    ln -sf /usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so.12 && \
    ldconfig 2>/dev/null || true

# Verify NVRx installation and check available modules
RUN python -c "\
from nvidia_resiliency_ext.checkpointing.async_ckpt.torch_ckpt import TorchAsyncCheckpoint; \
print('Async checkpoint module: OK'); \
from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import LocalCheckpointManager; \
print('Local checkpoint module: OK'); \
from nvidia_resiliency_ext.inprocess import Wrapper; \
print('In-process restart module: OK'); \
"

# Copy training code and utilities
COPY src/ /app/src/
COPY prepare_dataset.py /app/

# Set environment variables
ENV PYTHONPATH=/app
ENV NCCL_DEBUG=INFO
ENV NVIDIA_RESILIENCY_EXT_VERSION=0.4.1
# Note: FI_EFA_USE_DEVICE_RDMA=1 is only for P5 instances (GPUDirect RDMA).
# For P4de, do NOT set it -- the EFA plugin will use non-RDMA transport.
# Set it in the K8s manifest env section if using P5.

# Default command (overridden by K8s manifests)
CMD ["bash"]
