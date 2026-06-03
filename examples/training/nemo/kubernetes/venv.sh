#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -e

# Install build tools and development dependencies
echo "Installing build tools and development dependencies..."
sudo apt update
sudo apt install -y build-essential gcc g++ cmake python3-dev python3-setuptools libffi-dev libssl-dev

# Install Skypilot
pip install skypilot[kubernetes]==0.8.0

# Install NeMo-Run
pip install git+https://github.com/NVIDIA/NeMo-Run.git@4d056535b5cce475b0536243e2cefcfa3897eee8

# # Install PyTorch
pip install torch==2.6.0
 
# Install Megatron-LM
pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@b5d90de8e7c7fae5f35be89d665f237970540bed

# # Download and install Mamba SSM
wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl  # Adjusted for torch 2.0
pip install mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install NeMo Toolkit
pip install nemo_toolkit['all']==2.1.0

# Install OpenCC
pip install opencc==1.1.6

# install megatron core dependencies
pip install "nvidia-modelopt[torch]>=0.19.0" pytest_asyncio pytest_cov pytest_random_order

pip install nvidia-resiliency-ext

echo "Environment setup complete."