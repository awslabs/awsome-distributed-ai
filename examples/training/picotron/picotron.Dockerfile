ARG EFA_INSTALLER_VERSION=1.37.0
ARG AWS_OFI_NCCL_VERSION=v1.13.2-aws
ARG NCCL_VERSION=v2.23.4-1
ARG NCCL_TESTS_VERSION=v2.13.10
FROM public.ecr.aws/hpc-cloud/nccl-tests:efa${EFA_INSTALLER_VERSION}-ofi${AWS_OFI_NCCL_VERSION}-nccl${NCCL_VERSION}-tests${NCCL_TESTS_VERSION}

SHELL ["/bin/bash", "-c"] 
RUN apt-get update \
    && apt-get install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y python3.10 python3.10-dev 

RUN git clone https://github.com/huggingface/picotron  \
    && cd picotron \
    && git checkout 59714b1bf255 \
    && pip3 install torch==2.1.0 \
    && pip3 install -e .

# Installation instructions from: https://developer.nvidia.com/nsight-systems/get-started
RUN apt-get update && apt-get install -y wget \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y nsight-compute-2025.1.1 nsight-systems-2024.6.2 \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /picotron
COPY train.py .
COPY create_config.py .
