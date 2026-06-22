# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# PointWorld: 3D world-model pre-training (PTv3 + DINOv3).
#
# PointWorld pins CUDA 12.4 wheels (torch==2.5.1). We base on the matching
# NVIDIA PyTorch container (24.10-py3 ships torch 2.5 + CUDA 12.4 + NCCL 2.22+)
# and reinstall the exact pinned dependency set from the upstream
# environments/requirements.txt to guarantee reproducibility.
#
# NOTE: This image targets H200 (p5en.48xlarge). For B200, the NCCL/aws-ofi-nccl
# stack in this base is too old for B200 EFA networking; use a NeMo container
# with NCCL >= 2.29 instead (see README "Known Limitations").
FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git git-lfs wget curl ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Install EFA user-space libraries (kernel module provided by the host).
ARG EFA_INSTALLER_VERSION=1.47.0
RUN cd /tmp && \
    curl -sL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz | tar xz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    cd /tmp && rm -rf aws-efa-installer

# Clone PointWorld (pinned to a tested commit) and its third-party submodules.
ARG POINTWORLD_COMMIT=05484826dfef74cbe278a3974179a5a16705d35d
RUN git clone https://github.com/NVlabs/PointWorld.git /pointworld && \
    cd /pointworld && \
    git checkout ${POINTWORLD_COMMIT} && \
    git submodule update --init --recursive
WORKDIR /pointworld

# Install the canonical pinned dependency set (CUDA 12.4 wheels: torch 2.5.1,
# torch-scatter, spconv-cu124, webdataset, etc.). Using the upstream
# requirements.txt verbatim keeps this in lockstep with the pinned commit.
RUN pip install --no-cache-dir -r environments/requirements.txt

# huggingface_hub is used by the dataset / checkpoint download steps.
RUN pip install --no-cache-dir huggingface_hub==0.26.2

# timm provides PTv3 DropPath; install without transitive deps so it does not
# perturb the pinned torch stack.
RUN pip install --no-cache-dir timm==1.0.19 --no-deps

# flash-attn imports torch at build time, so it must be installed after the
# base torch is present. --no-build-isolation reuses the installed torch.
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# Keep urdfpy-compatible graph deps on a Python 3.10-safe networkx release.
RUN pip install --no-cache-dir networkx==3.4.2 --no-deps

# DINOv3 (scene encoder backbone) is a gated submodule. The submodule source is
# vendored above; its checkpoint weights are gated by Meta and must be supplied
# at run time on the shared filesystem. See scripts/2.download_dinov3.sh and the
# README "DINOv3 (gated dependency)" section. The training entrypoint reads the
# weights from third_party/dinov3/checkpoints/ which we symlink to /fsx at run
# time, so no gated artifact is baked into this image.

ENV PYTHONPATH="/pointworld:${PYTHONPATH}"
ENV LOCAL_DATASET_DIR="/dataset"

WORKDIR /pointworld
