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
    git git-lfs wget curl ca-certificates zstd \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Install EFA user-space libraries (kernel module provided by the host).
# NOTE: the EFA installer shells out to apt for its own dependencies
# (environment-modules, tcl, ...), so the apt index must be present here. We
# refresh it inside this layer (the previous layer purged /var/lib/apt/lists)
# and purge again at the end to keep the layer small.
ARG EFA_INSTALLER_VERSION=1.47.0
RUN apt-get update && cd /tmp && \
    curl -sL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz | tar xz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    cd /tmp && rm -rf aws-efa-installer && \
    rm -rf /var/lib/apt/lists/*

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

# The NVIDIA base image ships its own opencv (4.7.0) whose native libraries live
# in site-packages/cv2 and are NOT owned by a pip distribution, so `pip uninstall`
# leaves them behind. requirements.txt also pulls `opencv-python` (4.11.x), and
# the mix breaks `import cv2` (cv2/typing references cv2.dnn.DictValue, removed in
# >=4.10; the loaded native module is 4.7.0). Remove the stale cv2 tree entirely,
# then install a single headless build so the dataloader imports cleanly.
RUN pip uninstall -y opencv opencv-python opencv-python-headless opencv-contrib-python || true && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2 \
           /usr/local/lib/python3.10/dist-packages/opencv* && \
    pip install --no-cache-dir opencv-python-headless==4.11.0.86

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
