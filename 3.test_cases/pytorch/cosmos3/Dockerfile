#
# Cosmos 3 action-policy post-training image (AWS Deep Learning Container base).
#
# WHY THE AWS DLC BASE (the EFA/NCCL multi-node story):
#   An NGC PyTorch base (nvcr.io/nvidia/pytorch) validates SINGLE-node but blocks
#   on MULTI-node: its baked-in /opt/amazon/aws-ofi-nccl is built against the
#   container's older NCCL, while the framework venv ships nvidia-nccl-cu13==2.28.9
#   (torch 2.10.0+cu130) -> ABI mismatch -> fi_getinfo() "No data available".
#
#   The AWS Deep Learning Container ships torch 2.10.0+cu130 (EXACT match to the
#   framework's cu130-train pin) with an AWS-tuned, version-MATCHED EFA stack
#   baked in (EFA 1.47.0 -> libfabric 2.4 + aws-ofi-nccl 1.18.0, gdrcopy 2.5.1).
#   Because the DLC's torch is the same 2.10.0+cu130 wheel the venv installs, its
#   NCCL (2.28.9) lines up with the DLC's aws-ofi-nccl build, so the multi-node
#   ABI mismatch does not arise -- no NGC-style workarounds
#   (NCCL_NET_PLUGIN=none / LD_LIBRARY_PATH="") are needed.
#
# VALIDATION OUTCOME (2-node EFA):
#   EFA/NCCL multi-node WORKS on this base: NCCL loaded the DLC's matched
#   aws-ofi-nccl 1.18.0 / Libfabric 2.4, selected provider efa (16 nics), RDMA.
#   That run then surfaced two DROID video-decode packaging issues (fixed below):
#     (1) Ubuntu 22.04 apt ffmpeg is 4.4 (libavutil.so.56); torchcodec 0.10.0
#         needs FFmpeg 5-8. Fixed below via prebuilt SHARED FFmpeg 8.
#     (2) The DLC builds system Python 3.13 WITHOUT --enable-shared, so
#         libpython3.13.so.1.0 is absent and torchcodec's custom-ops .so cannot
#         load. The framework's cosmos-deps wheels are cp313-ONLY (an AL2023 3.12
#         base was tried and rejected: flash-attn has no cp312 wheel). Fixed below
#         by building the venv on uv-MANAGED CPython 3.13, which bundles a SHARED
#         libpython3.13.so.1.0, then exposing it via ldconfig.

# pytorch-training 2.10.0, py313, CUDA 13.0, Ubuntu 22.04, EC2/EKS/ECS variant.
# Public registry (no ECR auth needed to pull the base):
#   https://github.com/aws/deep-learning-containers
ARG DLC_TAG=2.10.0-gpu-py313-cu130-ubuntu22.04-ec2
FROM public.ecr.aws/deep-learning-containers/pytorch-training:${DLC_TAG}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl xz-utils fio && \
    rm -rf /var/lib/apt/lists/*

# --- Prebuilt SHARED FFmpeg (8.x) onto the default loader path ----------------
# torchcodec 0.10.0 dlopen()s libav*.so.<ver>. Ubuntu 22.04 apt only ships
# FFmpeg 4.4 (libavutil.so.56), too old (torchcodec needs 5-8). The pinned BtbN
# gpl-shared build below provides libavutil.so.60 (FFmpeg 8) -> torchcodec loads.
# We install BOTH the shared libs (for torchcodec, action-policy + generation) AND the ffmpeg/ffprobe
# BINARIES (the vision-SFT dataloader's get_video_metadata shells out to `ffprobe`,
# needed by the vision-SFT stages) onto the default lib + bin paths.
# Installed to /usr/local/lib + ldconfig so we do NOT touch LD_LIBRARY_PATH
# (that carries the DLC's EFA/ofi-nccl stack we must not perturb).
#
# REPRODUCIBILITY NOTE: BtbN/FFmpeg-Builds is a ROLLING autobuild repo — old
# release assets are pruned over time, so a bare URL will eventually 404. We pin a
# specific asset AND verify its SHA-256 (FFMPEG_SHA256), so a silent CDN swap or a
# truncated download fails the build loudly instead of baking a bad lib. If the
# asset is gone, mirror the tarball to your own ECR/S3 (or pin a newer BtbN build)
# and update FFMPEG_URL + FFMPEG_SHA256 to match. Override both via --build-arg.
ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-06-18-14-21/ffmpeg-N-125093-gd2d371d10d-linux64-gpl-shared.tar.xz
ARG FFMPEG_SHA256=d322b8b53cc8068da2b1a511df60d3279493364c609c6c9b2baa9149243adff4
RUN cd /tmp && curl -sL --retry 3 -o ff.tar.xz "${FFMPEG_URL}" && \
    echo "${FFMPEG_SHA256}  ff.tar.xz" | sha256sum -c - && \
    mkdir -p ffx && tar -xf ff.tar.xz -C ffx --strip-components=1 && \
    cp -a ffx/lib/*.so* /usr/local/lib/ && \
    cp -a ffx/bin/ffmpeg ffx/bin/ffprobe /usr/local/bin/ && \
    ldconfig && rm -rf /tmp/ff.tar.xz /tmp/ffx && \
    ldconfig -p | grep -E "libavutil|libavcodec" | head && \
    /usr/local/bin/ffprobe -version | head -1

# uv for the framework's locked env.
COPY --from=ghcr.io/astral-sh/uv:0.10.8 /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy

# --- uv-managed CPython 3.13 (ships a SHARED libpython3.13.so.1.0) ------------
# The framework's cosmos-dependencies wheels (flash-attn, transformer-engine,
# natten) are published cp313-ONLY, so the venv must be Python 3.13. But the AWS
# DLC builds its system Python 3.13 WITHOUT --enable-shared, so there is no
# libpython3.13.so.1.0 -> torchcodec's libtorchcodec_custom_ops*.so cannot load.
# uv's managed standalone CPython 3.13 bundles the shared libpython; we use it for
# the venv and expose its libpython on the default loader path via ldconfig.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.13

WORKDIR /workspace

# Clone + install cosmos-framework (pin via --build-arg COSMOS_FRAMEWORK_REF) on
# the uv-managed Python 3.13. The cu130-train group pins
# torch 2.10.0+cu130 + NCCL 2.28.9 (the NCCL that matches the DLC's aws-ofi-nccl).
ARG COSMOS_FRAMEWORK_REF=f358a625875310096cf01d75b11b7818c438aa1e
RUN git clone https://github.com/NVIDIA/cosmos-framework.git && \
    cd cosmos-framework && git checkout ${COSMOS_FRAMEWORK_REF} && \
    uv sync --all-extras --group=cu130-train --python 3.13

# Expose the uv-managed shared libpython3.13.so.1.0 on the default loader path so
# torchcodec's custom-ops .so resolves it (NOT via LD_LIBRARY_PATH -> keep EFA path clean).
RUN LP=$(find /opt/uv-python -name 'libpython3.13.so.1.0' | head -1) && \
    test -n "$LP" && cp -a "$LP" /usr/local/lib/ && ldconfig && \
    ldconfig -p | grep libpython3.13

# Install the sample package (cosmos3_aws) onto PYTHONPATH.
COPY src /workspace/cosmos3-aws/src
COPY toml /workspace/cosmos3-aws/toml
ENV PYTHONPATH=/workspace/cosmos3-aws/src

# --- Runtime env: DLC EFA stack (DIFFERENT from the NGC-base workarounds) ---
# Do NOT set LD_LIBRARY_PATH="" here: on the DLC we WANT the AWS EFA libfabric +
# aws-ofi-nccl on the loader path. The DLC sets LD_LIBRARY_PATH to include
# /opt/amazon/efa/lib and /opt/amazon/ofi-nccl/lib already; we leave it intact.
# Do NOT set NCCL_NET_PLUGIN=none: that forces TCP cross-node. The DLC's tuner
# plugin is the matched one we want NCCL to load (the whole point of the DLC base).
ENV NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV FI_EFA_FORK_SAFE=1
ENV PYTHONHASHSEED=42

# venv torchrun on PATH.
ENV PATH="/workspace/cosmos-framework/.venv/bin:${PATH}"
WORKDIR /workspace/cosmos-framework
