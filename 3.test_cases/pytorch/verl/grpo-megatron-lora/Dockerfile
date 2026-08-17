# =============================================================================
# Dockerfile for VERL with Megatron LoRA and EFA support
# Extends verlai/verl vllm020.dev2 which matches verl v0.8.0's stable stack:
# vLLM 0.20.2, Megatron-Core 0.16.1, Python 3.12 (see verl v0.8.0
# docker/Dockerfile.stable.vllm).
# We add: EFA networking and Megatron-Bridge (LoRA).
# verl, transformers, and pip extras are installed at job submission time via
# runtime_env.yaml (passed to `ray job submit --runtime-env`). This allows
# fast iteration without rebuilding the image. See runtime_env.yaml.
#
# Migration note (verl v0.8.0 upgrade): base bumped vllm012.dev4 -> vllm020.dev2.
# Megatron-Bridge bumped v0.3.0 -> v0.5.0 — verl v0.8.0's PEFT checkpoint path
# imports create_peft_hook / load_peft_adapter_checkpoint / save_hf_adapter,
# which only exist at MB v0.5.0.
# =============================================================================
ARG BASE_TAG=vllm020.dev2
FROM verlai/verl:${BASE_TAG}

ARG MBRIDGE_VERSION=v0.5.0
ENV EFA_VERSION=1.47.0

# System dependencies for EFA and multi-node SSH
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    pciutils \
    libnl-3-dev \
    libnl-route-3-dev \
    libevent-dev \
    libhwloc-dev \
    iproute2 \
    ethtool \
    openssh-server \
    openssh-client \
    environment-modules \
    tcl \
    udev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

#################################################
## Clean up HPC-X to avoid conflicts with EFA
RUN rm -rf /opt/hpcx /usr/local/mpi && \
    rm -f /etc/ld.so.conf.d/hpcx.conf && ldconfig

#################################################
## EFA SETUP - Install EFA with all dependencies
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_VERSION}.tar.gz \
    && tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer-${EFA_VERSION}.tar.gz $HOME/aws-efa-installer

# EFA paths (order matters!)
ENV PATH="/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# OpenMPI configuration to use EFA and avoid conflicts
ENV OMPI_MCA_pml=^ucx
ENV OMPI_MCA_btl=tcp,self
ENV OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent
ENV OPAL_PREFIX=/opt/amazon/openmpi

# EFA/NCCL configuration for optimal performance
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV FI_EFA_FORK_SAFE=1
ENV FI_EFA_ENABLE_SHM_TRANSFER=1
ENV NCCL_PROTO=simple
ENV NCCL_NET_GDR_LEVEL=2
ENV NCCL_SOCKET_IFNAME=^docker,lo,veth
ENV NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-ofi-tuner.so
ENV PMIX_MCA_gds=hash

# Additional pip dependencies not in the base image or verl's setup.py
# NOTE: verl, transformers, and pip extras (scipy, mlflow, boto3, etc.) have been
# moved to runtime_env.yaml for fast iteration. Only infrastructure-level deps
# that require --no-deps or system-level fixes remain here.
#
# NOTE: pyext is NOT installed — it fails to build on Python 3.12 (uses removed
# inspect.getargspec). Code reward scoring uses Sandbox Fusion instead.

# Remove distutils-managed blinker 1.4 that pip cannot uninstall, then install a
# pip-managed copy so mlflow >=3.0 can upgrade it normally (via runtime_env).
RUN rm -rf /usr/lib/python3/dist-packages/blinker* \
           /usr/lib/python3.12/dist-packages/blinker* && \
    pip install --no-cache-dir blinker

#####################
# Install Megatron-Bridge for LoRA support (REQUIRED for Megatron LoRA)
# Provides the `megatron.bridge` namespace used by verl when vanilla_mbridge=False.
# --no-deps: megatron-core 0.15.0 and TransformerEngine 2.10 are already in the
# base image, so we skip the heavy transitive deps (nv-grouped-gemm, nvidia-modelopt,
# mamba-ssm, etc.) that would require CUDA extensions to build.
# NOTE: Megatron-Bridge is NOT bundled in verl main — still an external dependency.
# NOTE: Megatron-Bridge stays in the Docker image (not runtime_env) because
# runtime_env pip does not support --no-deps.
#####################
WORKDIR /workspace
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/NVIDIA-NeMo/Megatron-Bridge@${MBRIDGE_VERSION}

# Megatron-Bridge v0.5.0 hard-imports modelopt (auto_bridge.py:
# `from modelopt.torch.quantization.utils import is_quantized`) and modelopt
# imports pulp. Installed --no-deps: quantization is unused in our path, we
# only need the modules importable. scipy (also imported) comes from
# runtime_env.yaml.
RUN pip install --no-cache-dir --no-deps nvidia-modelopt pulp

# Pin Ray to match the KubeRay cluster head (rayproject/ray:2.53.0-py312).
# The vllm020.dev2 base image ships Ray 2.55.1, but Ray requires matching
# versions between head and workers — mismatched versions fail at node
# registration. Downgrade is safe: verl v0.8.0 requires unpinned ray[default]
# and vLLM 0.20.2 declares no ray pin.
RUN pip install --no-cache-dir "ray[default]==2.53.0"

# Blackwell B200 (SM100) vLLM LoRA PDL/GDC fix — vllm-project/vllm#30872.
# Triton's LLVM pipeliner cannot handle gdc_wait() inline ASM on SM100.
# vLLM 0.20.2 includes PR #32836, which adds this env-var switch to force the
# non-PDL kernel path. This replaces the old sed patch of
# vllm/lora/ops/triton_ops/utils.py that was required on vLLM 0.12.0.
ENV VLLM_LORA_DISABLE_PDL=1

# Disable the FlashInfer TRTLLM BF16 MoE kernel on Blackwell. vLLM 0.20.2's
# unquantized-MoE oracle prefers FLASHINFER_TRTLLM, whose block layout
# requires K % 128 == 0 — Qwen3 MoE intermediate sizes at rollout TP=8
# violate this (30B: 768/8=96, 235B: 1536/8=192) and engine init dies with
# "AssertionError: K must be divisible by blockK" during weight conversion
# (flashinfer convert_to_block_layout). Setting 0 removes TRTLLM from the
# backend list; vLLM falls back to FlashInfer CUTLASS / Triton fused MoE,
# the same class of kernels used by our validated vLLM 0.12 runs.
ENV VLLM_USE_FLASHINFER_MOE_FP16=0

# Environment variables required for Megatron LoRA
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV NCCL_NVLS_ENABLE=0
ENV VLLM_USE_V1=1

# Custom reward function — thin wrapper around default_compute_score that
# injects sandbox_fusion_url for code tasks. Compatible with both verl v0.7.0
# and verl main (0.8.0.dev). The **kwargs absorb any new arguments (e.g.,
# reward_router_address) that verl main may pass when Reward Loop is active.
COPY scripts/custom_reward_fn.py /workspace/custom_reward_fn.py

# Set working directory
WORKDIR /workspace

# Expose Ray ports
EXPOSE 8265 10001 6379
