# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# ============================================================================
# NeMo-RL + Megatron-LM + DeepEP V2 (NCCL-GIN over AWS EFA) for HyperPod EKS
# ============================================================================
#
# MoE expert-parallel all-to-all for RL post-training, carried by DeepEP V2's
# NCCL backend (ElasticBuffer) over NCCL-GIN's CPU-proxy and aws-ofi-nccl on
# AWS EFA. Everything is built from PUBLIC sources on the NGC PyTorch base —
# no private registry, no fork URL. Every pin is an immutable SHA with a WHY.
#
# Two image flavors from this one file:
#   docker build -f nemo-rl.Dockerfile -t <registry>/nemo-rl-deepep-efa:<tag> .
#       -> BASELINE: upstream-only trees. Gates: imports, ElasticBuffer
#          bring-up, cross-node EFA transport probe, non-DeepEP train step.
#   docker build --build-arg APPLY_DRAFT_ROLLOUT_PATCHES=1 ...
#       -> OPT-IN: additionally bakes 3 DRAFT upstream PRs (see the patches/
#          script) that the full NeMo-RL GRPO rollout-over-DeepEP path needs.
#          The baseline image has ZERO dependence on them.

# Base: same NGC base as the sibling RL test case (slime). Bakes torch 2.11 /
# CUDA 13 with TransformerEngine, apex and flash-attn compiled against that
# exact ABI — which is what megatron.core's H100 path needs; rebuilding any of
# those from PyPI against the baked torch is where images usually go wrong.
ARG NGC_PYTORCH_BASE=nvcr.io/nvidia/pytorch:26.02-py3
FROM ${NGC_PYTORCH_BASE}
ARG NGC_PYTORCH_BASE   # re-declare: pre-FROM ARGs go out of scope after FROM

LABEL org.opencontainers.image.description="NeMo-RL + Megatron-LM + DeepEP V2 MoE all-to-all over AWS EFA (NCCL-GIN CPU-proxy)"
LABEL org.opencontainers.image.licenses="MIT-0"
LABEL org.opencontainers.image.source="https://github.com/awslabs/awsome-distributed-ai"

# ---- pins (every one justified; no floating refs) --------------------------
# NCCL v2.30.4-1: the GIN device API generation the measured substrate ran
# (ships include/nccl_device.h — asserted below). Same NCCL line the NGC base
# bakes, so the ld.so.conf override below introduces no version drift; built
# from source so the DeepEP build has one controlled root of headers + libs.
ARG NCCL_VERSION=v2.30.4-1
# EFA 1.48.0: the userspace of the measured substrate (see README pins table).
# Bumping it is a re-measure event, not a routine bump.
ARG EFA_INSTALLER_VERSION=1.48.0
# gdrcopy v2.5.2 == commit c91ad9f: commit pin, not tag (a bare tag is a
# moving ref upstream can re-point). GIN REQUIRES gdrapi.h at aws-ofi-nccl
# configure time; without it GIN init fails at run time.
ARG GDRCOPY_SHA=c91ad9f178e5fb729fc5b6dc62a77c3bb364d6c9
# aws-ofi-nccl @9c44d34 + PR#1351 head c2e773d: the GIN CPU-proxy plugin pins
# of the measured NCCL-GIN substrate (same pins as the TensorRT-LLM NcclEP
# sibling sample). Immutable SHAs — refs/pull/N/head is a moving ref.
ARG AWS_OFI_NCCL_SHA=9c44d34476f90ddbf4a12d0ac4fc412d46bd8ab4
ARG AWS_OFI_NCCL_PR=1351
ARG AWS_OFI_NCCL_PR_SHA=c2e773dfb2c75b765b3415f8ffd1b47e7c239a7b
# DeepEP: upstream deepseek-ai/DeepEP main @01dc3aa — carries EPv2 (the
# ElasticBuffer + NCCL backend, merged upstream in PR#605) and is the EXACT
# base of draft PR deepseek-ai/DeepEP#612, so the opt-in layer applies
# --check-clean. NOT the amazon-contributing fork: baseline is stock upstream.
ARG DEEPEP_SHA=01dc3aaac82068020353dce2c302e38153c0bfaa
# Megatron-LM: the exact base of draft PR NVIDIA/Megatron-LM#4632 (DeepEP V2
# ElasticBuffer support in the flex dispatcher) for --check-clean opt-in.
ARG MEGATRON_LM_SHA=19deef67f910c96c213f33b33b30277be8b94d6d
# NeMo-RL @46be4e8: the exact base of the EFA-recipe draft PR NVIDIA-NeMo/RL#2410
# (its parent commit) — declares requires-python ">=3.12" and torch==2.9.0, both
# of which the NGC base above satisfies. requirements.txt is generated from THIS
# revision (its version pins match line-for-line). Do NOT bump to #2411's base
# cc75cad: that revision bumped requires-python to ">=3.13.13" and torch to
# 2.10.0, so `pip install -e` (Layer 7) hard-fails the interpreter check on this
# py3.12 base (--no-deps does NOT suppress the Requires-Python floor). #2411 is
# metadata-only for this image (deep_ep is built from /opt/DeepEP, not NeMo-RL's
# pin) and is dropped from the opt-in layer; #2410 applies clean on this base.
# Re-pinning to a release tag is a re-measure event.
ARG NEMO_RL_SHA=46be4e8e2b335722c9af75f84e82ad807dad5bf5
# 9.0 = Hopper (H100/H200, sm_90) — the only measured arch. Override for
# Blackwell with "9.0;10.0" / matching gencode; that is a re-measure event.
ARG TORCH_CUDA_ARCH_LIST="9.0"
ARG NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ---- Layer 1: system deps; drop the distro/HPC-X verbs + MPI stacks --------
# Same removal the slime sibling does: the EFA installer below provides
# libfabric + Open MPI, and a leftover HPC-X/UCX on the loader path is a
# classic source of wrong-transport surprises.
RUN apt-get update -y && apt-get install -y --no-install-recommends \
      autoconf automake build-essential cmake curl git jq kmod libtool \
      libhwloc-dev pkg-config \
    && apt-get remove -y --allow-change-held-packages \
      ibverbs-utils libibverbs-dev libibverbs1 libmlx5-1 || true
RUN rm -rf /opt/hpcx/ompi /usr/local/mpi /usr/local/ucx && ldconfig

# ---- Layer 2: AWS EFA userspace (public installer) -------------------------
# --disable-ngc/--disable-build-ngc: the NGC base trips the installer's NGC
# auto-detect, which would reroute to the libnccl-ofi-ngc plugin path; that
# stock plugin does not carry GIN, and two plugins on the loader path is a
# which-one-won guessing game. We build the GIN plugin from source in Layer 5.
RUN apt-get update -y \
    && curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz | tar -xzf - -C /tmp \
    && cd /tmp/aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --disable-ngc --disable-build-ngc \
    && echo "${EFA_INSTALLER_VERSION}" > /opt/efa-installer.version \
    && rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/*
ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:${LD_LIBRARY_PATH:-}

# ---- Layer 3: gdrcopy userspace ---------------------------------------------
# /usr/local prefix so aws-ofi-nccl's configure finds gdrapi.h without flags.
# The matching gdrdrv kernel module must exist on the HOST (see the manifest
# header for the privileged/device-plugin trade-off).
RUN git clone https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy && git fetch origin ${GDRCOPY_SHA} && git checkout ${GDRCOPY_SHA} \
    && make prefix=/usr/local lib lib_install && ldconfig \
    && rm -rf /tmp/gdrcopy

# ---- Layer 4: NCCL (GIN-capable, single copy wins) --------------------------
# The nccl_device.h assert is the point: DeepEP V2's NCCL backend compiles
# against the GIN device API, and a device-header-less NCCL fails only at
# DeepEP build time with a confusing include error. The ld.so.conf entry is
# named 00-* so THIS libnccl.so.2 outranks the base image's baked copy at run
# time — recipe/verify-image.sh asserts which path actually resolves.
ENV NCCL_HOME=/opt/nccl/build
RUN git clone https://github.com/NVIDIA/nccl.git /opt/nccl-src \
    && cd /opt/nccl-src && git checkout ${NCCL_VERSION} \
    && make -j"$(nproc)" src.build BUILDDIR=${NCCL_HOME} CUDA_HOME=/usr/local/cuda NVCC_GENCODE="${NVCC_GENCODE}" \
    && test -f ${NCCL_HOME}/include/nccl_device.h \
       || { echo "ERROR: nccl_device.h missing — ${NCCL_VERSION} is not GIN-capable" >&2; exit 1; } \
    && echo "${NCCL_HOME}/lib" > /etc/ld.so.conf.d/00-nccl-gin.conf && ldconfig \
    && pip3 uninstall -y nvidia-nccl-cu13 nvidia-nccl-cu12 nvidia-nccl 2>/dev/null || true \
    && rm -rf /opt/nccl-src/.git
ENV LD_LIBRARY_PATH=${NCCL_HOME}/lib:${LD_LIBRARY_PATH}

# ---- Layer 5: aws-ofi-nccl GIN plugin (in-tree script, COPY'd not curled) ---
COPY setup_nemo_rl_deepep_efa.sh /opt/setup_nemo_rl_deepep_efa.sh
RUN chmod +x /opt/setup_nemo_rl_deepep_efa.sh \
    && AWS_OFI_NCCL_SHA=${AWS_OFI_NCCL_SHA} AWS_OFI_NCCL_PR=${AWS_OFI_NCCL_PR} \
       AWS_OFI_NCCL_PR_SHA=${AWS_OFI_NCCL_PR_SHA} NCCL_HOME=${NCCL_HOME} \
       /opt/setup_nemo_rl_deepep_efa.sh ofi
ENV LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}
ENV NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so

# ---- Layer 6: clone the three upstream trees at their pinned SHAs ----------
# Clones only here; the DeepEP BUILD is deferred to Layer 9 so the opt-in
# patch layer (Layer 8) can land its .cuh/.py edits BEFORE kernels compile.
RUN git clone https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP \
    && cd /opt/DeepEP && git fetch origin ${DEEPEP_SHA} && git checkout ${DEEPEP_SHA} \
    && git clone https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM \
    && cd /opt/Megatron-LM && git fetch origin ${MEGATRON_LM_SHA} && git checkout ${MEGATRON_LM_SHA} \
    && git clone https://github.com/NVIDIA-NeMo/RL.git /opt/NeMo-RL \
    && cd /opt/NeMo-RL && git fetch origin ${NEMO_RL_SHA} && git checkout ${NEMO_RL_SHA}

# ---- Layer 7: NeMo-RL + Megatron-LM python installs ------------------------
# NeMo-RL's pyproject pins torch exactly (e.g. torch==2.9.0), which would
# DOWNGRADE the NGC-baked torch and orphan the baked TE/apex/flash-attn ABI —
# so --no-deps is mandatory here, with the import-relevant dependency set
# pinned in requirements.txt instead. Megatron-LM stays a PYTHONPATH tree
# (matching the slime sibling's runtime layout) so the opt-in patch layer's
# in-place edits are exactly what executes.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -c 'import sys; assert sys.version_info >= (3, 12), f"NeMo-RL needs python>=3.12, base has {sys.version}"' \
    && pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && pip3 install --no-cache-dir --no-deps -e /opt/NeMo-RL
ENV PYTHONPATH=/opt/Megatron-LM:${PYTHONPATH:-}

# ---- Layer 7b: make the pip NVSHMEM (linked into deep_ep/_C.so) win at runtime
# deep_ep/_C.so is built against the pip nvidia-nvshmem-cu13 wheel (Layer 7, 3.7.x),
# but torch/lib/libtorch_nvshmem.so — imported before deep_ep — has a NEEDED
# libnvshmem_host.so.3 with a RUNPATH ending in /usr/local/cuda/lib64, where the
# NGC base's OLDER dpkg NVSHMEM (3.4.x, no nvshmem_selected_device_transport) lives.
# RUNPATH OUTRANKS ld.so.cache in the loader search order, so an ld.so.conf.d entry
# does NOT win: the 3.4.x lib loads first by soname and `import deep_ep` then dies
# with `undefined symbol: nvshmem_selected_device_transport, version NVSHMEM`. Only
# LD_LIBRARY_PATH outranks RUNPATH — so symlink the wheel's lib dir to a stable path
# (version-agnostic: no python3.X hardcode) and PREPEND it. Verified: recipe/
# verify-image.sh deep_ep+ElasticBuffer gate fails without this, passes with it.
RUN ln -sfn "$(python3 -c 'import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])')/lib" /opt/nvshmem-pip-lib \
    && test -f /opt/nvshmem-pip-lib/libnvshmem_host.so.3 \
       || { echo "ERROR: pip nvshmem lib dir not found — requirements.txt must install nvidia-nvshmem-cu13" >&2; exit 1; }
ENV LD_LIBRARY_PATH=/opt/nvshmem-pip-lib:${LD_LIBRARY_PATH}

# ---- Layer 8 (OPT-IN, default OFF): the 3 draft upstream PRs ----------------
# NVIDIA-NeMo/RL#2410, NVIDIA/Megatron-LM#4632, deepseek-ai/DeepEP#612
# — the full GRPO rollout-over-DeepEP path depends on them; the BASELINE image
# does not. Commits are pinned inside patches/apply_nemo_rl_patches.py at
# immutable SHAs and applied fail-loud (git apply --check first): if a hunk no
# longer applies, the BUILD fails rather than shipping an ambiguous image.
# Retire each entry when its PR merges (the script self-neutralizes).
ARG APPLY_DRAFT_ROLLOUT_PATCHES=0
COPY patches/apply_nemo_rl_patches.py /opt/patches/apply_nemo_rl_patches.py
RUN if [ "${APPLY_DRAFT_ROLLOUT_PATCHES}" = "1" ]; then \
      python3 /opt/patches/apply_nemo_rl_patches.py \
        --deepep-root /opt/DeepEP --megatron-root /opt/Megatron-LM --nemo-rl-root /opt/NeMo-RL \
        --marker /opt/.draft-rollout-patches-applied; \
    else echo "draft-PR layer skipped (APPLY_DRAFT_ROLLOUT_PATCHES=0 — upstream-only baseline)"; fi

# ---- Layer 9: build DeepEP V2 (NCCL backend) from the (possibly patched) tree
RUN EP_NCCL_ROOT_DIR=${NCCL_HOME} TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    /opt/setup_nemo_rl_deepep_efa.sh deepep

# ---- Layer 10: recipe scripts (LAST — iteration never invalidates heavy layers)
COPY recipe/verify-image.sh /opt/verify-image.sh
COPY recipe/run-rollout-probe.sh /opt/run-rollout-probe.sh
COPY recipe/probe_rollout.py /opt/probe_rollout.py
COPY recipe/train-step.sh /opt/train-step.sh
COPY recipe/train_moe_step.py /opt/train_moe_step.py
RUN chmod 755 /opt/verify-image.sh /opt/run-rollout-probe.sh /opt/train-step.sh

# ---- runtime transport contract (also repeated in the manifests) ------------
ENV FI_PROVIDER=efa \
    FI_EFA_USE_DEVICE_RDMA=1 \
    FI_EFA_FORK_SAFE=1 \
    FI_EFA_ENABLE_SHM_TRANSFER=0 \
    RDMAV_FORK_SAFE=1 \
    NCCL_GIN_TYPE=2 \
    NCCL_GIN_ENABLE=1 \
    OFI_NCCL_GIN_GDAKI=0 \
    OFI_NCCL_GIN_MAX_REQUESTS=512 \
    OFI_NCCL_PROTOCOL=RDMA \
    NCCL_NVLS_ENABLE=0 \
    NCCL_DEBUG=WARN \
    NCCL_SOCKET_IFNAME=^lo,docker,veth \
    RAY_memory_monitor_refresh_ms=0 \
    TOKENIZERS_PARALLELISM=false \
    DEEP_EP_USE_V2_SHIM=0
# Read only by the PATCHED deep_ep (DeepEP#612 adds the EFA awareness); inert
# on the baseline image. Kept here so both flavors run with one manifest.
ENV EP_EFA_MAX_QPS=2 \
    EP_EFA_RDMA_GBS=25.0

WORKDIR /opt/NeMo-RL
CMD ["/bin/bash", "-lc", "echo 'gates: /opt/verify-image.sh (in verify mode) | /opt/run-rollout-probe.sh {leader|worker} <ip> | /opt/train-step.sh {leader|worker} <ip>'; sleep infinity"]
