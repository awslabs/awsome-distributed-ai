# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
#
# vLLM + DeepEP-V2 MoE all-to-all over AWS EFA (NCCL-GIN CPU-proxy). NGC-from-scratch.
# DeepEP's default NVSHMEM/IBGDA transport is dead on EFA; this uses aws-ofi-nccl's GIN
# CPU-proxy (NCCL_GIN_TYPE=2). The GIN plugin comes from the EFA installer (>= 1.50.0
# bundles aws-ofi-nccl 1.21.1 with GA GIN support). DeepEP source staging lives in
# setup_deepep_v2_efa.sh (COPY'd, not curled) so it is in-tree + reviewable; the DeepEP
# _C.so is built IN-POD on first boot (recipe/build_deepep.sh) — it needs a live CUDA
# context the build sandbox lacks.
#
#   setup/build-push.sh builds + pushes ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} (setup/env_vars);
#   manual equivalent: DOCKER_BUILDKIT=1 docker build -t <registry>/vllm-deepep-v2-efa:<tag> .
#
ARG CUDA_VER=13.0.0                                   # cu13 to match the torch 2.11+cu130 ABI DeepEP links
FROM nvcr.io/nvidia/cuda:${CUDA_VER}-devel-ubuntu22.04

LABEL org.opencontainers.image.description="vLLM + DeepEP-V2 MoE all-to-all over AWS EFA (NCCL-GIN CPU-proxy)"
LABEL org.opencontainers.image.licenses="MIT-0"
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ---- Layer 1: system + build deps -------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential autoconf automake libtool pkg-config git curl wget ca-certificates \
      libnuma-dev libhwloc-dev python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Layer 2: AWS EFA (public installer; >= 1.50.0 bundles the GIN-capable aws-ofi-nccl plugin) ----
# --disable-ngc: this NGC base ships /opt/nvidia/nvidia_entrypoint.sh, which trips the
# installer's NGC auto-detect and silently reroutes it down the libnccl-ofi-ngc path.
# The plain install drops aws-ofi-nccl 1.21.1 at /opt/amazon/ofi-nccl — GIN support is GA
# there (host-proxy + EFA-GDA) since aws-ofi-nccl v1.21.0 (2026-08), so the plugin is no
# longer built from source here. Same migration as the canonical deepep-v2-benchmark
# (upstream #1239): the ncclGinPlugin_v14 gate fails the build loudly on a too-old
# installer (pre-1.21 GIN builds export at most the v11/v13 op-tables; the v14 table is
# 1.21.x+), and the strings check asserts the bundled plugin was compiled WITH gdrcopy
# (a gdrapi-less plugin fails nccl_ofi_gin_init at serve with ginType==NONE).
ARG EFA_INSTALLER_VER=1.50.0                          # pin — no 'latest' (the stack under test)
RUN apt-get update \
    && curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VER}.tar.gz | tar -xzf - -C /tmp \
    && cd /tmp/aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --disable-ngc \
    && ldconfig \
    && { nm -D /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so 2>/dev/null | grep -qw ncclGinPlugin_v14 \
         && echo "OK: bundled aws-ofi-nccl is GIN-capable (ncclGinPlugin_v14 present)" \
         || { echo "ERROR: installer's bundled plugin has no ncclGinPlugin_v14 export — GIN needs installer >= 1.50" >&2; exit 1; }; } \
    && { [ "$(strings /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so | grep -c 'GDRCopy support not available at compile time')" -eq 0 ] \
         || { echo "ERROR: bundled plugin was built WITHOUT gdrcopy — GIN would fail at serve with ginType==NONE" >&2; exit 1; }; } \
    && echo "${EFA_INSTALLER_VER}" > /opt/efa-installer.version \
    && rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/*
ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:${LD_LIBRARY_PATH:-}

# ---- Layer 3: torch cu13 stack (pinned; nccl-cu13 2.30.4 carries the GIN/LSA device symbols) ----
ARG TORCH_SPEC="torch==2.11.0"                        # DeepEP _C.so links this exact ABI
ARG TORCH_INDEX="https://download.pytorch.org/whl/cu130"
ARG NVIDIA_NCCL_CU13="2.30.4"                         # 2.28.x lacks the GIN/LSA symbols the plugin needs
ARG NVSHMEM_CU13="3.6.5"
RUN pip3 install --no-cache-dir --index-url "${TORCH_INDEX}" "${TORCH_SPEC}" \
    && pip3 install --no-cache-dir "nvidia-nccl-cu13==${NVIDIA_NCCL_CU13}" "nvidia-nvshmem-cu13==${NVSHMEM_CU13}" ninja
# ninja is baked here EXPLICITLY (build_deepep.sh device-links the DeepEP extension, which
# requires it): first pod boot must not depend on PyPI egress — private-subnet/air-gapped
# clusters would otherwise fail inside a compile step instead of a clear network error.
# resolve the pip nccl onto the linker path (dynamic — never hardcode python3.NN) so it wins over any system libnccl
RUN NCCL_ROOT=$(python3 -c "import nvidia.nccl, pathlib; print(pathlib.Path(nvidia.nccl.__path__[0]))") \
    && echo "$NCCL_ROOT/lib" > /etc/ld.so.conf.d/00-pip-nccl.conf && ldconfig \
    && ldconfig -p | grep "libnccl.so.2 " | head -1 | grep -q "$NCCL_ROOT/lib"   # assert the pinned NCCL resolves FIRST (2.28.x lacks the GIN/LSA symbols the plugin needs)

# ---- Layer 3b: gdrcopy userspace (PUBLIC: github.com/NVIDIA/gdrcopy) ----
# The GIN plugin dlopens libgdrapi at runtime (it is not a NEEDED dependency of the
# bundled .so), so the userspace lib must be baked here or nccl_ofi_gin_init fails at
# serve and DeepEP's ElasticBuffer asserts ginType==NCCL_GIN_TYPE_NONE. The HOST must
# run gdrcopy/gdrdrv >= 2.5: aws-ofi-nccl declined the gdrdrv-2.4 forced-PCIe override
# (aws/aws-ofi-nccl#1351, closed 2026-08-21 — gdrcopy 2.4.x has silent-data-corruption
# issues), so 2.5+ hosts are the only supported pairing for GIN.
# gdrcopy v2.5.2 == commit c91ad9f: pin the commit, not the tag (a bare tag is a moving ref
# upstream can re-point — every other dep here is SHA-pinned).
ARG GDRCOPY_SHA=c91ad9f178e5fb729fc5b6dc62a77c3bb364d6c9
RUN git clone https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy && git fetch origin ${GDRCOPY_SHA} && git checkout ${GDRCOPY_SHA} \
    && make prefix=/usr/local lib lib_install && ldconfig \
    && rm -rf /tmp/gdrcopy

# ---- Layer 4: DeepEP-V2 source staging (in-tree script, COPY'd not curled) ----
# The aws-ofi-nccl plugin comes from the EFA installer (Layer 2); this layer only stages
# the pinned DeepEP-V2 source. The one DeepEP BUILD happens in-pod on first boot
# (recipe/build_deepep.sh — needs a live CUDA context the build sandbox lacks).
COPY setup_deepep_v2_efa.sh /opt/setup_deepep_v2_efa.sh
ARG DEEPEP_SHA=b306af06afd412c88e51e71802951606e40b7358
ARG DEEPEP_PR=612
ARG DEEPEP_PR_SHA=28d1f7fb173f728be51632ce0026fea23243e350
RUN chmod +x /opt/setup_deepep_v2_efa.sh \
    && DEEPEP_SHA=${DEEPEP_SHA} DEEPEP_PR=${DEEPEP_PR} DEEPEP_PR_SHA=${DEEPEP_PR_SHA} \
       /opt/setup_deepep_v2_efa.sh
ENV LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:${LD_LIBRARY_PATH}

# ---- Layer 5: vLLM (pinned wheel = vLLM #52632's merge commit — the empty-ExpertTokensMetadata guard that unblocks default compilation) ----
# The wheel FILENAME is derived from the per-commit index at build time, so bumping
# VLLM_SHA ALONE is sufficient (hardcoding the filename would embed the SHA twice and a
# lone bump 404s).
# NOTE: wheels.vllm.ai nightly wheels are garbage-collected upstream, so this pin has a
# shelf life independent of this repo; the fail-loud check below names that cause.
# Pin is now on vLLM main at the merge commit of #52632 (merged 2026-08-20), the
# empty-ExpertTokensMetadata guard that unblocks default (CUDA-graph) compilation; this
# commit also carries #47785 (shared-experts -1 sentinel fix, merged earlier). This is
# the "bump the pin past #52632" edit the README describes. Was e2f993dc (PR#41183's
# first deepep_v2 backend commit); wheel for the new SHA verified indexed 2026-08-25.
ARG VLLM_SHA=14617c2b6c1257ac0d6c7b5e05b195ca30013827
RUN WHEEL_HREF=$(curl -fsSL "https://wheels.vllm.ai/${VLLM_SHA}/vllm/" | grep -oE 'href="[^"]*manylinux[^"]*x86_64\.whl"' | head -1 | cut -d'"' -f2) \
    && { [ -n "$WHEEL_HREF" ] || { echo "FATAL: no x86_64 vllm wheel indexed for ${VLLM_SHA} (bad SHA, or nightly wheel garbage-collected upstream)"; exit 1; }; } \
    && pip3 install --no-cache-dir "https://wheels.vllm.ai/${VLLM_SHA}/$(basename "$WHEEL_HREF")" \
    && python3 -c "import vllm; print('vllm', vllm.__version__)"

# ---- Layer 5b: re-pin the ABI-critical cu13 libs AFTER the last pip layer (order matters) ----
# The vLLM wheel's dependency resolution drags nvidia-nccl-cu13 back down to torch's 2.28.9
# (and nvshmem to 3.4.5) — silently, into the SAME directory, so the Layer-3 linker-precedence
# assert stays green while the GIN/LSA device symbols vanish and DeepEP's nccl.cu no longer
# compiles in-pod. Any pip layer added later must keep this re-pin LAST.
RUN pip3 install --no-cache-dir --no-deps --force-reinstall \
      "nvidia-nccl-cu13==${NVIDIA_NCCL_CU13}" "nvidia-nvshmem-cu13==${NVSHMEM_CU13}" \
    && NCCL_ROOT=$(python3 -c "import nvidia.nccl, pathlib; print(pathlib.Path(nvidia.nccl.__path__[0]))") \
    && grep -rq ncclGetLsaDevicePointer "$NCCL_ROOT/include/" \
    && [ "$(nm -D "$NCCL_ROOT/lib/libnccl.so.2" | grep -c ncclGetLsaDevicePointer)" -ge 1 ]  # GIN/LSA symbols present (2.28.x lacks them)

# ---- Layer 6: the serve/build scripts (LAST — script iteration never invalidates heavy layers) ----
COPY recipe/build_deepep.sh /opt/build_deepep.sh
COPY recipe/serve.sh /opt/serve.sh
COPY recipe/run-kernel-test.sh /opt/run-kernel-test.sh
COPY recipe/benchmark_probe.py /opt/benchmark_probe.py
COPY recipe/benchmark.sh /opt/benchmark.sh
RUN chmod +x /opt/build_deepep.sh /opt/serve.sh /opt/run-kernel-test.sh /opt/benchmark.sh

# The segfault-fix env (EP_REUSE_NCCL_COMM=0) + proxy-Gin contract are set by recipe/serve.sh at launch.
CMD ["/bin/bash", "-lc", "echo 'run: /opt/serve.sh {leader|worker} <ip>; /opt/build_deepep.sh runs once on first boot'; sleep infinity"]
