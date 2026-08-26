# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
#
# vLLM + DeepEP-V2 MoE all-to-all over AWS EFA — GDAKI (GPU-initiated, kernel-posted WQE)
# variant. NGC-from-scratch. This is the NCCL_GIN_TYPE=3 (GDAKI) analog of
# ../deepep-v2-efa (the NCCL-GIN CPU-proxy, NCCL_GIN_TYPE=2 package): same base, same
# torch/NCCL/NVSHMEM/vLLM pins, same DeepEP base+PR. The deltas are the GDAKI transport
# stack, built here from public source and pinned by immutable SHA:
#   Layer 2b  rdma-core @ master post-PR#1701 (EFA completion-counter verbs; merged
#             2026-08-04, no release tag yet)
#   Layer 2c  libfabric @ main post-PR#12591  (prov/efa comp-cntr caps; merged 2026-07-28)
#             built against that rdma-core
#   Layer 5   aws-ofi-nccl @ a3d2680 --enable-gdaki (SHA pin, no local patches);
#             a3d2680 carries PR#1311/5b1f6dd (per-platform EFA hw-counter tristate) and
#             6e504db (GIN seq-space aliasing fix). Built in setup_deepep_v2_gdaki_efa.sh.
#   Layer 5   DeepEP b306af06 + PR#612 (deepseek-ai/DeepEP, filed upstream, OPEN — EFA
#             auto-QP cap; pinned to the PR's immutable head SHA). No local source patches.
#             Staged in setup_deepep_v2_gdaki_efa.sh.
#
#   DOCKER_BUILDKIT=1 docker build -t vllm-deepep-v2-gdaki-efa:latest .
#
# HONEST SCOPE: the GDAKI *transport* is proven on p5en (the vLLM serve over
# NCCL_GIN_TYPE=3 is validated by benchmarks/README.md — quote only what is there; the
# transport A/B is same-node-set env-flip only). SRD ground truth still applies (NO IBGDA,
# NO atomics, NO ordering, CQ completion is local). NEVER set NVSHMEM_IB_ENABLE_IBGDA=1.
#
# NODE PRECONDITION beyond the proxy package: GDAKI hw-counter mode needs node efa.ko >=
# 3.3.0 (efa_linux_3.3.0, 2026-07-28). On older nodes set OFI_NCCL_GDAKI_EFA_HW_COUNTER=off
# (the PR#1311 tristate — a STRING enum; "0"/"1" abort plugin init) — see kubernetes/ YAML.
#
# KNOWN-MISSING UPSTREAM FIX (documented, NOT applied): vLLM PR#47785 ("handle topk_ids
# padding in align sum kernel", merged 2026-07-10) post-dates the pinned vLLM e2f993dc4 and
# patches a COMPILED kernel — a source cherry-pick is INERT under the precompiled-wheel
# install. Kept at e2f993dc4 (first commit with the deepep_v2 backend; eager path validated at this pin).
# Swap seam: move VLLM_SHA + wheel URL to any commit >= f378f79b and re-run the gate.
#
ARG CUDA_VER=13.0.0                                   # cu13 to match the torch 2.11+cu130 ABI DeepEP links
FROM nvcr.io/nvidia/cuda:${CUDA_VER}-devel-ubuntu22.04

LABEL org.opencontainers.image.description="vLLM + DeepEP-V2 MoE all-to-all over AWS EFA (GDAKI GPU-initiated, NCCL_GIN_TYPE=3)"
LABEL org.opencontainers.image.licenses="MIT-0"
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ---- Layer 1: system + build deps (+ cmake/libnl for the rdma-core source build) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential autoconf automake libtool pkg-config git curl wget ca-certificates \
      cmake libnl-3-dev libnl-route-3-dev libudev-dev \
      libnuma-dev libhwloc-dev python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Layer 2: AWS EFA (public installer; pinned, no 'latest') ----
# Provides the baseline /opt/amazon/efa libfabric + OpenMPI; the GDAKI plugin builds
# against the NEWER libfabric in Layer 2c instead. Frozen to the proxy package's 1.48.0
# (1.49.0 exists, 2026-06-26 — bumping is a one-ARG seam + re-smoke, per the freshness rule).
ARG EFA_INSTALLER_VER=1.48.0
RUN apt-get update \
    && curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VER}.tar.gz | tar -xzf - -C /tmp \
    && cd /tmp/aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify \
    && echo "${EFA_INSTALLER_VER}" > /opt/efa-installer.version \
    && rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/*
ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:${LD_LIBRARY_PATH:-}

# ---- Layer 2b: rdma-core @ master post-PR#1701 (PUBLIC: github.com/linux-rdma/rdma-core) ----
# PR#1701 (merged 2026-08-04) adds the EFA completion-counter verbs (efadv comp-cntr /
# ibv_query_comp_cntr_caps) that libfabric's efa provider + the GDAKI hw-counter mode consume.
# Master pin a4b8d50 (2026-08-11) — no release tag carries it yet (freshness-checked 2026-08-14).
ARG RDMA_CORE_REPO=https://github.com/linux-rdma/rdma-core.git
ARG RDMA_CORE_SHA=a4b8d50e6357
RUN set -eux; \
    git clone ${RDMA_CORE_REPO} /tmp/rdma-core; \
    cd /tmp/rdma-core; \
    git checkout "${RDMA_CORE_SHA}"; \
    git rev-parse HEAD > /opt/rdma-core-gdaki.sha; \
    mkdir build && cd build; \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/rdma-core-gdaki -DNO_MAN_PAGES=1 -DCMAKE_BUILD_TYPE=Release ..; \
    make -j"$(nproc)"; make install; \
    COMP_SYMS="$(nm -D /opt/rdma-core-gdaki/lib/libefa.so.1 | grep -ci comp_cntr || true)"; \
    [ "${COMP_SYMS}" -ge 1 ] || { echo "ASSERT FAIL: no comp_cntr verbs in libefa (PR#1701 missing?)"; exit 93; }; \
    rm -rf /tmp/rdma-core

# ---- Layer 2c: libfabric @ main post-PR#12591 (PUBLIC: github.com/ofiwg/libfabric) ----
# PR#12591 (merged 2026-07-28, 3a1a8bdc) makes prov/efa consume the comp-cntr caps. Main pin
# cac9e7b0 (2026-08-13) >= that merge. Built against Layer 2b's rdma-core so efa sees the new verbs.
ARG LIBFABRIC_REPO=https://github.com/ofiwg/libfabric.git
ARG LIBFABRIC_SHA=cac9e7b0f9bb
RUN set -eux; \
    git clone ${LIBFABRIC_REPO} /tmp/libfabric; \
    cd /tmp/libfabric; \
    git checkout "${LIBFABRIC_SHA}"; \
    git rev-parse HEAD > /opt/libfabric-gdaki.sha; \
    ./autogen.sh; \
    PKG_CONFIG_PATH=/opt/rdma-core-gdaki/lib/pkgconfig \
    CPPFLAGS="-I/opt/rdma-core-gdaki/include" \
    LDFLAGS="-L/opt/rdma-core-gdaki/lib -Wl,-rpath,/opt/rdma-core-gdaki/lib" \
    ./configure --prefix=/opt/libfabric-gdaki \
      --enable-efa --with-cuda=/usr/local/cuda --enable-cuda-dlopen \
      --disable-verbs --disable-psm3 --disable-opx --disable-usnic --disable-rstream; \
    make -j"$(nproc)"; make install; \
    test -f /opt/libfabric-gdaki/lib/libfabric.so; \
    rm -rf /tmp/libfabric

# ---- Layer 3: gdrcopy userspace (PUBLIC: github.com/NVIDIA/gdrcopy) ----
# NODE PRECONDITION: the host gdrdrv kernel module must be GDRCopy >= 2.5 (this image bakes
# the 2.5.2 userspace; a mismatched < 2.5 host gdrdrv is the config aws-ofi-nccl PR#1351 was
# opened for, but the maintainer closed #1351 in favour of requiring GDRCopy 2.5+ on the node
# instead — so we require it here rather than carrying that cherry-pick). Check the node:
#   cat /sys/module/gdrdrv/version   # must be >= 2.5  (or `modinfo gdrdrv | grep ^version`)
# recipe/verify-image.sh + the K8s launcher fail loud if the node gdrdrv is < 2.5.
ARG GDRCOPY_VER=v2.5.2
RUN git clone --depth 1 --branch ${GDRCOPY_VER} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy && make prefix=/usr/local lib lib_install && ldconfig \
    && rm -rf /tmp/gdrcopy

# ---- Layer 4: torch cu13 stack (pinned; nccl-cu13 2.30.4 carries the GIN/LSA + GDAKI device symbols) ----
ARG TORCH_SPEC="torch==2.11.0"                        # DeepEP _C.so links this exact ABI
ARG TORCH_INDEX="https://download.pytorch.org/whl/cu130"
ARG NVIDIA_NCCL_CU13="2.30.4"                         # 2.28.x lacks the GIN/LSA + GDAKI symbols the plugin needs
ARG NVSHMEM_CU13="3.6.5"
RUN pip3 install --no-cache-dir --break-system-packages --index-url "${TORCH_INDEX}" "${TORCH_SPEC}" \
    && pip3 install --no-cache-dir --break-system-packages --no-deps \
         "nvidia-nccl-cu13==${NVIDIA_NCCL_CU13}" "nvidia-nvshmem-cu13==${NVSHMEM_CU13}" \
    && python3 -c "import torch; assert torch.__version__.startswith('2.11'), torch.__version__; print('torch', torch.__version__, '| cuda', torch.version.cuda)"
# resolve the pip nccl onto the linker path (dynamic — never hardcode python3.NN) so it wins over any system libnccl
RUN NCCL_ROOT=$(python3 -c "import nvidia.nccl, pathlib; print(pathlib.Path(nvidia.nccl.__path__[0]))") \
    && echo "$NCCL_ROOT/lib" > /etc/ld.so.conf.d/00-pip-nccl.conf && ldconfig \
    && ldconfig -p | grep "libnccl.so.2 " | head -1 | grep -q "$NCCL_ROOT/lib"   # assert the pinned NCCL resolves FIRST (2.28.x lacks the GIN/LSA symbols the plugin needs)

# ---- Layer 5: aws-ofi-nccl GDAKI + DeepEP-V2 source (in-tree script, COPY'd not curled) ----
# The setup script builds --enable-gdaki against Layer 2c libfabric + Layer 2b rdma-core (SHA
# pin, no local patches), and stages DeepEP b306af06 + PR#612 (filed upstream, open). The
# DeepEP _C.so is built IN-POD on first boot (recipe/build_deepep.sh) — it needs a live CUDA
# context the build sandbox lacks.
COPY setup_deepep_v2_gdaki_efa.sh /opt/setup_deepep_v2_gdaki_efa.sh
ARG AWS_OFI_NCCL_SHA=a3d268024576c159e97916151666c2ef20f91813
ARG DEEPEP_SHA=b306af06afd412c88e51e71802951606e40b7358
ARG DEEPEP_PR=612
ARG DEEPEP_PR_SHA=28d1f7fb173f728be51632ce0026fea23243e350
RUN chmod +x /opt/setup_deepep_v2_gdaki_efa.sh \
    && AWS_OFI_NCCL_SHA=${AWS_OFI_NCCL_SHA} \
       DEEPEP_SHA=${DEEPEP_SHA} DEEPEP_PR=${DEEPEP_PR} DEEPEP_PR_SHA=${DEEPEP_PR_SHA} \
       /opt/setup_deepep_v2_gdaki_efa.sh
ENV LD_LIBRARY_PATH=/opt/aws-ofi-nccl-gdaki/lib:/opt/libfabric-gdaki/lib:/opt/rdma-core-gdaki/lib:${LD_LIBRARY_PATH}

# ---- Layer 6: in-pod _C.so build script (run once on pod boot; idempotent) ----
COPY recipe/build_deepep.sh /opt/build_deepep.sh
RUN chmod +x /opt/build_deepep.sh

# ---- Layer 7: vLLM (pinned wheel = the first commit with the deepep_v2 backend, PR#41183) ----
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_SHA=e2f993dc4116ba96ea10ec40b7bd2ee45b27b2b8
ARG VLLM_WHEEL_URL="https://wheels.vllm.ai/e2f993dc4116ba96ea10ec40b7bd2ee45b27b2b8/vllm-0.22.1rc1.dev283%2Bge2f993dc4-cp38-abi3-manylinux_2_28_x86_64.whl"
RUN set -eux; \
    curl -fsSL -o /opt/vllm-precompiled.whl "${VLLM_WHEEL_URL}"; \
    python3 - /opt/vllm-precompiled.whl <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); names = set(z.namelist())
need = {"vllm/_C.abi3.so", "vllm/_C_stable_libtorch.abi3.so", "vllm/_moe_C.abi3.so"}
missing = sorted(need - names)
assert not missing, f"WHEEL-PIN ASSERT FAIL: wheel missing {missing}"
print("wheel-pin OK: contains vllm/_C.abi3.so + _moe_C.abi3.so + _C_stable_libtorch")
PY
RUN set -eux; \
    git clone ${VLLM_REPO} /opt/vllm; \
    cd /opt/vllm; \
    git checkout "${VLLM_SHA}"; \
    git rev-parse HEAD > /opt/vllm/.git-sha; \
    pip install --no-cache-dir --break-system-packages --force-reinstall --no-deps \
      "torch==2.11.0+cu130" "torchaudio==2.11.0+cu130" "torchvision==0.26.0+cu130" \
      --index-url https://download.pytorch.org/whl/cu130; \
    pip install --no-cache-dir --break-system-packages \
      "nvidia-cudnn-cu13==9.19.0.56" "nvidia-cusparselt-cu13==0.8.0"; \
    VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=/opt/vllm-precompiled.whl \
      pip install --no-cache-dir --break-system-packages -e . \
        --extra-index-url https://download.pytorch.org/whl/cu130; \
    test -f /opt/vllm/vllm/_C.abi3.so; \
    echo "Layer 7 OK: vllm @ ${VLLM_SHA} with _C.abi3.so present"

# ---- Layer 8: re-pin NCCL + NVSHMEM LAST (vLLM's resolver downgrades them; both ----
#      --no-deps + --force-reinstall load-bearing; never self-symlink the real .so). ----
RUN set -eux; \
    pip uninstall -y --break-system-packages nvidia-nccl-cu12 2>/dev/null || true; \
    pip install --no-cache-dir --break-system-packages --no-deps --force-reinstall \
      "nvidia-nccl-cu13==2.30.4" "nvidia-nvshmem-cu13==3.6.5"; \
    NCCL_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d | head -1)"; \
    test -n "${NCCL_LIB}"; \
    REAL_SO2="${NCCL_LIB}/libnccl.so.2"; test -f "${REAL_SO2}"; \
    ln -sf "libnccl.so.2" "${NCCL_LIB}/libnccl.so"; \
    ln -sf "${REAL_SO2}" /usr/local/lib/libnccl.so.2; \
    ln -sf "${REAL_SO2}" /usr/local/lib/libnccl.so; \
    echo "${NCCL_LIB}" > /etc/ld.so.conf.d/aa-nvidia-nccl.conf; \
    NVER="$(strings "${REAL_SO2}" | grep "NCCL version" | tail -1)"; \
    echo "baked-nccl-banner: ${NVER}"; \
    case "${NVER}" in *"2.30.4"*) : ;; *) echo "ASSERT FAIL nccl != 2.30.4: ${NVER}"; exit 91;; esac; \
    GIN_SYMS="$(nm -D "${REAL_SO2}" | grep -c ncclGetLsaDevicePointer)"; \
    [ "${GIN_SYMS}" -ge 1 ] || { echo "ASSERT FAIL: no GIN/LSA symbols in ${REAL_SO2}"; exit 92; }; \
    NVSHMEM_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nvshmem/lib' -type d | head -1)"; \
    test -n "${NVSHMEM_LIB}"; \
    HOST_SO="$(ls "${NVSHMEM_LIB}"/libnvshmem_host.so.* | head -1)"; test -n "${HOST_SO}"; \
    ln -sf "$(basename "${HOST_SO}")" "${NVSHMEM_LIB}/libnvshmem_host.so"; \
    echo "${NVSHMEM_LIB}" > /etc/ld.so.conf.d/zz-nvshmem.conf; \
    ldconfig; \
    pip show nvidia-nvshmem-cu13 | grep "Version: 3.6.5"; \
    python3 -c "import vllm; print('vllm', vllm.__version__)"; \
    python3 -c "from vllm.distributed.device_communicators.all2all import DeepEPV2All2AllManager; print('PR#41183 symbol OK')"; \
    echo "Layer 8 OK: nccl 2.30.4 (GIN) + nvshmem 3.6.5 re-pinned LAST"

# -----------------------------------------------------------------------------
# Runtime defaults — the GDAKI (GPU-initiated) GIN transport contract.
# Contrast with ../deepep-v2-efa (proxy): NCCL_GIN_TYPE 2→3, OFI_NCCL_GIN_GDAKI 0→1,
# plugin path → the GDAKI-enabled build. Everything else mirrors the proxy contract
# (incl. EP_REUSE_NCCL_COMM=0, the vLLM null-comm segfault fix). The
# OFI_NCCL_GDAKI_EFA_HW_COUNTER tristate is left to the kubernetes/ YAML: set "off" on
# nodes with efa.ko < 3.3.0, "auto" (or unset) on 3.3.0+.
# -----------------------------------------------------------------------------
ENV NCCL_NET_PLUGIN=/opt/aws-ofi-nccl-gdaki/lib/libnccl-net-ofi.so
ENV NCCL_GIN_ENABLE=1
ENV NCCL_GIN_TYPE=3
ENV OFI_NCCL_GIN_GDAKI=1
ENV EP_REUSE_NCCL_COMM=0
ENV OFI_NCCL_GIN_MAX_REQUESTS=512
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV FI_EFA_ENABLE_SHM_TRANSFER=0
ENV FI_EFA_FORK_SAFE=1
ENV OFI_NCCL_PROTOCOL=RDMA
ENV NCCL_CUMEM_ENABLE=1
ENV NCCL_NVLS_ENABLE=0
ENV NCCL_IGNORE_DISABLED_P2P=1
ENV DEEP_EP_BACKEND=nccl

# ---- Layer 9: the serve/build scripts (LAST — script iteration never invalidates heavy layers) ----
COPY recipe/serve.sh /opt/serve.sh
COPY recipe/run-kernel-test.sh /opt/run-kernel-test.sh
COPY recipe/benchmark_probe.py /opt/benchmark_probe.py
RUN chmod +x /opt/serve.sh /opt/run-kernel-test.sh

RUN { echo "vllm-deepep-v2-GDAKI-efa image (GPU-initiated, NCCL_GIN_TYPE=3)"; \
      echo "efa:           $(cat /opt/efa-installer.version)"; \
      echo "rdma-core:     $(cat /opt/rdma-core-gdaki.sha) (post-PR#1701 comp-cntr)"; \
      echo "libfabric:     $(cat /opt/libfabric-gdaki.sha) (post-PR#12591)"; \
      echo "aws-ofi-nccl:  $(cat /opt/aws-ofi-nccl-gdaki.sha) (--enable-gdaki, SHA pin)"; \
      echo "deepep base:   $(cat /opt/deepep.base.sha) + PR#612 (upstream, open; built in-pod)"; \
      echo "vllm:          $(cat /opt/vllm/.git-sha) (PR#41183, wheel-pinned; PR#47785 documented-missing)"; \
      echo "built:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > /opt/GUIDE-PROVENANCE.txt \
    && cat /opt/GUIDE-PROVENANCE.txt

WORKDIR /opt
# The segfault-fix env (EP_REUSE_NCCL_COMM=0) + GDAKI-Gin contract are set by recipe/serve.sh at launch.
CMD ["/bin/bash", "-lc", "cat /opt/GUIDE-PROVENANCE.txt; echo 'run: recipe/serve.sh {leader|worker} <ip>; build_deepep.sh runs once on first boot'; sleep infinity"]
