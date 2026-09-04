# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
#
# NVIDIA Dynamo (frontend + dynamo.vllm) serving DeepEP-V2 MoE all-to-all over AWS EFA
# (NCCL-GIN CPU-proxy). NGC-from-scratch. This is the vLLM-DeepEP-V2 substrate (identical to
# ../../vllm/deepep-v2-efa through Layer 5b) plus a Dynamo serving front (Layer 5c). DeepEP's
# default NVSHMEM/IBGDA transport is dead on EFA; this uses aws-ofi-nccl's GIN CPU-proxy
# (NCCL_GIN_TYPE=2). Build logic lives in setup_deepep_v2_efa.sh (COPY'd, not curled) so it is
# in-tree + reviewable. The DeepEP _C.so is built IN-POD on first boot (recipe/build_deepep.sh)
# — it needs a live CUDA context the build sandbox lacks.
#
#   setup/build-push.sh builds + pushes ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} (setup/env_vars);
#   manual equivalent: DOCKER_BUILDKIT=1 docker build -t <registry>/dynamo-deepep-v2-efa:<tag> .
#
ARG CUDA_VER=13.0.0                                   # cu13 to match the torch 2.11+cu130 ABI DeepEP links
FROM nvcr.io/nvidia/cuda:${CUDA_VER}-devel-ubuntu22.04

LABEL org.opencontainers.image.description="NVIDIA Dynamo serving DeepEP-V2 MoE all-to-all over AWS EFA (NCCL-GIN CPU-proxy)"
LABEL org.opencontainers.image.licenses="MIT-0"
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ---- Layer 1: system + build deps -------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential autoconf automake libtool pkg-config git curl wget ca-certificates \
      libnuma-dev libhwloc-dev python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Layer 2: AWS EFA (public installer; pinned, verified fresher than 1.48 per freshness rule) ----
# --disable-ngc: this NGC base ships /opt/nvidia/nvidia_entrypoint.sh, which trips the
# installer's NGC auto-detect and silently reroutes it down the libnccl-ofi-ngc path.
# We build aws-ofi-nccl from source ourselves (Layer 4), so force the normal install —
# same explicit choice as the sibling sglang/dsr1-deepep-efa and vllm/dsv3-uccl-nixl samples.
ARG EFA_INSTALLER_VER=1.49.0                          # pin — no 'latest' (the stack under test)
RUN apt-get update \
    && curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VER}.tar.gz | tar -xzf - -C /tmp \
    && cd /tmp/aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --disable-ngc \
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
# aws-ofi-nccl's GIN path REQUIRES gdrapi.h at configure time — without it the plugin
# compiles with "GDRCopy support not available", nccl_ofi_gin_init fails at serve, and
# DeepEP's ElasticBuffer asserts ginType==NCCL_GIN_TYPE_NONE. Userspace 2.5.2 on a
# gdrdrv-2.4 host is exactly the pairing OFI_NCCL_GDRCOPY_FORCED_PCIE_COPY=1 (#1351) fixes.
# gdrcopy v2.5.2 == commit c91ad9f: pin the commit, not the tag (a bare tag is a moving ref
# upstream can re-point — every other dep here is SHA-pinned).
ARG GDRCOPY_SHA=c91ad9f178e5fb729fc5b6dc62a77c3bb364d6c9
RUN git clone https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy && git fetch origin ${GDRCOPY_SHA} && git checkout ${GDRCOPY_SHA} \
    && make prefix=/usr/local lib lib_install && ldconfig \
    && rm -rf /tmp/gdrcopy

# ---- Layer 4: aws-ofi-nccl GIN + DeepEP-V2 source (in-tree script, COPY'd not curled) ----
COPY setup_deepep_v2_efa.sh /opt/setup_deepep_v2_efa.sh
ARG AWS_OFI_NCCL_SHA=9c44d34476f90ddbf4a12d0ac4fc412d46bd8ab4
ARG AWS_OFI_NCCL_PR=1351
ARG AWS_OFI_NCCL_PR_SHA=c2e773dfb2c75b765b3415f8ffd1b47e7c239a7b  # IMMUTABLE PR#1351 head (FORCED_PCIE param; a bare refs/pull/N/head is a moving ref)
ARG DEEPEP_SHA=b306af06afd412c88e51e71802951606e40b7358
ARG DEEPEP_PR=612
ARG DEEPEP_PR_SHA=28d1f7fb173f728be51632ce0026fea23243e350
RUN chmod +x /opt/setup_deepep_v2_efa.sh \
    && AWS_OFI_NCCL_SHA=${AWS_OFI_NCCL_SHA} AWS_OFI_NCCL_PR=${AWS_OFI_NCCL_PR} AWS_OFI_NCCL_PR_SHA=${AWS_OFI_NCCL_PR_SHA} \
       DEEPEP_SHA=${DEEPEP_SHA} DEEPEP_PR=${DEEPEP_PR} DEEPEP_PR_SHA=${DEEPEP_PR_SHA} \
       /opt/setup_deepep_v2_efa.sh
ENV LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}

# ---- Layer 5: vLLM (pinned wheel = PR#41183's merge commit — the first, and only measured-working, deepep_v2 backend) ----
# The wheel FILENAME is derived from the per-commit index at build time, so bumping
# VLLM_SHA ALONE is sufficient (hardcoding the filename would embed the SHA twice and a
# lone bump 404s).
# NOTE: wheels.vllm.ai nightly wheels are garbage-collected upstream, so this pin has a
# shelf life independent of this repo; the fail-loud check below names that cause.
# Pin = e2f993dc4 = the merge commit of vLLM PR#41183 (the commit that ADDED the deepep_v2
# backend, DeepEPV2All2AllManager) = 0.22.1rc1.dev283. This is the EXACT substrate the
# benchmarks/ tables were measured on, so a rebuild reproduces them. It is served EAGER
# (--enforce-eager, the shipped default): non-eager at this pin crashes deterministically
# in profile_run and needs the #52632 empty-ExpertTokensMetadata guard, which is NOT in
# this pin — see README "eager vs non-eager".
# Why not a newer pin: vLLM 0.26 (the line that carries #52632) regresses the deepep_v2
# combine on this exact DeepEP _C.so / EFA substrate — DP16/EP16 faults
# `CUDA_ERROR_LAUNCH_FAILED (719)` in profile_run/determine_available_memory during KV-cache
# sizing (measured 2026-09-04, eager AND non-eager; the standalone kernel-test still passes,
# so the fault is 0.26's DeepEP-V2 driver, not the kernels). e2f993dc4 is the newest vLLM
# proven to serve deepep_v2 DP16/EP16 on EFA, so the pin stays here until a newer line is
# re-measured green on this substrate.
ARG VLLM_SHA=e2f993dc4116ba96ea10ec40b7bd2ee45b27b2b8
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

# ---- Layer 5c: NVIDIA Dynamo serving front (--no-deps: this is the ONLY delta vs the vLLM twin) ----
# This image is the vLLM-DeepEP-V2 substrate (Layers 1-5b, byte-identical to ../../vllm/deepep-v2-efa)
# with Dynamo's serving front added on top. dynamo.vllm wraps the SAME vLLM engine (it forwards every
# unknown CLI flag straight into vLLM's AsyncEngineArgs), so nothing about the DeepEP-V2/EFA transport
# changes — only an OpenAI-compatible frontend + a DP/EP-aware worker wrapper are added.
#   --no-deps is LOAD-BEARING, not caution: `ai-dynamo==1.3.1` declares `vllm[...]==0.23.0` as a
#   dependency (the [vllm] extra). Without --no-deps, pip would pull the 0.23.0 RELEASE wheel and
#   overwrite the pinned VLLM_SHA build from Layer 5 (which also drags nvidia-nccl-cu13 back to torch's
#   2.28.x, undoing Layer 5b). With --no-deps, ai-dynamo (pure-python, py3-none-any) + ai-dynamo-runtime
#   (compiled cp310-abi3, and it carries NO vllm pin at all — it is the vLLM-independent Rust core)
#   install their own files ONLY; the sole real deps (pydantic, uvloop) are already present from the
#   vLLM wheel. Both packages are PEP-420 namespace packages under dynamo/, so the compiled core and the
#   pure-python components union cleanly with no dynamo/__init__.py shadowing.
#   1.3.1 (not 1.4.x): the pinned vLLM substrate is 0.22 (e2f993dc4, Layer 5). dynamo.vllm is a thin
#   arg-forwarding wrapper (it delegates every unknown flag to vLLM's own AsyncEngineArgs.add_cli_args
#   and constructs AsyncLLM directly), so the version that matters is the vLLM API surface its Python was
#   authored against. 1.3.1 targets 0.23.0 — ONE minor from the 0.22 substrate, the closest released
#   dynamo.vllm surface (1.2.1→0.20.1 is older AND further; 1.4.x→0.26.0 targets the line whose deepep_v2
#   combine regresses here). Verified offline (vllm git tree @ e2f993dc4) that every vLLM module path
#   dynamo.vllm.main imports exists at this pin — including the vllm/utils.py→vllm/utils/ PACKAGE split
#   (vllm.utils.system_utils, vllm.utils.hashing) that 1.3.1's Python assumes; the 0.22 substrate already
#   carries the post-split layout.
#   The Layer-5c assert imports `dynamo.vllm.MAIN` — the real serve entrypoint (`python3 -m dynamo.vllm`),
#   NOT bare `dynamo.vllm` (whose __init__ only sets __version__ and imports NO vLLM, so it would gate
#   nothing). main.py pulls the full serve-path vLLM API surface (vllm.config, vllm.v1.engine.async_llm,
#   vllm.usage.usage_lib, vllm.distributed.kv_events, vllm.v1.metrics.prometheus, and transitively the
#   ~22 vllm.* paths of its dynamo.vllm.* closure) at MODULE scope, while main() stays behind
#   `if __name__ == "__main__"` — so the import resolves the 0.22↔1.3.1 API contract WITHOUT running the
#   engine, parsing argv, or needing a GPU (vLLM imports clean on a CPU-only buildkit box). This is the
#   build-time gate on 0.22↔1.3.1 serve-path compatibility: the build FAILS LOUD on any drifted symbol,
#   catching an incompatibility in ~5 min at build instead of ~40 min into a live pod. (main.py's only
#   non-vLLM third-party top-level imports — huggingface_hub, prometheus_client, uvloop — are already
#   provided by the vLLM wheel, so --no-deps does not false-fail this gate.)
ARG AI_DYNAMO_VER=1.3.1
RUN pip3 install --no-cache-dir --no-deps \
      "ai-dynamo-runtime==${AI_DYNAMO_VER}" "ai-dynamo==${AI_DYNAMO_VER}" \
    && python3 -c "import dynamo.vllm.main, dynamo.frontend; print('dynamo.vllm.main + dynamo.frontend import OK — serve-path vLLM API resolved against the pinned wheel')" \
    && NCCL_ROOT=$(python3 -c "import nvidia.nccl, pathlib; print(pathlib.Path(nvidia.nccl.__path__[0]))") \
    && [ "$(nm -D "$NCCL_ROOT/lib/libnccl.so.2" | grep -c ncclGetLsaDevicePointer)" -ge 1 ]  # re-assert Layer 5b survived the dynamo install (must, given --no-deps)

# ---- Layer 6: the serve/build scripts (LAST — script iteration never invalidates heavy layers) ----
COPY recipe/build_deepep.sh /opt/build_deepep.sh
COPY recipe/serve.sh /opt/serve.sh
COPY recipe/run-kernel-test.sh /opt/run-kernel-test.sh
COPY recipe/benchmark_probe.py /opt/benchmark_probe.py
COPY recipe/benchmark.sh /opt/benchmark.sh
RUN chmod +x /opt/build_deepep.sh /opt/serve.sh /opt/run-kernel-test.sh /opt/benchmark.sh

# The segfault-fix env (EP_REUSE_NCCL_COMM=0) + proxy-Gin contract are set by recipe/serve.sh at launch.
CMD ["/bin/bash", "-lc", "echo 'run: /opt/serve.sh {leader|worker} <ip>; /opt/build_deepep.sh runs once on first boot'; sleep infinity"]
