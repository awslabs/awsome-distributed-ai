# syntax=docker/dockerfile:1.7
ARG NEMO_IMAGE=nvcr.io/nvidia/nemo@sha256:d8257dd0c4da714843aa1e371fe4dc8ee5a7972c247abbfd28d680bb197b6195
ARG DEEPEP_V2_QUALIFIED_PARENT=159553542841.dkr.ecr.ap-south-1.amazonaws.com/adai/kimi-k2-megatron-ep-2608-deepep-v2-gin-gda@sha256:112738c87d35af42802f3eed80bf6a867ca7e3879345f7830f82870cd30ae5c1
FROM ${NEMO_IMAGE} AS common-build

ARG BRIDGE_COMMIT=c93251151adeeadbae3ff2a2bf5ee7a1c34cff01
ARG MCORE_COMMIT=16ad357ee7973af32916fc1ca39d71065e5f03d4
ARG NCCL_TAG=v2.31.2-1
ARG NCCL_COMMIT=7b83616df3ae082a1f32bb74c27458bfe8153a13
ARG EFA_INSTALLER_VERSION=1.50.0
ARG EFA_INSTALLER_SHA256=fa6dff8593d866866c13cb4640d9059835cd4efa427971f100ab40c97bef2841
ARG GDRCOPY_TAG=v2.5.2
ENV BRIDGE_COMMIT=${BRIDGE_COMMIT} \
    MCORE_COMMIT=${MCORE_COMMIT} \
    NCCL_TAG=${NCCL_TAG} \
    NCCL_COMMIT=${NCCL_COMMIT} \
    EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION} \
    EFA_INSTALLER_SHA256=${EFA_INSTALLER_SHA256} \
    GDRCOPY_TAG=${GDRCOPY_TAG} \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:/opt/gdrcopy/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/nccl/build/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/amazon/efa/lib:/opt/gdrcopy/lib:/usr/local/cuda/lib64 \
    LIBRARY_PATH=/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/gdrcopy/lib \
    CPATH=/opt/nccl/build/include:/opt/amazon/efa/include:/opt/gdrcopy/include

RUN mkdir -p /opt/benchmark/patches /opt/benchmark/case
COPY 3.test_cases/megatron/megatron-bridge/docker/patches/ /opt/benchmark/patches/
COPY 3.test_cases/megatron/megatron-bridge/docker/install-common.sh /opt/benchmark/install-common.sh
COPY 3.test_cases/megatron/megatron-bridge/docker/collect-build-manifest.sh /opt/benchmark/collect-build-manifest.sh
COPY 3.test_cases/megatron/megatron-bridge/docker/verify-image.sh /opt/benchmark/verify-image.sh
COPY 3.test_cases/megatron/megatron-bridge/versions.lock.yaml /opt/benchmark/versions.lock.yaml
COPY 3.test_cases/megatron/megatron-bridge/bench/ /opt/benchmark/case/bench/
COPY 3.test_cases/megatron/megatron-bridge/kimi-k2/ /opt/benchmark/case/kimi-k2/
RUN --mount=type=cache,target=/root/.cache/ccache /opt/benchmark/install-common.sh
COPY 3.test_cases/megatron/megatron-bridge/docker/rebuild-pytorch-nccl.sh /opt/benchmark/rebuild-pytorch-nccl.sh
RUN --mount=type=cache,target=/root/.cache/ccache /opt/benchmark/rebuild-pytorch-nccl.sh
ENV LD_PRELOAD=/opt/nccl/build/lib/libnccl.so.2
RUN /opt/benchmark/collect-build-manifest.sh common /opt/benchmark/common-build-manifest.json
RUN python3 - <<'PY'
import json
from pathlib import Path

manifest_path = Path("/opt/benchmark/common-build-manifest.json")
manifest = json.loads(manifest_path.read_text())
manifest["torch_source_commit"] = Path("/opt/benchmark/pytorch-upstream-commit.txt").read_text().strip()
manifest["torch_onnx_commit"] = Path("/opt/benchmark/pytorch-onnx-commit.txt").read_text().strip()
manifest["torch_source_tree_sha256"] = Path("/opt/benchmark/pytorch-source-tree.sha256").read_text().split()[0]
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

FROM common-build AS nccl-alltoall-final
RUN printf '%s\n' '{"ep_arm":"nccl-alltoall","dispatcher":"alltoall","backend":null}' > /opt/benchmark/backend.json
RUN EP_ARM=nccl-alltoall /opt/benchmark/verify-image.sh
WORKDIR /opt/benchmark/case

FROM common-build AS uccl-final
ARG UCCL_COMMIT=a8c94e485da302b7cc963d2f74edb5b4e9115748
ENV UCCL_COMMIT=${UCCL_COMMIT} PER_EXPERT_BATCHING=1
COPY 3.test_cases/megatron/megatron-bridge/docker/install-uccl.sh /opt/benchmark/install-uccl.sh
RUN --mount=type=cache,target=/root/.cache/ccache /opt/benchmark/install-uccl.sh
RUN printf '%s\n' '{"ep_arm":"uccl","dispatcher":"flex","backend":"deepep"}' > /opt/benchmark/backend.json
RUN EP_ARM=uccl /opt/benchmark/verify-image.sh
WORKDIR /opt/benchmark/case

FROM common-build AS deepep-v1-nvshmem-final
ARG DEEPEP_V1_COMMIT=567632dd59810d77b3cc05553df953cc0f779799
ARG NVSHMEM_VERSION=3.7.0
ENV DEEPEP_V1_COMMIT=${DEEPEP_V1_COMMIT} \
    NVSHMEM_VERSION=${NVSHMEM_VERSION} \
    NVSHMEM_REMOTE_TRANSPORT=libfabric \
    NVSHMEM_LIBFABRIC_PROVIDER=efa \
    LD_LIBRARY_PATH=/opt/amazon/nvshmem/lib:${LD_LIBRARY_PATH} \
    LD_PRELOAD=/opt/amazon/nvshmem/lib/libnvshmem_host.so.3:${LD_PRELOAD}
COPY micro-benchmarks/expert-parallelism/deepep-benchmark/setup_deepep_efa.sh /opt/benchmark/setup_deepep_efa.sh
COPY 3.test_cases/megatron/megatron-bridge/docker/install-deepep-v1.sh /opt/benchmark/install-deepep-v1.sh
RUN --mount=type=cache,target=/root/.cache/ccache /opt/benchmark/install-deepep-v1.sh
RUN printf '%s\n' '{"ep_arm":"deepep-v1-nvshmem","dispatcher":"flex","backend":"deepep"}' > /opt/benchmark/backend.json
RUN EP_ARM=deepep-v1-nvshmem /opt/benchmark/verify-image.sh
WORKDIR /opt/benchmark/case

FROM common-build AS deepep-v2-gin-gda-final
ARG DEEPEP_V2_BASE=02efc268a37802fc00812ede8f5ad7f535ceea0e
ARG DEEPEP_PR3_HEAD=dd0f87261a80cf0ce8aa66e4ab2041843851d810
ARG DEEPEP_PR3_PATCH_SHA256=1a7b3604ff5e01d97cdc043ae2c2480ac64ae229dfb34ea457633ac8f5362bc4
ARG DEEPEP_PR5_HEAD=2542d9641f2ec280213e875feb04be7862dda57c
ARG DEEPEP_PR5_PATCH_SHA256=36d6fcac58dcc197f8bc0419822b9381296cecba40bd640440c7633796c25f7e
ARG DEEPEP_NOTIFY_WARPS_PATCH_SHA256=198fe6148b14b3c4533542f43e043835c47a861af37021628f4942f5b114a039
ARG DEEPEP_SYNTHETIC_TREE=ab43791defa8961b88fdee1a63e8877e317da9af
ARG DEEPEP_SYNTHETIC_COMMIT=0ff264eb08d196621f49c7d011270c62f8716996
ARG NVSHMEM_VERSION=3.7.0
ARG NVSHMEM_SHA256=f954ed91fa270f4b905391e03d808f60c73b894d88ef0aae865916d685353b8c
ENV DEEPEP_V2_BASE=${DEEPEP_V2_BASE} \
    DEEPEP_PR3_HEAD=${DEEPEP_PR3_HEAD} \
    DEEPEP_PR3_PATCH_SHA256=${DEEPEP_PR3_PATCH_SHA256} \
    DEEPEP_PR5_HEAD=${DEEPEP_PR5_HEAD} \
    DEEPEP_PR5_PATCH_SHA256=${DEEPEP_PR5_PATCH_SHA256} \
    DEEPEP_NOTIFY_WARPS_PATCH_SHA256=${DEEPEP_NOTIFY_WARPS_PATCH_SHA256} \
    DEEPEP_SYNTHETIC_TREE=${DEEPEP_SYNTHETIC_TREE} \
    DEEPEP_SYNTHETIC_COMMIT=${DEEPEP_SYNTHETIC_COMMIT} \
    EP_NCCL_ROOT_DIR=/opt/nccl/build \
    EP_NVSHMEM_ROOT_DIR=/opt/amazon/nvshmem \
    NVSHMEM_DIR=/opt/amazon/nvshmem \
    NVSHMEM_VERSION=${NVSHMEM_VERSION} \
    NVSHMEM_SHA256=${NVSHMEM_SHA256} \
    LD_LIBRARY_PATH=/opt/amazon/nvshmem/lib:${LD_LIBRARY_PATH}
COPY 3.test_cases/megatron/megatron-bridge/docker/install-deepep-v2.sh /opt/benchmark/install-deepep-v2.sh
RUN --mount=type=cache,target=/root/.cache/ccache /opt/benchmark/install-deepep-v2.sh
RUN printf '%s\n' '{"ep_arm":"deepep-v2-gin-gda","dispatcher":"flex","backend":"deepep_v2","synthetic_tree":"ab43791defa8961b88fdee1a63e8877e317da9af","synthetic_commit":"0ff264eb08d196621f49c7d011270c62f8716996","notify_warps_patch_sha256":"198fe6148b14b3c4533542f43e043835c47a861af37021628f4942f5b114a039"}' > /opt/benchmark/backend.json
RUN EP_ARM=deepep-v2-gin-gda /opt/benchmark/verify-image.sh
WORKDIR /opt/benchmark/case

# Fast, reproducible qualification derivative of the prior accepted V2 image.
# The parent digest already contains base + PR #3 + PR #5 and the byte-identical
# common stack. The full source target above independently constructs the same
# resulting synthetic tree for clean rebuilds.
FROM ${DEEPEP_V2_QUALIFIED_PARENT} AS deepep-v2-gin-gda-notify-final
ENV DEEPEP_NOTIFY_WARPS_PATCH_SHA256=198fe6148b14b3c4533542f43e043835c47a861af37021628f4942f5b114a039 \
    DEEPEP_SYNTHETIC_TREE=ab43791defa8961b88fdee1a63e8877e317da9af \
    DEEPEP_SYNTHETIC_COMMIT=0ff264eb08d196621f49c7d011270c62f8716996
COPY 3.test_cases/megatron/megatron-bridge/docker/patches/deepep-v2-notify-warps.patch /opt/benchmark/patches/deepep-v2-notify-warps.patch
COPY 3.test_cases/megatron/megatron-bridge/docker/apply-deepep-v2-notify-hotfix.sh /opt/benchmark/apply-deepep-v2-notify-hotfix.sh
COPY 3.test_cases/megatron/megatron-bridge/versions.lock.yaml /opt/benchmark/versions.lock.yaml
RUN --mount=type=cache,target=/root/.cache/ccache /opt/benchmark/apply-deepep-v2-notify-hotfix.sh
RUN printf '%s\n' '{"ep_arm":"deepep-v2-gin-gda","dispatcher":"flex","backend":"deepep_v2","parent_digest":"sha256:112738c87d35af42802f3eed80bf6a867ca7e3879345f7830f82870cd30ae5c1","synthetic_tree":"ab43791defa8961b88fdee1a63e8877e317da9af","synthetic_commit":"0ff264eb08d196621f49c7d011270c62f8716996","notify_warps_patch_sha256":"198fe6148b14b3c4533542f43e043835c47a861af37021628f4942f5b114a039"}' > /opt/benchmark/backend.json
RUN EP_ARM=deepep-v2-gin-gda /opt/benchmark/verify-image.sh
WORKDIR /opt/benchmark/case

# Acceptance-only PR #5 control. PR #3 remains present on Blackwell.
FROM common-build AS deepep-v2-pr5-control
ARG DEEPEP_V2_BASE=02efc268a37802fc00812ede8f5ad7f535ceea0e
ARG DEEPEP_PR3_PATCH_SHA256=1a7b3604ff5e01d97cdc043ae2c2480ac64ae229dfb34ea457633ac8f5362bc4
ARG DEEPEP_CONTROL_TREE=6624dfd5ca88aa53fc680cd00c659191cc38c481
ARG DEEPEP_CONTROL_COMMIT=9533c8e34ae2608ca256dff2ffe665ade9b6430c
ARG NVSHMEM_VERSION=3.7.0
ARG NVSHMEM_SHA256=f954ed91fa270f4b905391e03d808f60c73b894d88ef0aae865916d685353b8c
ENV DEEPEP_V2_BASE=${DEEPEP_V2_BASE} \
    DEEPEP_PR3_PATCH_SHA256=${DEEPEP_PR3_PATCH_SHA256} \
    DEEPEP_CONTROL_TREE=${DEEPEP_CONTROL_TREE} \
    DEEPEP_CONTROL_COMMIT=${DEEPEP_CONTROL_COMMIT} \
    EP_NCCL_ROOT_DIR=/opt/nccl/build \
    EP_NVSHMEM_ROOT_DIR=/opt/amazon/nvshmem \
    NVSHMEM_DIR=/opt/amazon/nvshmem \
    NVSHMEM_VERSION=${NVSHMEM_VERSION} \
    NVSHMEM_SHA256=${NVSHMEM_SHA256} \
    LD_LIBRARY_PATH=/opt/amazon/nvshmem/lib:${LD_LIBRARY_PATH}
COPY 3.test_cases/megatron/megatron-bridge/docker/install-deepep-v2-control.sh /opt/benchmark/install-deepep-v2-control.sh
RUN --mount=type=cache,target=/root/.cache/ccache /opt/benchmark/install-deepep-v2-control.sh
RUN printf '%s\n' '{"ep_arm":"deepep-v2-pr5-control","dispatcher":"native-gate","backend":"deepep_v2_base_pr3"}' > /opt/benchmark/backend.json
WORKDIR /opt/benchmark/case
