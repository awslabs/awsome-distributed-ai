# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
FROM public.ecr.aws/hpc-cloud/nccl-tests@sha256:a5390d3f0eb50f3e5854085e2ae0fef90b9eb4f3b7486ed4cab0d47864809d4c AS nccl-baseline
# Rebuild the pinned source versions for A100 and the Blackwell rehearsal GPUs.
# The published base has no sm_120 kernels or PTX fallback.
ARG NCCL_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103 -gencode=arch=compute_120,code=sm_120"
ARG NCCL_BUILD_JOBS=32
RUN make -C /opt/nccl src.build BUILDDIR=/tmp/aim344-nccl-build -j${NCCL_BUILD_JOBS} NVCC_GENCODE="${NCCL_GENCODE}" \
    && cp -a /tmp/aim344-nccl-build/lib/libnccl.so* /opt/nccl/build/lib/ \
    && rm -rf /tmp/aim344-nccl-build
RUN make -C /opt/nccl-tests -j${NCCL_BUILD_JOBS} BUILDDIR=/tmp/aim344-nccl-tests \
      MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME=/opt/nccl/build NVCC_GENCODE="${NCCL_GENCODE}" \
    && cp /tmp/aim344-nccl-tests/*_perf /opt/nccl-tests/build/ \
    && rm -rf /tmp/aim344-nccl-tests

FROM nccl-baseline AS aim344
COPY requirements.txt /opt/aim344-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /opt/aim344-requirements.txt
COPY . /opt/aim344/
WORKDIR /opt/aim344
