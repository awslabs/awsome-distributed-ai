FROM public.ecr.aws/hpc-cloud/nccl-tests:cuda13.0.2-efa1.48.0-ofiv1.19.0-ncclv2.30.4-1-testsv2.18.3
COPY requirements.txt constraints.txt /tmp/
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/constraints.txt
# The base's NCCL binary lacks sm_120 and PTX. Rebuild the same source version
# for the lab GPU families, including RTX PRO Blackwell and A100.
ARG NCCL_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103 -gencode=arch=compute_120,code=sm_120"
ARG NCCL_BUILD_JOBS=32
RUN make -C /opt/nccl src.build BUILDDIR=/tmp/aim347-nccl-build -j${NCCL_BUILD_JOBS} NVCC_GENCODE="${NCCL_GENCODE}" \
    && cp -a /tmp/aim347-nccl-build/lib/libnccl.so* /opt/nccl/build/lib/ \
    && rm -rf /tmp/aim347-nccl-build
# nccl-tests has its own CUDA correctness kernels and needs the same targets.
RUN make -C /opt/nccl-tests -j${NCCL_BUILD_JOBS} BUILDDIR=/tmp/aim347-nccl-tests \
      MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME=/opt/nccl/build NVCC_GENCODE="${NCCL_GENCODE}" \
    && cp /tmp/aim347-nccl-tests/*_perf /opt/nccl-tests/build/ \
    && rm -rf /tmp/aim347-nccl-tests
COPY . /opt/aim347
RUN nvcc -O3 /opt/aim347/lib/gemm.cu -lcublas -o /usr/local/bin/aim347-gemm
RUN nvcc -O3 /opt/aim347/lib/dram.cu -o /usr/local/bin/aim347-dram
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
WORKDIR /opt/aim347
