# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Transformer Engine GEMM microbenchmark for GB200 (arm64 / Grace).
# Built on an NGC PyTorch base that already ships a Blackwell-ready TE + CUDA toolchain
# rather than compiling TE from scratch.

# arm64/sbsa NGC PyTorch (Grace-Blackwell ready). 25.04+ carries TE >= 2.16, CUDA 12.8+.
# Do NOT swap to an x86_64 base -- GB200 hosts are ARM (the B200/B300 HGX distinction).
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN pip install --no-cache-dir "transformer-engine[pytorch]>=2.16" || true   # usually already present in NGC

# The benchmark script lives in the TE source tree (benchmarks/gemm/benchmark_gemm.py).
# Vendor a copy of our wrappers; benchmark_gemm.py is resolved from the installed TE.
WORKDIR /opt/te-gemm
COPY run_gemm_bench.sh roofline.py /opt/te-gemm/
RUN chmod +x /opt/te-gemm/run_gemm_bench.sh

# CUTLASS narrow-precision GEMM example (optional, sm_100a) for a cuBLAS-independent check.
ARG CUTLASS_REF=v3.8.0
RUN git clone --depth 1 --branch ${CUTLASS_REF} https://github.com/NVIDIA/cutlass.git /opt/cutlass || true

ENV TE_GEMM_HOME=/opt/te-gemm
CMD ["/opt/te-gemm/run_gemm_bench.sh"]
