# OLMo-core protein pretraining container (MoE + FLOP-matched dense from one image)
#
# Base: AWS Deep Learning Container whose PyTorch (2.10) satisfies OLMo-core's
# declared range (torch>=2.10,<2.14), so pip does NOT replace the DLC's
# PyTorch/CUDA/NCCL stack. Swap region/account for your nearest DLC registry:
# https://github.com/aws/deep-learning-containers
ARG DLC=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker
FROM ${DLC}

# OLMo-core pinned to the tag this recipe was validated against. The pin also
# fixes the training script itself — the recipe runs OLMo-core's own
# src/examples/llm/train.py from this checkout.
ARG OLMO_CORE_REF=v3.0.0
RUN git clone --depth 1 --branch ${OLMO_CORE_REF} https://github.com/allenai/OLMo-core.git /opt/OLMo-core
WORKDIR /opt/OLMo-core
RUN pip install --no-cache-dir -e .

# grouped_gemm is required here, not optional: the recipe's full-size model is
# the MoE, and without grouped_gemm OLMo-core warns the MoE "will be
# substantially slower". A build that cannot compile it should fail visibly.
#
# Install it exactly the way OLMo-core's own image does (src/Dockerfile at
# v3.0.0): a pinned git SHA with GROUPED_GEMM_CUTLASS=1. OLMo-core passes
# batch_sizes on GPU, which needs the CUTLASS backend — the PyPI release
# builds the cuBLAS path and fails at runtime with "Expected
# batch_sizes.is_cpu() to be true". See tgale96/grouped_gemm#21.
#
# TORCH_CUDA_ARCH_LIST must be set because there is no GPU to probe during an
# image build (A10G 8.6, A100 8.0, H100 9.0; trim to your fleet to shorten
# the build). --no-build-isolation compiles against the image's PyTorch.
ARG GROUPED_GEMM_SHA="f1429a3c44c98f7912aa4b00125144cdf4e7fdb2"
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" GROUPED_GEMM_CUTLASS="1" \
    pip install --no-build-isolation --no-cache-dir \
    "grouped_gemm @ git+https://github.com/tgale96/grouped_gemm.git@${GROUPED_GEMM_SHA}"

COPY 1.prepare_uniref.py protein_overrides.sh /opt/olmo-protein/

# Import check (CPU-safe) — prints which PyTorch the image actually ends up with.
RUN python -c "import torch; from olmo_core.version import VERSION; from olmo_core.nn.transformer import TransformerConfig; print('OK olmo-core', VERSION, '| torch', torch.__version__)"
