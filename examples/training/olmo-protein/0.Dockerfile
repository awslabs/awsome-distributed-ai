# OLMo-core protein pretraining container (MoE + FLOP-matched dense from one image)
# Base: AWS Deep Learning Container for PyTorch. Swap region/account for your
# nearest DLC registry: https://github.com/aws/deep-learning-containers
ARG DLC=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker
FROM ${DLC}

# OLMo-core — AI2's actively developed training framework (successor to
# allenai/OLMo). Native MoE support: no megablocks, no custom kernel builds.
RUN git clone --depth 1 https://github.com/allenai/OLMo-core.git /opt/OLMo-core
WORKDIR /opt/OLMo-core
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir grouped_gemm || echo "grouped_gemm optional — MoE falls back to a slower loop without it"

COPY 1.build_tokenizer.py 2.prepare_uniref.py 3.train.py /opt/olmo-protein/

# Import check (CPU-safe)
RUN python -c "from olmo_core.version import VERSION; from olmo_core.nn.transformer import TransformerConfig; print('OK', VERSION)"
