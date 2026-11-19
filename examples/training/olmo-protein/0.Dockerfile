# OLMo/OLMoE protein pretraining container (MoE + dense from one image)
# Base: AWS Deep Learning Container for PyTorch. Swap region/account for your
# nearest DLC registry: https://github.com/aws/deep-learning-containers
ARG DLC=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker
FROM ${DLC}

# OLMo fork, Muennighoff/MoE branch — the branch OLMoE was trained with
# (`main` does not support block_type: moe).
RUN git clone --depth 1 --branch Muennighoff/MoE https://github.com/allenai/OLMo.git /opt/OLMo
WORKDIR /opt/OLMo

# Optional: mirror training metrics to MLflow (e.g. SageMaker managed MLflow).
# Adds a no-op-safe mlflow_log() beside each wandb.log call; training runs
# identically if no MLflow run is active.
COPY 4.mlflow-mirroring.patch /tmp/
RUN git apply /tmp/4.mlflow-mirroring.patch

RUN pip install --no-cache-dir -e ".[train]" && \
    pip install --no-cache-dir datasets ninja packaging wheel mlflow sagemaker-mlflow

# megablocks MoE kernels (OLMoE fork). --no-deps prevents it from pulling
# torch 2.3.1 alongside the DLC's torch (doubles image size); stanford-stk is
# its only real dependency.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
RUN pip install --no-cache-dir stanford-stk && \
    pip install --no-cache-dir --no-build-isolation --no-deps \
    git+https://github.com/Muennighoff/megablocks.git@olmoe

# Import check only — megablocks touches CUDA at import, no GPU during build
RUN python -c "import torch; from olmo.config import TrainConfig; print('OK', torch.__version__)"
