# DETR-ResNet50 Object Detection Training Container
# Base image: HPC-optimized with pre-configured EFA and NCCL for distributed training
FROM public.ecr.aws/hpc-cloud/nccl-tests:cuda12.8.1-efa1.43.2-ofiv1.16.3-ncclv2.27.7-1-testsv2.16.9

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

WORKDIR /workspace

# Install system dependencies for OpenCV/Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies with pinned versions
# Note: qai-hub-models installed without [detr-resnet50] extra to avoid pulling
# fiftyone (~2GB). We only need the model class + transformers for weights.
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchmetrics==1.6.1 \
    qai-hub-models==0.30.2 \
    transformers==4.51.3 \
    object-detection-metrics==0.4.post1 \
    shapely==2.0.3 \
    Pillow==11.1.0 \
    numpy==1.26.4 \
    timm==1.0.12

# Copy training script
COPY detr_main.py /workspace/detr_main.py

# Create working directories
RUN mkdir -p /workspace/data \
    /workspace/outputs \
    /workspace/checkpoints \
    /workspace/logs

# Pre-download DETR-ResNet50 weights so the container works in air-gapped environments.
# The weights originate from facebook/detr-resnet-50 on HuggingFace Hub; the QAI Hub
# Model.from_pretrained() wrapper downloads them via the transformers library.
# NOTE: This step requires internet access during docker build.
ENV HF_HOME=/workspace/cache/huggingface
RUN mkdir -p /workspace/cache/huggingface && \
    python3 -c "from qai_hub_models.models.detr_resnet50 import Model; m = Model.from_pretrained(); print('DETR-ResNet50 weights cached')"

# Verify installations
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "from qai_hub_models.models.detr_resnet50 import Model; print('QAI Hub DETR-ResNet50 model class loaded')" && \
    python3 -c "import torchmetrics; print(f'torchmetrics version: {torchmetrics.__version__}')"

CMD ["python3", "detr_main.py", "--help"]
