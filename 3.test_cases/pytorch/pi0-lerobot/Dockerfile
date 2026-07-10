# Pi0 (LeRobot) — Training Container for HyperPod EKS
#
# Bakes the LeRobot π0 fine-tuning environment into a container image so
# HyperPod EKS can launch training jobs without runtime setup.
#
# Build (from this directory):
#   docker buildx build --platform linux/amd64 \
#     --build-arg AWS_REGION=${AWS_REGION} \
#     --load -f Dockerfile -t ${REGISTRY}pi0-lerobot:${IMAGE_TAG} .
#
# Push to ECR:
#   aws ecr get-login-password --region ${AWS_REGION} \
#     | docker login --username AWS --password-stdin ${REGISTRY}
#   docker image push ${REGISTRY}pi0-lerobot:${IMAGE_TAG}

ARG AWS_REGION=us-east-2
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker-v1.9

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ──
RUN apt-get update -qq && \
    apt-get install -y -qq curl ffmpeg libx11-dev libgl1 libglib2.0-0 git-lfs tree wget && \
    rm -rf /var/lib/apt/lists/*

# ── Install LeRobot with π0 extras ──
# Disable git-lfs smudge to avoid downloading large sample files during pip install
ENV GIT_LFS_SKIP_SMUDGE=1
RUN git config --global filter.lfs.clean cat && \
    git config --global filter.lfs.smudge cat && \
    git config --global filter.lfs.process "" && \
    git config --global filter.lfs.required false

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "lerobot[pi,dataset] @ git+https://github.com/huggingface/lerobot.git@v0.5.0" \
        accelerate \
        safetensors \
        huggingface-hub

# ── Sanity check: imports succeed ──
RUN python -c "import lerobot; import accelerate; import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
RUN python -c "from lerobot.policies.pi0 import PI0Policy; print('PI0Policy import OK')"

# ── Copy entrypoint script ──
COPY src/ /opt/pi0-lerobot/src/
RUN chmod +x /opt/pi0-lerobot/src/*.sh

# ── Patch LeRobot: skip Hub lookups for local datasets ──
# The training script handles this at runtime via the lerobot_local_patch,
# but we pre-copy the patch script into the image for convenience.
COPY src/lerobot_local_patch.py /opt/pi0-lerobot/src/

# ── Environment defaults ──
ENV HF_HOME=/data/hf-cache
ENV HF_LEROBOT_HOME=/data/lerobot-cache
ENV NCCL_DEBUG=WARN
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /opt/pi0-lerobot
ENTRYPOINT ["/bin/bash"]
