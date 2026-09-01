FROM public.ecr.aws/hpc-cloud/nccl-tests:latest

RUN apt update && apt install -y nvtop

RUN pip install torch==2.9.1 numpy torchvision==0.24.1 pillow datasets huggingface-hub==0.36.1 transformers==4.57.3 wandb einops accelerate loguru lmms_eval sagemaker-mlflow

RUN mkdir -p /nanoVLM 
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY nanoVLM/ /nanoVLM/

WORKDIR /nanoVLM
