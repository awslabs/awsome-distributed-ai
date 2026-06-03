FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

RUN apt update && apt upgrade -y

RUN pip install --break-system-packages mlflow==2.13.2 sagemaker-mlflow==0.1.0
COPY ddp.py /workspace

