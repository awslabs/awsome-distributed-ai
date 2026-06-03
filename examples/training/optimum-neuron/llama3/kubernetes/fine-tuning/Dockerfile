FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04

# Set pip repository pointing to the Neuron repository
RUN python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Upgrade optimum-neuron to 0.4.5 with training extras (trl, peft, accelerate)
RUN python3 -m pip install --upgrade 'optimum-neuron[training]==0.4.5'

RUN mkdir workspace

COPY src workspace

WORKDIR /workspace
