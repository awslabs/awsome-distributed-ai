FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training-neuronx:2.9.0-neuronx-py312-sdk2.28.0-ubuntu24.04

RUN git clone --branch v2.27.0 --depth 1 https://github.com/aws-neuron/neuronx-distributed.git
COPY ./src /workspace
RUN cp -r neuronx-distributed/examples/training/llama/* workspace/
RUN cp -r neuronx-distributed/examples/training/llama/tp_zero1_llama_hf_pretrain/* workspace/
RUN cp -r neuronx-distributed/examples/training/llama/tp_zero1_llama_hf_pretrain/8B_config_llama3.1 workspace/config_8b_llama3.1
RUN cp -r neuronx-distributed/examples/training/llama/tp_zero1_llama_hf_pretrain/8B_config_llama3 workspace/config_8b_llama3
RUN mv workspace/tp_zero1_llama_hf_pretrain.py workspace/train.py

WORKDIR /workspace

RUN pip install -r requirements.txt
