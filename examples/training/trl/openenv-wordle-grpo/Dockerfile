# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

FROM public.ecr.aws/hpc-cloud/nccl-tests:latest@sha256:4498c1fa9e76983dad9d7a0a5d0002652ffe55dd4d18593d5bd66119d3ee3291

RUN apt update && apt install -y nvtop

RUN mkdir -p /wordle-grpo /checkpoints
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Miniconda (Python 3.12)
RUN mkdir -p /opt/miniconda3 \
    && curl -L https://repo.anaconda.com/miniconda/Miniconda3-py312_25.3.1-1-Linux-x86_64.sh -o /tmp/Miniconda3.sh \
    && bash /tmp/Miniconda3.sh -b -f -p /opt/miniconda3 \
    && rm /tmp/Miniconda3.sh \
    && /opt/miniconda3/bin/conda init bash
ENV PATH="/opt/miniconda3/bin:${PATH}"

# Install Rust (required by TRL's 'outlines' dependency)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY src/ /wordle-grpo/
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /wordle-grpo/requirements.txt

RUN pip install --no-cache-dir --no-deps 'trl==0.26.2'

RUN pip install --no-cache-dir --no-deps mergekit immutables==0.21

# Fix GRPOTrainer compatibility with vLLM 0.8.x (logprobs_mode kwarg not supported).
# This can be removed once vLLM is upgraded to >=0.10.2.
RUN python -c "\
import pathlib; \
p = pathlib.Path('/opt/miniconda3/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py'); \
src = p.read_text(); \
src = src.replace('logprobs_mode=\"processed_logprobs\",\n', ''); \
p.write_text(src)"

# Verify the TRL patch was applied successfully (fail the build if it wasn't)
RUN python -c "\
import pathlib; \
p = pathlib.Path('/opt/miniconda3/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py'); \
src = p.read_text(); \
assert 'logprobs_mode=\"processed_logprobs\"' not in src, \
    'ERROR: TRL patch failed to apply. The logprobs_mode kwarg is still present in grpo_trainer.py'"

# Flash Attention (requires torch, must install separately with --no-build-isolation)
RUN pip install "flash-attn>=2.5.0,<3.0" --no-build-isolation

# FlashInfer for optimized inference
RUN pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/

WORKDIR /wordle-grpo
