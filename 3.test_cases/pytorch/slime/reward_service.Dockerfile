# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# CPU-only image for the SLIME remote reward service. Deliberately lightweight
# (no CUDA / NGC base) so it schedules on cheap CPU instances and starts fast.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /opt/reward_service

# Build tools needed by sentencepiece/tokenizers wheels on slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY reward_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY reward_service/app.py /opt/reward_service/app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
