#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Launch the Grace-offload training variant on GB200.
#   VARIANT=superoffload -> DeepSpeed ZeRO-3 + super_offload to Grace memory
#   VARIANT=fsdp         -> PyTorch FSDP CPUOffload(offload_params=True) baseline
set -euo pipefail
: "${VARIANT:=superoffload}"
: "${MODEL:=meta-llama/Llama-3.1-8B}"

# Prefetch hint: pull hot pages to HBM ahead of use to hide C2C page-fault latency.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$VARIANT" in
  superoffload)
    echo "GB200 Grace offload :: DeepSpeed SuperOffload ($MODEL)"
    exec deepspeed --num_gpus 4 train_offload.py \
      --deepspeed --deepspeed_config /opt/grace-offload/ds_config_superoffload.json \
      --model "$MODEL"
    ;;
  fsdp)
    echo "GB200 Grace offload :: FSDP CPUOffload ($MODEL)"
    exec torchrun --nproc_per_node 4 train_offload.py \
      --fsdp --fsdp-cpu-offload --model "$MODEL"
    ;;
  *) echo "Unknown VARIANT=$VARIANT (superoffload|fsdp)"; exit 2 ;;
esac
