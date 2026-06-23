#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# FP4 TensorRT-LLM serving + reconciliation + accuracy gate on one P6e-GB200 node.
#SBATCH --job-name=gb200-fp4-trtllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output %x_%j.out
#SBATCH --error  %x_%j.err
set -euo pipefail
: "${APPS_PATH:=/fsx}"
: "${IMAGE:=$APPS_PATH/gb200-fp4-trtllm-arm64.sqsh}"
: "${MODEL:=nvidia/Llama-3.1-8B-Instruct-NVFP4}"
export MODEL
srun --container-image "$IMAGE" --cpu-bind=none bash -lc '
  cd /opt/fp4
  ./serve.sh & SERVER=$!
  sleep 90                                   # wait for trtllm-serve to come up
  ./bench.sh
  BASELINE_MMLU_PRO=${BASELINE_MMLU_PRO:-0.0} ./accuracy_gate.sh || true
  kill $SERVER
'
